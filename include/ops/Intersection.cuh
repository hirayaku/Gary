#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/std/limits>
#include <thrust/swap.h>
#include "Defines.h"
#include "ops/Search.cuh"
#include "ops/DeviceIterator.cuh"

namespace cg = cooperative_groups;

// Intersection of two sorted span using binary seach.
// Returns the size of intersection and appends the results to `out`.
// The results could be left unsorted, which is slightly faster.
// Returns **per-thread** count of the intersection
// (warp-cooperative implementation)
template <typename T = vidT, bool OutSorted=true> inline DEVICE_ONLY int
intersectBinarySearch(Span<T, void> sA, Span<T, void> sB, T *out) {
  if (sA.size() > sB.size()) {
    thrust::swap(sA, sB);
  }

  __shared__ int count[MAX_WARPS_PER_BLOCK];

  const auto &thisBlock = cg::this_thread_block();
  const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto &warpLane = subGroupId(thisWarp, thisBlock);
  int *localCount = count + warpLane;
  if (thisWarp.thread_rank() == 0) *localCount = 0;

  int threadCount = 0;
  bool found = false;
  for (const auto key : sA.cooperate(thisWarp)) {
    if (OutSorted) {
      const auto &coalesced = cg::coalesced_threads();
      found = binary_search(sB, key);
      coalesced.sync();
    } else {
      found = binary_search(sB, key);
    }
    if (found) {
      threadCount++;
      int pos = atomicAdd(localCount, 1);
      out[pos] = key;
    }
  }
  thisWarp.sync();
  return threadCount;
}

// Intersection of two sorted span using binary seach.
// This method only computes the size of results.
// Returns **per-thread** count of the intersection
// (warp-cooperative implementation)
template <typename T = vidT> inline DEVICE_ONLY int
intersectBinarySearch(Span<T, void> sA, Span<T, void> sB) {
  if (sA.size() > sB.size()) {
    thrust::swap(sA, sB);
  }

  const auto &thisBlock = cg::this_thread_block();
  const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);

  int threadCount = 0;
  for (const auto key : sA.cooperate(thisWarp)) {
    threadCount += binary_search(sB, key);
  }
  thisWarp.sync();
  return threadCount;
}

// Intersection of two sorted span using binary seach with a small index cache
// Returns **per-thread** count of the intersection
// (warp-cooperative implementation)
template <typename T = vidT, size_t N, bool OutSorted=true> inline DEVICE_ONLY int
intersectBinarySearch(Span<T, void> sA, Span<T, void> sB, Array<T, N> &cache, T *out) {
  if (sA.size() > sB.size()) {
    thrust::swap(sA, sB);
  }

  const auto &thisBlock = cg::this_thread_block();
  const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto &warpLane = subGroupId(thisWarp, thisBlock);
  __shared__ int count[MAX_WARPS_PER_BLOCK];
  int *localCount = count + warpLane;
  if (thisWarp.thread_rank() == 0) *localCount = 0;

  build_cache(sB, cache, thisWarp);
  thisWarp.sync();

  int threadCount = 0;
  bool found = false;
  for (const auto key : sA.cooperate(thisWarp)) {
    if (OutSorted) {
      // TODO: it doesn't seem to be the correct usage of coalesced_threads()
      // we need a mask for the last iteration
      const auto &coalesced = cg::coalesced_threads();
      found = binary_search_2phase(sB, cache, key);
      coalesced.sync();
    } else {
      found = binary_search_2phase(sB, cache, key);
    }
    if (found) {
      threadCount++;
      int pos = atomicAdd(localCount, 1);
      out[pos] = key;
    }
  }
  thisWarp.sync();
  return threadCount;
}

// Intersection of two sorted span using binary seach with a small index cache.
// This method only computes the size of results.
// Returns **per-thread** count of the intersection
// (warp-cooperative implementation)
template <typename T = vidT, size_t N> inline DEVICE_ONLY int
intersectBinarySearch(Span<T, void> sA, Span<T, void> sB, Array<T, N> &cache) {
  if (sA.size() > sB.size()) {
    thrust::swap(sA, sB);
  }

  const auto &thisBlock = cg::this_thread_block();
  const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  build_cache(sB, cache, thisWarp);
  thisWarp.sync();

  int threadCount = 0;
  for (const auto key : sA.cooperate(thisWarp)) {
    threadCount += binary_search_2phase(sB, cache, key);
  }
  thisWarp.sync();
  return threadCount;
}

// Intersection of sorted spans when the first span `sA` fits into `spad`
// (warp-cooperative implementation)
// NOTE: for this function to generate meaningful speedups, we need
// - async memcpy
// - early termination
template <typename T = vidT, size_t N> inline DEVICE_ONLY int
intersectSmall(Span<T, void> sA, Span<T, void> sB, Array<T, N> &spad) {
  if (sA.size() > sB.size()) {
    thrust::swap(sA, sB);
  }

  int threadCount = 0;
  const auto &thisBlock = cg::this_thread_block();
  const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);

  // initialize spad from sA
  cg::memcpy_async(thisWarp, spad.begin(), sA.begin(), sA.size());
  cg::wait(thisWarp);

  Span<T, void> spadA(spad.begin(), sA.size());
  const T lastA = spadA[sA.size()-1];
  for (const auto key: sB.cooperate(thisWarp)) {
    if (key > lastA) break;
    threadCount += binary_search(spadA, key);
  }
  return threadCount;
}

static constexpr unsigned floorLog2(unsigned x){
    return x == 1 ? 0 : 1+floorLog2(x >> 1);
}

// Intersection of two large sorted spans using merge.
// (warp-cooperative implementation)
template <typename T = vidT, size_t N, bool OutSorted=true> inline DEVICE_ONLY int
intersectMergeLarge(Span<T, void> sA, Span<T, void> sB, Array<T, N> &spad) {
  // const auto &thisBlock = cg::this_thread_block();
  // const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  // int posA = 0, posB = 0;
  // auto begin = reinterpret_cast<std::uintptr_t>(sA.begin()) >> floorLog2(sizeof(T));
  // auto aligned = (begin >> floorLog2(N)) << floorLog2(N);
  return 0;
}

// Take insersection of two sorted span and returns the result size.
// There are several cases:
// - they are both small (fits into spad)
// - one of them is small, the other is large
// - both of them are large but sizes are skewed
// - both of them are large and sizes are relatively even
template <typename T = vidT, size_t N> inline DEVICE_ONLY int
intersect(Span<T, void> sA, Span<T, void> sB, Array<T, N> &spad) {
  if (sA.size() == 0 || sB.size() == 0) return 0;
  if (sA.size() <= N || sB.size() <= N) return intersectSmall(sA, sB, spad);
  return intersectBinarySearch(sA, sB, spad);
}

template <typename T = vidT> inline DEVICE_ONLY int
intersect(Span<T, void> sA, Span<T, void> sB) {
  if (sA.size() == 0 || sB.size() == 0) return 0;
  return intersectBinarySearch(sA, sB);
}

/*
// original implementation
template <typename T = vidT> __forceinline__ DEVICE_ONLY T
intersectBinarySearchOG(T* a, int size_a, T* b, int size_b, T* c) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ int count[MAX_WARPS_PER_BLOCK];
  T* lookup = a;
  T* search = b;
  T lookup_size = size_a;
  T search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }

  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    int found = 0;
    T key = lookup[i]; // each thread picks a vertex as the key
    if (binary_search<T>(search, key, search_size))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}
*/

// warp-wise intersection of two lists using the 2-phase binary seach algorithm with caching
// Returns **per-warp** count of the intersection
template <typename T = int32_t>
__forceinline__ __device__ T intersect_bs_cache(T* a, T size_a, T* b, T size_b, T* c, T* cache) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[MAX_WARPS_PER_BLOCK];
  int32_t *lookup = a;
  int32_t *search = b;
  int32_t lookup_size = size_a;
  int32_t search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  if (thread_lane == 0) count[warp_lane] = 0;
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    int32_t key = lookup[i]; // each thread picks a vertex as the key
    int found = 0;
    if (binary_search_2phase(search, cache, key, search_size))
      found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

// warp-wise intersetion of two lists using the 2-phase binary seach algorithm with caching
// Returns **per-thread** count of the intersection
template <typename T = int32_t>
__forceinline__ __device__ T intersect_bs_cache(T* a, T size_a, T* b, T size_b, T* cache) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  T num = 0;
  int32_t *lookup = a;
  int32_t *search = b;
  int32_t lookup_size = size_a;
  int32_t search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    int32_t key = lookup[i]; // each thread picks a vertex as the key
    if (binary_search_2phase(search, cache, key, search_size))
      num++;
  }
  return num;
}

// warp-wise intersetion of two lists using the merge based algorithm
// template <typename T = int32_t>
// __forceinline__ __device__ T intersect_merge(T* a, T size_a, T* b, T size_b, T* c) {
//   int thread_lane = threadIdx.x & (WARP_SIZE-1);
//   int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the cta
//   __shared__ T cache[BLOCK_SIZE];
//   int p = warp_lane * WARP_SIZE;
//   T count = 0;
//   auto a_off = thread_lane; 
//   auto b_off = thread_lane; 
//   T key_a = -1;
//   if (a_off < size_a) key_a = a[a_off];
//   if (b_off < size_b) cache[threadIdx.x] = b[b_off];
//   else cache[threadIdx.x] = -2;
//   __syncwarp();
//   while (1) {
//     T last_a = SHFL(key_a, WARP_SIZE-1);
//     T last_b = cache[p+WARP_SIZE-1];
//     if (key_a >= 0 && binary_search_enhanced(&cache[p], key_a, WARP_SIZE))
//       count += 1;
//     bool done_a = last_a<0;
//     bool done_b = last_b<0;
//     __syncwarp();
//     if (done_a && done_b) break;
//     if (done_a) last_a = a[size_a-1];
//     if (done_b) last_b = b[size_b-1];
//     if (last_a == last_b) {
//       if (done_a || done_b) break;
//       a_off += WARP_SIZE;
//       b_off += WARP_SIZE;
//       if (a_off < size_a) key_a = a[a_off];
//       else key_a = -1;
//       if (b_off < size_b) cache[threadIdx.x] = b[b_off];
//       else cache[threadIdx.x] = -2;
//     } else if (last_a > last_b) {
//       if (done_b) break;
//       b_off += WARP_SIZE;
//       if (b_off < size_b) cache[threadIdx.x] = b[b_off];
//       else cache[threadIdx.x] = -2;
//     } else {
//       if (done_a) break;
//       a_off += WARP_SIZE;
//       if (a_off < size_a) key_a = a[a_off];
//       else key_a = -1;
//     }
//     __syncwarp();
//   }
//   return count;
// }

// warp-wise intersection using hybrid method (binary search + merge)
template <typename T = int32_t>
__forceinline__ __device__ T intersect(T* a, T size_a, T *b, T size_b, T* c) {
  //if (size_a > ADJ_SIZE_THREASHOLD && size_b > ADJ_SIZE_THREASHOLD)
  //  return intersect_merge(a, size_a, b, size_b, c);
  //else
    return intersect_bs_cache(a, size_a, b, size_b, c);
}

/*
// warp-wise intersection using hybrid method (binary search + merge)
template <typename T = int32_t>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T *b, T size_b) {
  //if (size_a > ADJ_SIZE_THREASHOLD && size_b > ADJ_SIZE_THREASHOLD)
  //  return intersect_num_merge(a, size_a, b, size_b);
  //else
    return intersect_num_bs_cache(a, size_a, b, size_b);
}


// warp-wise intersection with upper bound using 2-phase binary search with caching
template <typename T = int32_t>
__forceinline__ __device__ T intersect_num_bs_cache(T* a, T size_a, T* b, T size_b, T upper_bound) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  T num = 0;
  T *lookup = a;
  T *search = b;
  T lookup_size = size_a;
  T search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i]; // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && binary_search_2phase(search, cache, key, search_size))
      found = 1;
    num += found;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  __syncwarp();
  return num;
}

// warp-wise intersection using hybrid method (binary search + merge)
template <typename T = int32_t>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T *b, T size_b, T upper_bound) {
  //if (size_a > ADJ_SIZE_THREASHOLD && size_b > ADJ_SIZE_THREASHOLD)
  //  return intersect_num_merge(a, size_a, b, size_b, upper_bound);
  //else
    return intersect_num_bs_cache(a, size_a, b, size_b, upper_bound);
}

template <typename T = int32_t>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T* b, T size_b, T upper_bound, T ancestor) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  AccType num = 0;
  auto lookup = a;
  auto search = b;
  auto lookup_size = size_a;
  auto search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i];
    unsigned active = __activemask();
    __syncwarp(active);
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (key != ancestor && is_smaller && binary_search_2phase(search, cache, key, search_size))
      found = 1;
    num += found;
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  return num;
}

template <typename T = int32_t>
__forceinline__ __device__ T intersect_num(T* a, T size_a, T* b, T size_b, T* ancestor, int n) {
  if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  AccType num = 0;
  auto lookup = a;
  auto search = b;
  auto lookup_size = size_a;
  auto search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  __shared__ T cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i];
    bool valid = true;
    for (int j = 0; j < n; j++) {
      if (key == ancestor[j]) {
        valid = false;
        break;
      }
    }
    if (valid && binary_search_2phase(search, cache, key, search_size))
      num += 1;
  }
  return num;
}
*/
