#pragma once
#include <cstdint>
#include <type_traits>
#include <cooperative_groups.h>
#include "Defines.h"
#include "ops/DeviceIterator.cuh"

namespace cg = cooperative_groups;

// thread-sequential linear search
template <typename T> __forceinline__ HOST_DEVICE int
linear_search(Span<T*, void> span, T key) {
  for (auto val: span) {
    if (val >= key) return val == key; // sorted array
  }
  return false;
}

// warp cooperative linear search
template <typename T> __forceinline__ DEVICE_ONLY int
linear_search(Span<T*, cg::thread_warp> span, T key) {
  for (auto iter = span.begin(); ; ++iter) {
    bool valid = (iter != span.end() && *iter <= key);
    bool found = (iter != span.end() && *iter == key);
    int found_key = span.group.any(found);
    int stop = span.group.any(!valid);
    if (stop || found_key) return found_key;
  }
  return false;
}

// thread-sequential binary search
template <typename T> __forceinline__ HOST_DEVICE bool
binary_search(T *list, T key, T size) {
  int l = 0;
  int r = size-1;
  while (r >= l) { 
    int mid = l + (r - l) / 2; 
    T value = list[mid];
    if (value == key) return true;
    if (value < key) l = mid + 1;
    else r = mid - 1;
  }
  return false;
}

// thread-sequential binary search
template <class Indexable, typename T = typename Indexable::value_type> __forceinline__ HOST_DEVICE bool
binary_search(Indexable span, T key) {
  int l = 0;
  int r = (int)span.size() - 1;

  while (r >= l) { 
    int mid = l + (r - l) / 2;
    T value = span[mid];
    if (value == key) return true;
    if (value < key) l = mid + 1;
    else r = mid - 1;
  }
  return false;
}

// thread-sequential binary search
//
// check if `key` is in `span` and set the position of the found element in `pos`;
// if `key` not found, `pos` is set to the position where `key` should be inserted after
template <class Indexable, typename T = typename Indexable::value_type> __forceinline__ HOST_DEVICE bool
binary_search(Indexable span, T key, int &pos) {
  int l = 0;
  int r = (int)span.size() - 1;

  while (r >= l) { 
    pos = l + (r - l) / 2;
    T value = span[pos];
    if (value == key) return true;
    if (value < key) l = pos + 1;
    else r = pos - 1;
  }
  pos = r;
  return false;
}

// TODO: is galloping search efficient on GPU?
// thread-sequential gallop search from the given `start` position
// if `key` if found, returns true and `start` is updated to the location of the `key`.
// otherwise, returns false, and `start` is updated to the end of span
// template <typename T>
// __forceinline__ __device__ bool gallop_search(
//   Span<T*, void> span, T key, typename Span<T*, void>::IterType &start
// );

// warp-level binary search
template <typename T> __forceinline__ DEVICE_ONLY bool
binary_search_2phase(T *list, T *cache, T key, T size) {
  // int p = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  auto thisWarp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  int p = thisWarp.meta_group_rank() * WARP_SIZE;

  int mid = 0;
  // phase 1: search in the cache
  int bottom = 0;
  int top = WARP_SIZE;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    T y = cache[p + mid];
    if (key == y) return true;
    if (key < y) top = mid;
    if (key > y) bottom = mid;
  }

  //phase 2: search in global memory
  bottom = bottom * size / WARP_SIZE;
  top = top * size / WARP_SIZE - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    T y = list[mid];
    if (key == y) return true;
    if (key < y) top = mid - 1;
    else bottom = mid + 1;
  }
  return false;
}

// initialize the indexCache array from list
template <typename T, size_t N, typename S=std::remove_cv_t<T>> inline HOST_DEVICE void
build_cache(Span<T*, void> list, Array<S, N> &indexCache) {
  const auto size = (int)list.size();
  for (size_t i = 0; i < N; i += 1) {
    indexCache[i] = list[i * (size-1) / (indexCache.size()-1)];
  }
}

// initialize the indexCache array from list cooperatively
template <typename T, size_t N, typename S=std::remove_cv_t<T>> inline DEVICE_ONLY void
build_cache(Span<T*, void> list, Array<S, N> &indexCache, const cg::thread_group &group) {
  const auto size = (int)list.size();
  for (auto i : IdRange<int, decltype(group)>(0, N, group))
    indexCache[i] = list[i * (size-1) / (indexCache.size()-1)];
}

// thread-sequential binary search with a shared index cache
// *indexCache.begin() should be the first item in list
// *(indexCache.end() - 1) should be the last item in list
template <typename T, size_t N, typename S=std::remove_cv_t<T>> inline HOST_DEVICE bool
binary_search_2phase(Span<T*, void> list, const Array<S, N> &indexCache, T key) {
  const auto cacheSize = (int)indexCache.size();

  // phase 1: search in the cache
  int pos = 0;
  bool foundInCache = binary_search<decltype(indexCache)>(indexCache, key, pos);
  if (foundInCache) {
    return true;
  }
  // only when key is not found in cache,
  // we could use "pos >= cacheSize - 1" to decide key is not in the list
  // if (pos < 0 || pos >= cacheSize - 1) {
  //   return false;
  // }
  pos = (pos < 0) ? 0 : pos;
  pos = (pos >= cacheSize - 1) ? cacheSize - 2 : pos;

  //phase 2: search in global memory
  const auto size = (int) list.size();
  int bottom = pos * (size - 1) / (cacheSize - 1);
  int top = (pos + 1) * (size - 1) / (cacheSize - 1);
  Span<T*, void> sublist {list.begin() + bottom, (size_t)(top - bottom) };
  return binary_search<decltype(sublist)>(sublist, key);
}
