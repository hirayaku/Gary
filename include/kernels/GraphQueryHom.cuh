#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/std/atomic>
#include "DeviceGraph.cuh"
#include "ops/Intersection.cuh"

namespace graph_query {

  namespace cg = cooperative_groups;

  size_t getIndexCacheSMem(dim3 blockDim, size_t cacheSize);

  template <bool useCache=true, size_t N=32>
  __global__ void triangleCountHom(DeviceGraph dataGraph, unsigned *count) {
    extern __shared__ vidT smem[];
    auto countAtomic = cuda::std::atomic_ref<unsigned>(*count);
    DeviceCOO coo = dataGraph.coo;
    DeviceCSR csr = dataGraph.csr;

    const auto &thisGrid = cg::this_grid();
    const auto &thisBlock = cg::this_thread_block();
    const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
    const auto warpLane = subGroupId(thisWarp, thisBlock);
    const int lane = thisWarp.thread_rank();

    unsigned threadCount = 0;

    Array<vidT, N> cache {smem + N * warpLane};
    for (eidT eid : coo.E().cooperate(thisWarp, thisGrid)) {
      vidT src = coo.rowIdx[eid];
      vidT dst = coo.colIdx[eid];
      Span<vidT, void> srcAdj {csr.colIdx + csr.indPtr[src], (size_t) csr.getDegree(src)};
      Span<vidT, void> dstAdj {csr.colIdx + csr.indPtr[dst], (size_t) csr.getDegree(dst)}; 
      if constexpr (useCache) {
        threadCount += intersectBinarySearch(srcAdj, dstAdj, cache);
      } else {
        threadCount += intersectBinarySearch(srcAdj, dstAdj);
      }
    }
    countAtomic.fetch_add(threadCount, cuda::std::memory_order_relaxed);

    // auto countAtomic = cuda::std::atomic_ref<unsigned>(*count);
    // __shared__ vidT cache[256];
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    // int warp_id   = thread_id   / WARP_SIZE;               // global warp index
    // int num_warps = (256 / WARP_SIZE) * gridDim.x;  // total number of active warps
    // unsigned tcount = 0;
    // auto coo = dataGraph.coo;
    // auto csr = dataGraph.csr;
    // eidT ne = dataGraph.coo.numE();
    // for (eidT eid = warp_id; eid < ne; eid += num_warps) {
    //   vidT v = coo.rowIdx[eid];
    //   vidT u = coo.colIdx[eid];
    //   vidT v_size = csr.getDegree(v);
    //   vidT u_size = csr.getDegree(u);
    //   tcount += intersect_bs_cache(
    //     csr.colIdx + csr.indPtr[v], v_size,
    //     csr.colIdx + csr.indPtr[u], u_size, cache
    //   );
    // }
    // countAtomic.fetch_add(tcount, cuda::std::memory_order_relaxed);
  }

  // legacy implementation of triangle counting
  __global__ void triangleCountLegacy(DeviceGraph dataGraph, unsigned *count);

  __global__ void rectCountHom(DeviceGraph dataGraph, unsigned *count);
}
