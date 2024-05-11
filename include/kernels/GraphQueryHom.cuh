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

    const auto &thisGrid = cg::this_grid();
    const auto &thisBlock = cg::this_thread_block();
    const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
    const auto warpLane = subGroupId(thisWarp, thisBlock);
    const int lane = thisWarp.thread_rank();

    DeviceCOO coo = dataGraph.coo;
    DeviceCSR csr = dataGraph.csr;
    Array<vidT, N> cache {smem + N * warpLane};

    for (eidT eid : coo.E().cooperate(thisWarp, thisGrid)) {
      vidT src = coo.rowIdx[eid];
      vidT dst = coo.colIdx[eid];
      Span<vidT, void> srcAdj {csr.colIdx + csr.indPtr[src], (size_t) csr.getDegree(src)};
      Span<vidT, void> dstAdj {csr.colIdx + csr.indPtr[dst], (size_t) csr.getDegree(dst)}; 
      int num = 0;
      if constexpr (useCache) {
        num = intersectBinarySearch(srcAdj, dstAdj, cache);
      } else {
        num = intersectBinarySearch(srcAdj, dstAdj);
      }
      countAtomic.fetch_add(num, cuda::std::memory_order_relaxed);
    }
  }

  // legacy implementation of triangle counting
  __global__ void triangleCountLegacy(DeviceGraph dataGraph, unsigned *count);

  __global__ void rectCountHom(DeviceGraph dataGraph, unsigned *count);
}
