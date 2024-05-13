#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/std/atomic>
#include "DeviceGraph.cuh"
#include "ops/DeviceIterator.cuh"
#include "ops/Intersection.cuh"

namespace graph_query {

  namespace cg = cooperative_groups;

  size_t getIndexCacheSMem(dim3 blockDim, size_t cacheSize);

  template <bool useCache=true, size_t N=32, WorkPartition Policy=interleavedPolicy>
  __global__ void triangleCountHom(DeviceGraph dataGraph, size_t *count) {
    extern __shared__ vidT smem[];
    auto countAtomic = cuda::std::atomic_ref<size_t>(*count);
    DeviceCOO coo = dataGraph.coo;
    DeviceCSR csr = dataGraph.csr;

    const auto &thisGrid = cg::this_grid();
    const auto &thisBlock = cg::this_thread_block();
    const auto &thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
    const auto warpLane = thisWarp.meta_group_rank();
    const int lane = thisWarp.thread_rank();

    unsigned threadCount = 0;

    Array<vidT, N> cache {smem + N * warpLane};
    for (eidT eid : coo.E().cooperate<Policy>(thisWarp, thisGrid)) {
      vidT src = coo.rowIdx[eid];
      vidT dst = coo.colIdx[eid];
      Span<vidT, void> sA {csr.colIdx + csr.indPtr[src], (size_t) csr.getDegree(src)};
      Span<vidT, void> sB {csr.colIdx + csr.indPtr[dst], (size_t) csr.getDegree(dst)};
      if constexpr (useCache) {
        threadCount += intersect(sA, sB, cache);
      } else {
        threadCount += intersect(sA, sB);
      }
    }
    countAtomic.fetch_add(threadCount, cuda::std::memory_order_relaxed);
  }

  // legacy implementation of triangle counting
  __global__ void triangleCountLegacy(DeviceGraph dataGraph, unsigned *count);

  __global__ void rectCountHom(DeviceGraph dataGraph, unsigned *count);
}
