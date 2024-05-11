#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "ops/Intersection.cuh"
#include "kernels/GraphQueryHom.cuh"

namespace graph_query {

  constexpr int BLOCK_SIZE = 256;
  typedef cub::BlockReduce<unsigned, BLOCK_SIZE> BlockReduceU32;

  size_t getIndexCacheSMem(dim3 blockDim, size_t cacheSize) {
    return warpsPerBlock(blockDim) * cacheSize;
  }

  __global__ void __launch_bounds__(BLOCK_SIZE, 8)
  triangleCountLegacy(DeviceGraph dataGraph, unsigned *total) {
    __shared__ typename BlockReduceU32::TempStorage temp_storage;
    __shared__ vidT cache[BLOCK_SIZE];
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int warp_id   = thread_id   / WARP_SIZE;               // global warp index
    int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
    unsigned count = 0;
    auto coo = dataGraph.coo;
    auto csr = dataGraph.csr;
    eidT ne = dataGraph.coo.numE();
    for (eidT eid = warp_id; eid < ne; eid += num_warps) {
      vidT v = coo.rowIdx[eid];
      vidT u = coo.colIdx[eid];
      vidT v_size = csr.getDegree(v);
      vidT u_size = csr.getDegree(u);
      count += intersect_bs_cache(
        csr.colIdx + csr.indPtr[v], v_size,
        csr.colIdx + csr.indPtr[u], u_size, cache
      );
    }
    unsigned block_num = BlockReduceU32(temp_storage).Sum(count);
    if (threadIdx.x == 0) atomicAdd(total, block_num);
  }

} // namespace graph_query
