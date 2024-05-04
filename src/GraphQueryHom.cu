#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "DeviceGraph.cuh"
// #include "ops/Search.cuh"

namespace graph_query {

  namespace cg = cooperative_groups;
  // vertex parallel triangle query
  __global__ void triangleQuery(DeviceGraph dataGraph) {
    const auto thisGrid = cg::this_grid();
    const auto thisBlock = cg::this_thread_block();
    const auto blockId = thisGrid.block_rank();
    const auto thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
    const auto thrId = thisGrid.thread_rank();
    const auto warpId = thrId / WARP_SIZE;
    const auto warpNum = thisGrid.num_threads() / WARP_SIZE;

    // for (vidT v0 : dataGraph.V(thisGrid))
    for (size_t v0 = warpId; v0 < dataGraph.numV(); v0 += warpNum) {
      for (const auto v1 : dataGraph.N(v0)) {
        // intersection
      }
    }
  }

} // namespace graph_query
