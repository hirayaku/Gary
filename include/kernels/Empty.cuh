#pragma once
#include <type_traits>
#include <cooperative_groups.h>
#include "Defines.h"
#include "ops/DeviceIterator.cuh"

namespace cg = cooperative_groups;

__global__ void emptyKernel() {
  const auto thisGrid = cg::this_grid();
  const auto numBlocks = thisGrid.num_blocks();
  const auto blockId = thisGrid.block_rank();
  const auto thisBlock = cg::this_thread_block();
  const auto blockSize = thisBlock.size();
  const auto thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto warpId = thisBlock.thread_rank() / WARP_SIZE;
  const auto threadIdBlock = thisBlock.thread_rank();

  IdRange<int, void> range(0, 65);
  // BCastSpan<int, decltype(thisWarp), decltype(thisGrid), decltype(range.begin())> span {
  //   range.begin(), range.size(), thisWarp, thisGrid};
  for (auto i : range.cooperate(thisWarp, thisGrid)) {
    printf("(%llu, %u): %d \n", blockId, threadIdBlock, i);
  }
}
