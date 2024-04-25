#pragma once

#include <assert.h>
#include <stdint.h>
#include <type_traits>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cub/cub.cuh>
#include "../Defines.h"

namespace cg = cooperative_groups;

template <typename SrcT, typename DstT>
size_t segmentKernelSMem(dim3 blockDim) {
  const auto blockSize = blockDim.x * blockDim.y * blockDim.z;
  // if DstSize is multiples of 4, PerThread = 1
  // if 4 is multiples of DstSize, PerThread = 4 / DstSize
  // if neither, PerThread = 4 * DstSize // TODO: should be lcm(4, DstSize)
  // in short, we make sure PerThread * DstSize is a multiple of 4
  constexpr int PerThread = (sizeof(DstT) / 4  * 4 == sizeof(DstT))? 1 : (
    4 / sizeof(DstT) * sizeof(DstT) == 4 ? 4 / sizeof(DstT) : 4 * sizeof(DstT)
  );
  return blockSize * PerThread * sizeof(SrcT) * 2;
}

// NOTE: we require sizeof(SrcT) to a multiple of sizeof(DstT)
template <typename SrcT=uint32_t, typename DstT=uint8_t,
          typename = std::enable_if_t<(sizeof(SrcT) > sizeof(DstT))>>
__global__ void segmentKernel(SrcT *array, size_t arraySize, DstT **segments) {
  extern __shared__ char sMem[];

  const auto numBlocks = cg::this_grid().num_blocks();
  const auto blockId = cg::this_grid().block_rank();
  const auto thisBlock = cg::this_thread_block();
  const auto blockSize = thisBlock.size();
  const auto threadId = cg::this_grid().thread_rank();
  const auto threadIdBlock = thisBlock.thread_rank();

  constexpr int SegN = sizeof(SrcT) / sizeof(DstT);
  // if DstSize is multiples of 4, PerThread = 1
  // if 4 is multiples of DstSize, PerThread = 4 / DstSize
  // if neither, PerThread = 4 * DstSize // TODO: should be lcm(4, DstSize)
  // in short, we make sure PerThread * DstSize is a multiple of 4
  constexpr int PerThread = (sizeof(DstT) / 4  * 4 == sizeof(DstT))? 1 : (
    4 / sizeof(DstT) * sizeof(DstT) == 4 ? 4 / sizeof(DstT) : 4 * sizeof(DstT)
  );
  // number of SrcT to be processed by a thread block
  const auto blockSrcSize = blockSize * PerThread;

  if (numBlocks * blockSize * PerThread < arraySize) {
    if (threadId == 0)
      printf("ERROR: Kernel launch #blocks too small to cover the input\n");
    assert(false);
    return;
  }

  // allocate shared memory
  SrcT *srcMem = (SrcT *)sMem;
  char *segMemPtr = sMem + sizeof(SrcT) * blockSrcSize;
  DstT *segMems[SegN] = {0};
  for (int i = 0; i < SegN; ++i) {
    segMems[i] = (DstT *) (segMemPtr + i * sizeof(DstT) * blockSrcSize);
    // if (threadIdBlock == 0)
    //   printf("0x%08lx\n", segMems[i]);
  }

  const size_t blockOffset = blockId * blockSrcSize;
  const size_t threadOffset = threadId * PerThread;
  size_t copySize = 0;
  if (blockOffset < arraySize) {
    copySize = (blockOffset + blockSrcSize <=  arraySize) ?
      blockSrcSize : arraySize - blockOffset;
  }

  if (blockOffset < arraySize) {
    // bulk load (it's okay to do this inside a branch)
    cg::memcpy_async(thisBlock, srcMem, &array[blockOffset], sizeof(SrcT) * copySize);
    cg::wait(thisBlock);
  }

  if (threadOffset < arraySize) {

    // segment SrcT array to DstT arrays
    for (int i = 0; i < PerThread; ++i) {
      SrcT srcBuf = srcMem[threadIdBlock * PerThread + i];
      DstT *castToDst = (DstT *)(&srcBuf);
      #pragma unroll
      for (int j = 0; j < SegN; ++j) {
        segMems[j][threadIdBlock*PerThread+i] = castToDst[j];
      }
    }

    // DstT dstBuf[SegN][PerThread];
    // for (int i = 0; i < PerThread; ++i) {
    //   SrcT srcBuf = srcMem[threadIdBlock * PerThread + i];
    //   DstT *castToDst = (DstT *)(&srcBuf);
    //   #pragma unroll
    //   for (int j = 0; j < SegN; ++j) {
    //     dstBuf[j][i] = castToDst[j];
    //   }
    // }
    // for (int i = 0; i < SegN; ++i) {
    //   // segMems[i][threadIdBlock * PerThread +: PerThread] <- dstBuf[i]
    //   #pragma unroll
    //   for (int j = 0; j < PerThread; ++j)
    //     segMems[i][threadIdBlock*PerThread+j] = dstBuf[i][j];
    // }

    // for (int i = 0; i < SegN; ++i) {
    //   for (int j = 0; j < PerThread; ++j) {
    //     segments[i][threadOffset+j] = segMems[i][threadIdBlock*PerThread+j];
    //   }
    // }
  }

  if (blockOffset < arraySize) {
    thisBlock.sync();

    for (int i = 0; i < SegN; ++i) {
      // if (threadIdBlock == 0) {
      //   printf("Copy 0x%08lx[%dx%lu] to 0x%08lx\n",
      //     segMems[i], sizeof(DstT), copySize, &segments[i][blockOffset]);
      //   printf("0x%02x 0x%02x\n", segMems[i][0], segMems[i][1]);
      // }
      cg::memcpy_async(thisBlock, &segments[i][blockOffset],
        segMems[i], sizeof(DstT) * copySize
      );
    }
    cg::wait(thisBlock);
  }

}
