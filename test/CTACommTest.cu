#include <unistd.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include "HelperCUDA.h"
#include "Timer.h"
#include "ops/DeviceIterator.cuh"

__global__ void OneTimeKernel(int iter) {
  __nanosleep(10 * 1000 * 1000);
  printf("One Time Kernel Finished, %d\n", iter);
}

__global__ void PersistentKernel(volatile int *signal) {
  const auto blockRank = cooperative_groups::this_grid().block_rank();
  int iter = 0;
  while (*signal) {
    __nanosleep(10 * 1000 * 1000);
    iter++;
    if (iter % 10 == 0) {
        printf("Persistent kernel is running, %d\n", iter);
    }
  }
}

__global__ void LoopingKernel(volatile int *signal) {
  const auto blockRank = cooperative_groups::this_grid().block_rank();
  size_t cycles = 0;
  while (signal[blockRank]) {
    cycles += 1;
  }
  printf("%llu: %llu cycles\n", blockRank, cycles);
}

__global__ void PingPongKernel(volatile int *pingpong) {
  const auto blockRank = cooperative_groups::this_grid().block_rank();
  int64_t loops = 0;
  int smid = 0;
  asm("mov.u32 %0, %smid;" : "=r"(smid) );
  printf("%llu: %d\n", blockRank, smid);
  if (blockRank == 0) {
    loops = clock64();
    while (*pingpong == 0);
    *pingpong = 0;
    loops = clock64() - loops;
    printf("[%llu] Loops: %lld\n", blockRank, loops);
  }
  if (blockRank == 1) {
    loops = clock64();
    *pingpong = 1;
    while (*pingpong == 1);
    loops = clock64() - loops;
    printf("[%llu] Loops: %lld\n", blockRank, loops);
  }
}

TEST(CTACommTest, Looping) {
  constexpr int numBlocks = 16;
  thrust::universal_vector<int> signals(numBlocks, 1);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  checkCudaErrors(cudaGetLastError());

  LoopingKernel<<<numBlocks, 1, 0, stream>>>((int *)thrust::raw_pointer_cast(signals.data()));
  for (int i = 0; i < numBlocks; ++i) {
    usleep((int)1e5);
    signals[i] = 0;
  }

  cudaStreamSynchronize(stream);
  checkCudaErrors(cudaGetLastError());
  GTEST_LOG_(INFO) << "looping kernel exited";
  // NOTE: Running LoopingKernel for 1 sec causes the on-device counter to reach ~9M.
  // This translates to 9e6 rps to the unified memory over the bus.
}

TEST(CTACommTest, PingPong) {
  constexpr int numBlocks = 16;
  thrust::device_vector<int> signals(1, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  checkCudaErrors(cudaGetLastError());

  PingPongKernel<<<numBlocks, 1, 0, stream>>>((int *)thrust::raw_pointer_cast(signals.data()));
  cudaStreamSynchronize(stream);
  checkCudaErrors(cudaGetLastError());
}

TEST(CTACommTest, Persistent) {
    cudaStream_t stream, stream_background;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_background, cudaStreamNonBlocking);
    checkCudaErrors(cudaGetLastError());
    volatile int *signal;
    cudaMallocManaged(&signal, sizeof(int));
    checkCudaErrors(cudaGetLastError());

    *signal = 1;

    PersistentKernel<<<2, 1, 0, stream_background>>>(signal);

    int n = 100;
    for (int i = 0; i < n; i++) {
        OneTimeKernel<<<1, 1, 0, stream>>>(i);
    }

    cudaStreamSynchronize(stream);
    checkCudaErrors(cudaGetLastError());
    *signal = 0;
    cudaStreamSynchronize(stream_background);
    checkCudaErrors(cudaGetLastError());
}
