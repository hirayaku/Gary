#include <random>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "gtest/gtest.h"
#include "HelperCUDA.h"
#include "Timer.h"
#include "Zipf.h"
#include "ops/DeviceIterator.cuh"
#include "ops/DeviceQueue.cuh"

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

// incrementing cycles counter until the signal is set
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

// take items from the input in a fixed order
__global__ void FixedSleepKernel(unsigned *durations, size_t size, long long *finishTime) {
  __shared__ volatile unsigned smem[128];
  auto spad = (unsigned *)smem;
  assert(size <= 128);

  const auto &thisBlock = cooperative_groups::this_thread_block();
  const auto &thisWarp = cooperative_groups::tiled_partition<WARP_SIZE>(thisBlock);
  const auto &lane = thisWarp.thread_rank();
  cooperative_groups::memcpy_async(thisBlock, spad, durations, size * sizeof(unsigned));
  cooperative_groups::wait(thisBlock);

  Span<unsigned, void> span {spad, size};
  const auto start = clock64();
  for (auto t : span.cooperate(thisWarp, thisBlock)) {
    __nanosleep(t * 1024); // us
  }
  thisWarp.sync();
  const auto end = clock64();
  if (lane == 0)
    finishTime[thisWarp.meta_group_rank()] = (end - start);
}

// take items from the input in a fair order
__global__ void FairSleepKernel(unsigned *durations, size_t size, long long *finishTime) {
  __shared__ volatile unsigned smem[128];
  auto spad = (unsigned *)smem;
  __shared__ volatile unsigned pos[2];
  assert(size <= 128);

  const auto &thisBlock = cooperative_groups::this_thread_block();
  const auto &thisWarp = cooperative_groups::tiled_partition<WARP_SIZE>(thisBlock);
  const auto &lane = thisWarp.thread_rank();
  CTAQueue<unsigned, 128, true> sharedQ {spad, (unsigned *)pos};
  sharedQ.fill(durations, size);

  unsigned t = 0;
  int loop = 1;
  const auto start = clock64();
  while (true) {
    if (lane == 0) loop = sharedQ.tryDeqX(t);
    // if sharedQ empty, exit
    if (!thisWarp.all(loop)) break;
    // sleep together
    t = thisWarp.shfl(t, 0);
    __nanosleep(t * 1024); // us
  }
  thisWarp.sync();
  const auto end = clock64();
  if (lane == 0)
    finishTime[thisWarp.meta_group_rank()] = (end - start);
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

  checkCudaErrors(cudaStreamSynchronize(stream));
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

template <typename T> static thrust::universal_vector<T>
GenerateRandomIntegers(size_t len, T min, T max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  utils::zipf_distribution<T> dis(max-min);
  thrust::universal_vector<T> result(len);
  for (size_t i = 0; i < len; ++i)
    result[i] = dis(gen) + min;
  return result;
}

TEST(CTACommTest, FairSleep) {
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
  float mHz = prop.clockRate / 1000;
  std::cout << mHz << " MHz\n";

  int numWarps = 8;
  size_t len = 64;
  auto durations = GenerateRandomIntegers(len, 10, 1024);
  std::cout << "Generated sleep cycles:\n" << fmt::format("{}", durations) << std::endl;
  std::vector<unsigned> accum(numWarps);
  for (size_t i = 0; i < len; ++i) {
    accum[ i % numWarps] += durations[i];
  }
  std::cout << "Per-warp sleep cycles:\n" << fmt::format("{}", accum) << std::endl;

  auto sleepTime = thrust::device_vector<long long>(numWarps);

  FixedSleepKernel<<<1, numWarps * WARP_SIZE>>>(
    (unsigned *)thrust::raw_pointer_cast(durations.data()), durations.size(),
    (long long *)thrust::raw_pointer_cast(sleepTime.data())
  );
  {
  thrust::host_vector<long long> cycles = sleepTime;
  for (int i = 0; i < numWarps; ++i) {
    std::cout << fmt::format("[{}]: {:.2f} us", i, (float)cycles[i] / mHz ) << std::endl;
  }
  long long sum = std::accumulate(sleepTime.begin(), sleepTime.end(), 0);
  long long max = *std::max_element(sleepTime.begin(), sleepTime.end());
  std::cout << fmt::format("Mean: {:.2f}, Max: {:.2f}", sum / numWarps / mHz, max / mHz) << std::endl;
  }

  FairSleepKernel<<<1, numWarps * WARP_SIZE>>>(
    (unsigned *)thrust::raw_pointer_cast(durations.data()), durations.size(),
    (long long *)thrust::raw_pointer_cast(sleepTime.data())
  );
  {
  thrust::host_vector<long long> cycles = sleepTime;
  for (int i = 0; i < numWarps; ++i) {
    std::cout << fmt::format("[{}]: {:.2f} us", i, (float)cycles[i] / mHz ) << std::endl;
  }
  long long sum = std::accumulate(sleepTime.begin(), sleepTime.end(), 0);
  long long max = *std::max_element(sleepTime.begin(), sleepTime.end());
  std::cout << fmt::format("Mean: {:.2f}, Max: {:.2f}", sum / numWarps / mHz, max / mHz) << std::endl;
  }
}
