#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/std/atomic>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "HelperCUDA.h"
#include "gtest/gtest.h"
#include "Timer.h"
#include "ops/DeviceIterator.cuh"
#include "ops/Intersection.cuh"

template <typename T>
__global__ void intersectKernel(T *sA, size_t sizeA, T *sB, size_t sizeB, T *sC, size_t *sizeC) {
  auto counter = cuda::std::atomic_ref<size_t>(*sizeC);
  Span<T, void> a {sA, sizeA}, b {sB, sizeB};
  size_t threadCount = intersectBinarySearch<T, false>(a, b, sC);
  counter.fetch_add(threadCount, cuda::std::memory_order_relaxed);
}

template <typename T, size_t N>
__global__ void intersectWithCacheKernel(T *sA, size_t sizeA, T *sB, size_t sizeB, T *sC, size_t *sizeC) {
  extern __shared__ T smem[];
  auto counter = cuda::std::atomic_ref<size_t>(*sizeC);

  const auto thisBlock = cg::this_thread_block();
  const auto thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto warpIdBlock = thisWarp.meta_group_rank();

  Span<T, void> a {sA, sizeA}, b {sB, sizeB};
  Array<T, N> cache {smem + warpIdBlock * N};
  size_t threadCount = intersectBinarySearch<T, N, false>(a, b, cache, sC);
  counter.fetch_add(threadCount, cuda::std::memory_order_relaxed);
}

template <typename T>
__global__ void intersectOGKernel(T *sA, size_t sizeA, T *sB, size_t sizeB, T *sC, size_t *sizeC) {
  extern __shared__ T smem[];
  // *sizeC = intersectBinarySearchOG(sA, sizeA, sB, sizeB, sC);
  *sizeC = intersect_bs_cache(sA, (T)sizeA, sB, (T)sizeB, sC, smem);
}

class SetOpTest : public testing::Test {
protected:
  static std::mt19937 gen;
  static thrust::host_vector<int> testSearch;
  static thrust::host_vector<int> testKeys;
  static constexpr int cacheSize = 32; 

  static void SetUpTestCase() {
    std::random_device rd;
    int seed = rd();
    gen = std::mt19937(seed);
    GTEST_LOG_(INFO) << "Using seed=" << seed;

    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    gpuDeviceInit(0);

    // generate common test data
    constexpr int low = 1024 * 1024 * 1024 / 16, high = 1024 * 1024 * 1024;
    const int keyLow = low, keyHigh = high;
    constexpr int numItems = 1024 * 1024 * 4;
    constexpr int numKeys = 1024 * 32;
    testSearch = GenerateRandomIntegers(numItems, low, high);
    testKeys = GenerateRandomIntegers(numKeys, keyLow, keyHigh);
  }

  // generate sorted random vectors
  template <typename T> static thrust::host_vector<T>
  GenerateRandomIntegers(size_t len, T min, T max) {
    std::uniform_int_distribution<T> dis(min, max);
    thrust::host_vector<T> result(len);
    for (size_t i = 0; i < len; ++i)
      result[i] = dis(gen);
    std::sort(result.begin(), result.end());
    return result;
  }
};

std::mt19937 SetOpTest::gen;
thrust::host_vector<int> SetOpTest::testSearch {};
thrust::host_vector<int> SetOpTest::testKeys {};


TEST_F(SetOpTest, SimpleIntersection) {
  size_t sizeA = testSearch.size(), sizeB = testKeys.size();
  std::vector<int> outRef(std::min(testSearch.size(), testKeys.size()));

  utils::Timer cpuTimer("intersection");
  cpuTimer.start();
  auto outEnd = std::set_intersection(testSearch.begin(), testSearch.end(),
    testKeys.begin(), testKeys.end(), outRef.begin());
  size_t sizeRef = std::distance(outRef.begin(), outEnd);
  cpuTimer.stop();
  GTEST_LOG_(INFO) << "CPU takes " << cpuTimer.microsecs() << "us";

  thrust::device_vector<int> devSetA = testSearch, devSetB = testKeys;
  thrust::device_vector<int> devSetC(std::min(sizeA, sizeB), 0);
  thrust::device_vector<size_t> sizeC(1);
  constexpr int cacheSize = 32;
  dim3 gridDim(1);
  dim3 blockDim(32);

  utils::CUDATimer timer("intersection");
  int smem = warpsPerBlock(blockDim) * cacheSize * sizeof(int);
  timer.start();
  intersectKernel<int> <<<gridDim, blockDim, 0, timer.stream()>>>(
    thrust::raw_pointer_cast(devSetA.data()), sizeA,
    thrust::raw_pointer_cast(devSetB.data()), sizeB,
    thrust::raw_pointer_cast(devSetC.data()), thrust::raw_pointer_cast(sizeC.data())
  );
  // intersectWithCacheKernel<int, cacheSize> <<<gridDim, blockDim, smem, timer.stream()>>>(
  //   thrust::raw_pointer_cast(devSetA.data()), sizeA,
  //   thrust::raw_pointer_cast(devSetB.data()), sizeB,
  //   thrust::raw_pointer_cast(devSetC.data()), thrust::raw_pointer_cast(sizeC.data())
  // );
  timer.stop();

  GTEST_LOG_(INFO) << "kernel takes " << timer.microsecs() << "us";

  // verify the corretness of results
  ASSERT_EQ(sizeC[0], sizeRef) << "incorrect sizes";
  thrust::host_vector<int> out = devSetC;
  std::sort(out.begin(), out.begin() + sizeRef);
  for (size_t i = 0; i < sizeC[0]; ++i) {
    ASSERT_EQ(out[i], outRef[i]) << "incorrect item at index " << i;
  }

  timer.start();
  smem = 32 * sizeof(int) * warpsPerBlock(blockDim);
  intersectOGKernel<int> <<<gridDim, blockDim, smem, timer.stream()>>>(
    thrust::raw_pointer_cast(devSetA.data()), sizeA,
    thrust::raw_pointer_cast(devSetB.data()), sizeB,
    thrust::raw_pointer_cast(devSetC.data()), thrust::raw_pointer_cast(sizeC.data())
  );
  timer.stop();
  GTEST_LOG_(INFO) << "OG kernel takes " << timer.microsecs() << "us";

  // verify the corretness of results
  ASSERT_EQ(sizeC[0], sizeRef) << "incorrect sizes";
  thrust::host_vector<int> out2 = devSetC;
  // std::sort(out2.begin(), out2.begin() + sizeRef);
  for (size_t i = 0; i < sizeC[0]; ++i) {
    ASSERT_EQ(out2[i], outRef[i]) << "incorrect item at index " << i;
  }
}
