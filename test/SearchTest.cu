#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "HelperCUDA.h"
#include "gtest/gtest.h"
#include "Timer.h"
#include "ops/DeviceIterator.cuh"
#include "ops/Search.cuh"

template <typename T>
__global__ void binarySearchKernel(T* data, size_t len, T* keys, size_t numKeys,
  bool* results) {

  const auto thisGrid = cg::this_grid();
  const auto thisBlock = cg::this_thread_block();
  const auto threadId = thisGrid.thread_rank();

  Span<T*, void> list {data, len};
  for (auto idx : IdRange<size_t, decltype(thisGrid)> {0, numKeys, thisGrid}) {
    results[idx] = binary_search(list, keys[idx]);
  }
}

template <typename T, size_t N=32>
__global__ void binarySearch2PhaseKernel(T* data, size_t len, T* keys, size_t numKeys,
  bool* results) {

  extern __shared__ T smem[];
  const auto thisGrid = cg::this_grid();
  const auto thisBlock = cg::this_thread_block();
  const auto thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto warpIdBlock = thisWarp.meta_group_rank();

  Span<T*, void> list {data, len};
  Array<T, N> warpCache {smem + warpIdBlock * N};
  build_cache(list, warpCache, thisWarp);
  thisWarp.sync();

  for (auto idx : IdRange<size_t, decltype(thisGrid)> {0, numKeys, thisGrid}) {
    results[idx] = binary_search_2phase(list, warpCache, keys[idx]);
  }
}

template <typename T, typename CudaGroup>
__global__ void linearSearchKernel(T* data, size_t len, T* keys, size_t numKeys,
  bool* results) {
  // TODO
}

class SearchTest : public testing::Test {
protected:
  static int seed;

  static void SetUpTestCase() {
    seed = 0;
    // std::random_device rd;
    // seed = rd();
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    gpuDeviceInit(0);
  }

  // static void TearDownTestCase();

  template <typename T> static thrust::host_vector<T>
  GenerateRandomIntegers(size_t len, T min, T max) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<T> dis(min, max);
    thrust::host_vector<T> result(len);
    for (size_t i = 0; i < len; ++i)
      result[i] = dis(gen);
    return result;
  }

  template <typename T> static thrust::host_vector<bool>
  RunStdBinarySearch(
    const thrust::host_vector<T> &vec, const thrust::host_vector<T> &search
  ) {
    thrust::host_vector<bool> results(search.size());
    for (size_t i = 0; i < search.size(); ++i) {
      results[i] = std::binary_search(vec.begin(), vec.end(), search[i]);
    }
    return results;
  }

  template <typename T> static thrust::host_vector<bool>
  RunCPUBinarySearch(
    const thrust::host_vector<T> &vec, const thrust::host_vector<T> &search
  ) {
    thrust::host_vector<bool> results(search.size());
    Span<const T*, void> list {thrust::raw_pointer_cast(&vec[0]), vec.size()};
    for (size_t i = 0; i < search.size(); ++i)
      results[i] = binary_search(list, search[i]);
    return results;
  }

  template <typename T, size_t N> static thrust::host_vector<bool>
  RunCPUBinarySearch2Phase(
    const thrust::host_vector<T> &vec, const thrust::host_vector<T> &search,
    const Array<T, N> &cache
  ) {
    thrust::host_vector<bool> results(search.size());
    Span<const T*, void> list {thrust::raw_pointer_cast(vec.data()), vec.size()};
    for (size_t i = 0; i < search.size(); ++i)
      results[i] = binary_search_2phase<const T>(list, cache, search[i]);
    return results;
  }

  template <typename T> static thrust::host_vector<bool>
  RunCUDABinarySearch(
    const thrust::host_vector<T> &h_in, const thrust::host_vector<T> &h_search,
    dim3 gridDim, dim3 blockDim, utils::CUDATimer &timer) {

    thrust::device_vector<T> d_in = h_in;
    thrust::device_vector<T> d_search = h_search;
    thrust::device_vector<bool> d_results(h_search.size());

    timer.start();
    binarySearchKernel<T> <<<gridDim, blockDim, 0, timer.stream()>>> (
      thrust::raw_pointer_cast(d_in.data()), d_in.size(),
      thrust::raw_pointer_cast(d_search.data()), d_search.size(),
      thrust::raw_pointer_cast(d_results.data()));
    timer.stop();
    checkCudaErrors(cudaGetLastError());

    thrust::host_vector<bool> h_results = d_results;
    return h_results;
  }

  template <typename T, size_t N> static thrust::host_vector<bool>
  RunCUDABinarySearch2Phase(
    const thrust::host_vector<T> &h_in, const thrust::host_vector<T> &h_search,
    dim3 gridDim, dim3 blockDim, utils::CUDATimer &timer) {

    thrust::device_vector<T> d_in = h_in;
    thrust::device_vector<T> d_search = h_search;
    thrust::device_vector<bool> d_results(h_search.size());
    size_t smem = N * sizeof(T) * warpsPerBlock(blockDim);

    timer.start();
    binarySearch2PhaseKernel<T, N> <<<gridDim, blockDim, smem, timer.stream()>>> (
      thrust::raw_pointer_cast(&d_in[0]), d_in.size(),
      thrust::raw_pointer_cast(&d_search[0]), d_search.size(),
      thrust::raw_pointer_cast(&d_results[0]));
    timer.stop();
    checkCudaErrors(cudaGetLastError());

    thrust::host_vector<bool> h_results = d_results;
    return h_results;
  }
};

int SearchTest::seed;

TEST_F(SearchTest, BinarySearchCPU) {
  auto vec = GenerateRandomIntegers(1024*1024, 1024*1024*32/2, 1024*1024*32);
  std::sort(vec.begin(), vec.end());
  auto keys = GenerateRandomIntegers(1024*1024, 0, 1024*1024*32*2);
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  utils::Timer timer;
  timer.start();
  const auto results = RunCPUBinarySearch(vec, keys);
  timer.stop();
  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": target takes " << timer.microsecs() << "us" << std::endl;

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    EXPECT_TRUE(results[i] == resultsRef[i]) << " at index: " << i << "\n";
}

TEST_F(SearchTest, BinarySearchWithCacheCPU) {
  auto vec = GenerateRandomIntegers(1024*1024, 1024*1024*32/2, 1024*1024*32);
  std::sort(vec.begin(), vec.end());
  auto keys = GenerateRandomIntegers(1024*1024, 0, 1024*1024*32*2);
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  utils::Timer timer;
  timer.start();

  constexpr int cacheSize = 96;
  thrust::host_vector<int> cache(cacheSize);
  Span<const int*, void> list {thrust::raw_pointer_cast(vec.data()), vec.size()};
  Array<int, cacheSize> cacheView {cache.data()};
  build_cache(list, cacheView);

  // printf("list:  min: %d, max: %d\n", vec[0], vec[vec.size()-1]);
  // printf("cache: [");
  // for (int i = 0; i < cacheSize; ++i)
  //   printf("%d ", cache[i]);
  // printf("]\n");

  const auto results = RunCPUBinarySearch2Phase(vec, keys, cacheView);
  timer.stop();
  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": target takes " << timer.microsecs() << "us" << std::endl;

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}

TEST_F(SearchTest, BinarySearchCUDA) {
  constexpr int low = 1024 * 1024 * 1024 / 2, high = 1024 * 1024 * 1024;
  const int keyLow = 0, keyHigh = high;
  constexpr int numKeys = 1024 * 1024 * 4;
  auto vec = GenerateRandomIntegers(numKeys, low, high);
  auto keys = GenerateRandomIntegers(numKeys, keyLow, keyHigh);
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA", stream);
  dim3 gridDim(1);
  dim3 blockDim(256);
  const auto results = RunCUDABinarySearch<int>(vec, keys, gridDim, blockDim, timer);

  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": target takes " << timer.microsecs() << "us" << std::endl;

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}

TEST_F(SearchTest, BinarySearchWithCacheCUDA_SameRange) {
  constexpr int low = 1024 * 1024 * 1024 / 2, high = 1024 * 1024 * 1024;
  const int keyLow = 0, keyHigh = high;
  constexpr int numKeys = 1024 * 1024 * 4;
  auto vec = GenerateRandomIntegers(numKeys, low, high);
  auto keys = GenerateRandomIntegers(numKeys, keyLow, keyHigh);
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA (cache)", stream);
  dim3 gridDim(1);
  dim3 blockDim(256);
  constexpr int cacheSize = 32; 
  const auto results = RunCUDABinarySearch2Phase<int, cacheSize>(
    vec, keys, gridDim, blockDim, timer);

  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": target takes " << timer.microsecs() << "us" << std::endl;

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}
