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

  Span<T, void> list {data, len};
  for (auto idx : IdRange<size_t, decltype(thisGrid)> {0, numKeys, thisGrid}) {
    results[idx] = binary_search(list, keys[idx]);
  }
}

template <typename T>
__global__ void binarySearchCountKernel(T* data, size_t len, T* keys, size_t numKeys,
  int* count) {

  const auto thisGrid = cg::this_grid();
  const auto thisBlock = cg::this_thread_block();
  const auto threadId = thisGrid.thread_rank();
  // __shared__ int blockCount[1];

  Span<T, void> list {data, len};
  for (auto idx : IdRange<size_t, decltype(thisGrid)> {0, numKeys, thisGrid}) {
    if (binary_search(list, keys[idx]))
      atomicAdd(count, 1);
  }
  // // block-level atomic add
  // thisBlock.sync();
  // if (thisBlock.thread_rank() == 0) {
  //   atomicAdd(count, blockCount[0]);
  // }
}

template <typename T, size_t N=32>
__global__ void binarySearch2PhaseKernel(T* data, size_t len, T* keys, size_t numKeys,
  bool* results) {

  extern __shared__ T smem[];
  const auto thisGrid = cg::this_grid();
  const auto thisBlock = cg::this_thread_block();
  const auto thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto warpIdBlock = thisWarp.meta_group_rank();

  Span<T, void> list {data, len};
  Array<T, N> warpCache {smem + warpIdBlock * N};
  build_cache(list, warpCache, thisWarp);
  thisWarp.sync();

  for (auto idx : IdRange<size_t, decltype(thisGrid)> {0, numKeys, thisGrid}) {
    results[idx] = binary_search_2phase(list, warpCache, keys[idx]);
  }
}

template <typename T>
__global__ void batchedBinarySearchKernel(int batchSize, T **data, size_t *dataSizes,
  T** keys, size_t *keySizes, bool **results) {

  const auto thisGrid = cg::this_grid();
  const auto thisBlock = cg::this_thread_block();
  const auto blockId = thisGrid.block_rank();
  const auto thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto blockWarps = warpsPerBlock(blockDim);
  const auto gridWarps = thisGrid.num_blocks() * blockWarps;
  const auto warpId = blockId * blockWarps + thisWarp.meta_group_rank();

  for (int i = warpId; i < batchSize; i += gridWarps) {
    Span<T, void> dataSpan  {data[i], dataSizes[i]};
    Span<T, void> keySpan   {keys[i], keySizes[i]};
    Span<bool, void> resultSpan {results[i], keySizes[i]};
    for (int keyIdx : IdRange<size_t, utils::thread_warp> {0, keySizes[i], thisWarp} ) {
      resultSpan[keyIdx] = binary_search(dataSpan, keySpan[keyIdx]);
    }
  }
} 

template <typename T, size_t N>
__global__ void batchedBinarySearch2PhaseKernel(int batchSize, T **data, size_t *dataSizes,
  T** keys, size_t *keySizes, bool **results) {

  extern __shared__ T smem[];
  const auto thisGrid = cg::this_grid();
  const auto thisBlock = cg::this_thread_block();
  const auto blockId = thisGrid.block_rank();
  const auto thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto blockWarps = warpsPerBlock(blockDim);
  const auto gridWarps = thisGrid.num_blocks() * blockWarps;
  const auto warpId = blockId * blockWarps + thisWarp.meta_group_rank();
  const auto warpIdBlock = thisWarp.meta_group_rank();

  // int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  // int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA

  for (int i = warpId; i < batchSize; i += gridWarps) {
    Span<T, void> dataSpan  {data[i], dataSizes[i]};
    Span<T, void> keySpan   {keys[i], keySizes[i]};
    Span<bool, void> resultSpan {results[i], keySizes[i]};

    Array<T, N> warpCache {smem + warpIdBlock * N};
    build_cache(dataSpan, warpCache, thisWarp);
    thisWarp.sync(); 

    for (int keyIdx : IdRange<size_t, utils::thread_warp> {0, keySizes[i], thisWarp} ) {
      resultSpan[keyIdx] = binary_search_2phase(dataSpan, warpCache, keySpan[keyIdx]);
    }

    // Span<T, void> search  {data[i], dataSizes[i]};
    // Span<T, void> lookup  {keys[i], keySizes[i]};
    // Span<bool, void> resultSpan {results[i], keySizes[i]};
    // size_t search_size = dataSizes[i];
    // size_t lookup_size = keySizes[i];
    // smem[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
    // __syncwarp();

    // for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    //   T key = lookup[i]; // each thread picks a vertex as the key
    //   resultSpan[i]  = binary_search_2phase(search.begin(), smem, key, (int) search_size);
    // }
  }
} 

template <typename T, size_t N=32>
__global__ void binarySearch2PhaseCountKernel(T* data, size_t len, T* keys, size_t numKeys,
  int* count) {

  // __shared__ int blockCount[WARP_SIZE];
  extern __shared__ T smem[];
  const auto thisGrid = cg::this_grid();
  const auto thisBlock = cg::this_thread_block();
  const auto thisWarp = cg::tiled_partition<WARP_SIZE>(thisBlock);
  const auto warpIdBlock = thisWarp.meta_group_rank();
  const auto threadIdBlock = thisBlock.thread_rank();

  Span<T, void> list {data, len};
  Array<T, N> warpCache {smem + warpIdBlock * N};
  build_cache(list, warpCache, thisWarp);
  thisWarp.sync();

  int *blockCount = smem + thisWarp.meta_group_size() * N;

  for (auto idx : IdRange<size_t, decltype(thisGrid)> {0, numKeys, thisGrid}) {
    if (binary_search_2phase(list, warpCache, keys[idx])) {
      atomicAdd_block(count, 1);
    }
  }
  // thisBlock.sync();
  // if (threadIdBlock == 0)
  //   atomicAdd(count, *blockCount);
}

template <typename T> __global__ void
binarySearch2PhaseOG(T *search, size_t search_size, T *lookup, size_t lookup_size,
  bool* results) {

  //if (size_a == 0 || size_b == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  // __shared__ T cache[BLOCK_SIZE];
  extern __shared__ T cache[];
  cache[warp_lane * WARP_SIZE + thread_lane] = search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    T key = lookup[i]; // each thread picks a vertex as the key
    results[i]  = binary_search_2phase(search, cache, key, (int) search_size);
  }
}

class SearchTest : public testing::Test {
protected:
  static std::mt19937 gen;
  static thrust::host_vector<int> testSearch;
  static thrust::host_vector<int> testKeys;
  // TODO: there's some bug when sizes are above this value
  static constexpr int cacheSize = 32; 

  static void SetUpTestCase() {
    // std::random_device rd;
    // int seed = rd();
    int seed = 0;
    gen = std::mt19937(seed);
    GTEST_LOG_(INFO) << "Using seed=" << seed;

    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    gpuDeviceInit(0);

    // generate common test data
    constexpr int low = 1024 * 1024 * 1024 / 2, high = 1024 * 1024 * 1024;
    const int keyLow = low, keyHigh = high;
    constexpr int numItems = 1024 * 1024 * 4;
    constexpr int numKeys = 1024 * 32;
    testSearch = GenerateRandomIntegers(numItems, low, high);
    testKeys = GenerateRandomIntegers(numKeys, keyLow, keyHigh);
  }

  // static void TearDownTestCase();

  template <typename T> static thrust::host_vector<T>
  GenerateRandomIntegers(size_t len, T min, T max) {
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
    Span<const T, void> list {thrust::raw_pointer_cast(&vec[0]), vec.size()};
    for (size_t i = 0; i < search.size(); ++i)
      results[i] = binary_search(list, search[i]);
    return results;
  }

  template <typename T, size_t N> static thrust::host_vector<bool>
  RunCPUBinarySearch2Phase(
    const thrust::host_vector<T> &vec, const thrust::host_vector<T> &keys,
    const Array<T, N> &cache
  ) {
    thrust::host_vector<bool> results(keys.size());
    Span<const T, void> list {thrust::raw_pointer_cast(vec.data()), vec.size()};
    for (size_t i = 0; i < keys.size(); ++i) {
      results[i] = binary_search_2phase<const T>(list, cache, keys[i]);
    }
    return results;
  }

  template <typename T> static thrust::host_vector<bool>
  RunCUDABinarySearch(
    const thrust::host_vector<T> &h_in, const thrust::host_vector<T> &h_search,
    dim3 gridDim, dim3 blockDim, utils::CUDATimer &timer, bool count) {

    thrust::device_vector<T> d_in = h_in;
    thrust::device_vector<T> d_search = h_search;
    thrust::device_vector<bool> d_results(h_search.size());

    timer.start();
    if (count) {
      binarySearchCountKernel<T> <<<gridDim, blockDim, 0, timer.stream()>>> (
        thrust::raw_pointer_cast(d_in.data()), d_in.size(),
        thrust::raw_pointer_cast(d_search.data()), d_search.size(),
        (int *)thrust::raw_pointer_cast(d_results.data()));
    } else {
      binarySearchKernel<T> <<<gridDim, blockDim, 0, timer.stream()>>> (
        thrust::raw_pointer_cast(d_in.data()), d_in.size(),
        thrust::raw_pointer_cast(d_search.data()), d_search.size(),
        thrust::raw_pointer_cast(d_results.data()));
    }
    timer.stop();
    checkCudaErrors(cudaGetLastError());

    thrust::host_vector<bool> h_results = d_results;
    return h_results;
  }

  template <typename T, size_t N> static thrust::host_vector<bool>
  RunCUDABinarySearch2Phase(
    const thrust::host_vector<T> &h_in, const thrust::host_vector<T> &h_search,
    dim3 gridDim, dim3 blockDim, utils::CUDATimer &timer, bool count) {

    thrust::device_vector<T> d_in = h_in;
    thrust::device_vector<T> d_search = h_search;
    thrust::device_vector<bool> d_results(h_search.size(), 0);
    size_t smem = N * sizeof(T) * warpsPerBlock(blockDim);

    timer.start();
    if (count) {
      binarySearch2PhaseCountKernel<T, N> <<<gridDim, blockDim, smem+4, timer.stream()>>> (
        thrust::raw_pointer_cast(d_in.data()), d_in.size(),
        thrust::raw_pointer_cast(d_search.data()), d_search.size(),
        (int *)thrust::raw_pointer_cast(d_results.data()));
    } else {
      binarySearch2PhaseKernel<T, N> <<<gridDim, blockDim, smem, timer.stream()>>> (
        thrust::raw_pointer_cast(d_in.data()), d_in.size(),
        thrust::raw_pointer_cast(d_search.data()), d_search.size(),
        thrust::raw_pointer_cast(d_results.data()));
    }
    timer.stop();
    checkCudaErrors(cudaGetLastError());

    thrust::host_vector<bool> h_results = d_results;
    return h_results;
  }
};

std::mt19937 SearchTest::gen;
thrust::host_vector<int> SearchTest::testSearch {};
thrust::host_vector<int> SearchTest::testKeys {};

TEST_F(SearchTest, BinarySearchCPU) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  utils::Timer timer;
  timer.start();
  const auto results = RunCPUBinarySearch(vec, keys);
  timer.stop();

  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    EXPECT_TRUE(results[i] == resultsRef[i]) << " at index: " << i << "\n";
}

TEST_F(SearchTest, BinarySearchWithCacheCPU) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  utils::Timer timer;
  timer.start();

  thrust::host_vector<int> cache(cacheSize);
  Span<const int, void> list {thrust::raw_pointer_cast(vec.data()), vec.size()};
  Array<int, cacheSize> cacheView {cache.data()};
  build_cache(list, cacheView);

  // printf("list:  min: %d, max: %d\n", vec[0], vec[vec.size()-1]);
  // printf("keys:  min: %d, max: %d\n", keys[0], keys[keys.size()-1]);
  // printf("cache: [");
  // for (int i = 0; i < cacheSize; ++i)
  //   printf("%d ", cache[i]);
  // printf("]\n");

  const auto results = RunCPUBinarySearch2Phase(vec, keys, cacheView);
  timer.stop();
  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}

TEST_F(SearchTest, BinarySearchCUDA) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA", stream);
  dim3 gridDim(1);
  dim3 blockDim(32);
  const auto results = RunCUDABinarySearch<int>(vec, keys, gridDim, blockDim, timer, false);

  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}

// 2-phase binary search doesn't give speedups compared to the straightforward one,
// when the key range falls within the search range
TEST_F(SearchTest, BinarySearchWithCacheCUDA) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA (cache)", stream);
  dim3 gridDim(1);
  dim3 blockDim(32);
  const auto results = RunCUDABinarySearch2Phase<int, cacheSize>(
    vec, keys, gridDim, blockDim, timer, false);

  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}

TEST_F(SearchTest, BinarySearchOriginalCUDA) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  thrust::device_vector<int> d_in = vec;
  thrust::device_vector<int> d_keys = keys;
  thrust::device_vector<bool> d_results(keys.size(), 0);

  dim3 gridDim(1);
  dim3 blockDim(32);
  // original method use an index cache of WARP_SIZE
  size_t smem = WARP_SIZE * sizeof(int) * warpsPerBlock(blockDim);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA (cache)", stream);
  timer.start();
  binarySearch2PhaseOG<int> <<<gridDim, blockDim, smem, timer.stream()>>> (
    thrust::raw_pointer_cast(d_in.data()), d_in.size(),
    thrust::raw_pointer_cast(d_keys.data()), d_keys.size(),
    thrust::raw_pointer_cast(d_results.data()));
  timer.stop();
  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  thrust::host_vector<bool> results = d_results;
  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}

TEST_F(SearchTest, BinarySearchCountCUDA) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);
  const auto countRef = std::accumulate(resultsRef.begin(), resultsRef.end(), 0);
  GTEST_LOG_(INFO) << countRef << " items found";

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA", stream);
  dim3 gridDim(8);
  dim3 blockDim(512);
  const auto results = RunCUDABinarySearch<int>(vec, keys, gridDim, blockDim, timer, true);

  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  ASSERT_EQ(*(int *)results.data(), countRef);
}

// 2-phase binary search doesn't give speedups compared to the straightforward one,
// when the key range falls within the search range
TEST_F(SearchTest, BinarySearchWithCacheCountCUDA) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  std::sort(keys.begin(), keys.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);
  const auto countRef = std::accumulate(resultsRef.begin(), resultsRef.end(), 0);
  GTEST_LOG_(INFO) << countRef << " items found";

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA (cache)", stream);
  dim3 gridDim(8);
  dim3 blockDim(512);
  const auto results = RunCUDABinarySearch2Phase<int, cacheSize>(
    vec, keys, gridDim, blockDim, timer, true);

  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  ASSERT_EQ(*(int *)results.data(), countRef);
}

TEST_F(SearchTest, BinarySearchUnsortedCUDA) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA", stream);
  dim3 gridDim(8);
  dim3 blockDim(512);
  const auto results = RunCUDABinarySearch<int>(vec, keys, gridDim, blockDim, timer, false);

  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}

// 2-phase binary search doesn't give speedups compared to the straightforward one,
// when keys are unsorted
TEST_F(SearchTest, BinarySearchWithCacheUnsortedCUDA) {
  auto vec = testSearch;
  auto keys = testKeys;
  std::sort(vec.begin(), vec.end());
  const auto resultsRef = RunStdBinarySearch(vec, keys);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Binary search on CUDA (cache)", stream);
  dim3 gridDim(8);
  dim3 blockDim(512);
  const auto results = RunCUDABinarySearch2Phase<int, cacheSize>(
    vec, keys, gridDim, blockDim, timer, false);

  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  ASSERT_EQ(results.size(), resultsRef.size());
  for (size_t i = 0; i < results.size(); ++i)
    ASSERT_EQ(results[i], resultsRef[i]) << " at index: " << i
      << " item: " << keys[i] << "\n";
}

TEST_F(SearchTest, BatchedBinarySearchCUDA) {
  constexpr int batchSize = 256;
  constexpr int numKeys = 1024 * 64;
  constexpr int low = 1024 * 1024 * 1024 / 16, high = 1024 * 1024 * 1024;
  std::vector<thrust::host_vector<int>> searchVecs(batchSize);
  std::vector<thrust::host_vector<int>> keyVecs(batchSize);
  thrust::host_vector<size_t> hostSearchSizes(batchSize), hostKeySizes(batchSize);

  // generate test vectors
  std::uniform_int_distribution<size_t> dis(1024 * 1024 * 4, 1024 * 1024 * 8);
  for (int i = 0; i < batchSize; ++i) {
    size_t len = dis(gen);
    searchVecs[i] = GenerateRandomIntegers(len, low, high);
    keyVecs[i] = GenerateRandomIntegers(numKeys, low, high);
    hostSearchSizes[i] = len;
    hostKeySizes[i] = numKeys;
    if (i % 16 == 0) std::cout << '.' << std::flush;
  }
  std::cout << std::endl;

  #pragma omp parallel for
  for (int i = 0; i < batchSize; ++i) {
    std::sort(searchVecs[i].begin(), searchVecs[i].end());
    std::sort(searchVecs[i].begin(), searchVecs[i].end());
    if (i % 16 == 0) std::cout << '.' << std::flush;
  }
  std::cout << std::endl;

  // get reference results on CPU
  std::vector<thrust::host_vector<bool>> resultRefVecs(batchSize);
  utils::Timer cpuTimer("Parallel binary lookup");
  cpuTimer.start();
  #pragma omp parallel for
  for (int i = 0; i < batchSize; ++i) {
    resultRefVecs[i] = RunStdBinarySearch(searchVecs[i], keyVecs[i]);
    if (i % 16 == 0) std::cout << '.' << std::flush;
  }
  cpuTimer.stop();
  std::cout << std::endl;
  GTEST_LOG_(INFO) << "CPU takes " << cpuTimer.microsecs() << "us";

  // move host vectors to device
  std::vector<thrust::device_vector<int>> devSearchVecs(batchSize), devKeyVecs(batchSize);
  thrust::host_vector<int *> hostSearchPtrs(batchSize), hostKeyPtrs(batchSize);
  std::vector<thrust::device_vector<bool>> resultVecs(batchSize);
  thrust::host_vector<bool *> hostResultPtrs(batchSize);
  for (int i = 0; i < batchSize; ++i) {
    // search vectors
    devSearchVecs[i] = searchVecs[i];
    hostSearchPtrs[i] = thrust::raw_pointer_cast(devSearchVecs[i].data());
    // key vectors
    devKeyVecs[i] = keyVecs[i];
    hostKeyPtrs[i] = thrust::raw_pointer_cast(devKeyVecs[i].data());
    // result vectors
    resultVecs[i] = thrust::device_vector<bool>(numKeys);
    hostResultPtrs[i] = thrust::raw_pointer_cast(resultVecs[i].data());
  }

  thrust::device_vector<int *> devSearchPtrs = hostSearchPtrs, devKeyPtrs = hostKeyPtrs;
  thrust::device_vector<size_t> devSearchSizes = hostSearchSizes, devKeySizes = hostKeySizes;
  thrust::device_vector<bool *> devResultPtrs = hostResultPtrs;

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("Batched binary search on CUDA", stream);
  dim3 gridDim(batchSize);
  dim3 blockDim(32);

  timer.start();
  batchedBinarySearchKernel<int> <<<gridDim, blockDim, 0, stream>>> (
    batchSize,
    thrust::raw_pointer_cast(devSearchPtrs.data()), thrust::raw_pointer_cast(devSearchSizes.data()),
    thrust::raw_pointer_cast(devKeyPtrs.data()), thrust::raw_pointer_cast(devKeySizes.data()),
    thrust::raw_pointer_cast(devResultPtrs.data())
  );
  timer.stop();
  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";
  for (int i = 0; i < batchSize; ++i) {
    thrust::host_vector<bool> hostResultVec = resultVecs[i];
    for (size_t keyIdx = 0; keyIdx < numKeys; ++keyIdx) 
      ASSERT_EQ(hostResultVec[keyIdx], resultRefVecs[i][keyIdx]) << " at index: "
        << '[' << i << ',' << keyIdx << "], item=" << keyVecs[i][keyIdx] << "\n";
  }

  timer.start();
  size_t smem = cacheSize * sizeof(int) * warpsPerBlock(blockDim);
  batchedBinarySearch2PhaseKernel<int, cacheSize> <<<gridDim, blockDim, smem, stream>>> (
    batchSize,
    thrust::raw_pointer_cast(devSearchPtrs.data()), thrust::raw_pointer_cast(devSearchSizes.data()),
    thrust::raw_pointer_cast(devKeyPtrs.data()), thrust::raw_pointer_cast(devKeySizes.data()),
    thrust::raw_pointer_cast(devResultPtrs.data())
  );
  timer.stop();
  GTEST_LOG_(INFO) << "target takes " << timer.microsecs() << "us";

  for (int i = 0; i < batchSize; ++i) {
    thrust::host_vector<bool> hostResultVec = resultVecs[i];
    for (size_t keyIdx = 0; keyIdx < numKeys; ++keyIdx) 
      ASSERT_EQ(hostResultVec[keyIdx], resultRefVecs[i][keyIdx]) << " at index: "
        << '[' << i << ',' << keyIdx << "], item=" << keyVecs[i][keyIdx] << "\n";
  }
}
