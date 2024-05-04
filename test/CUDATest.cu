#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "HelperCUDA.h"
#include "gtest/gtest.h"
#include "kernels/Empty.cuh"
#include "kernels/SegmentArray.cuh"
#include "Timer.h"

class CUDATest : public testing::Test {
protected:
  static void SetUpTestCase() {
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    gpuDeviceInit(0);
  }
  // static void TearDownTestCase();

  template <typename SrcT, typename DstT>
  static std::vector<thrust::host_vector<DstT>>
  RunSegmentArray(const thrust::host_vector<SrcT> &h_in,
    dim3 gridDim, dim3 blockDim, utils::CUDATimer &timer) {

    constexpr int SPLITS = sizeof(SrcT) / sizeof(DstT);
    thrust::device_vector<SrcT> d_in = h_in;
    std::vector<thrust::device_vector<DstT>> segments(SPLITS);
    thrust::host_vector<DstT *> segments_hptr(SPLITS);
    for (int i = 0; i < SPLITS; ++i) {
      segments[i] = thrust::device_vector<DstT>(h_in.size());
      segments_hptr[i] = thrust::raw_pointer_cast(segments[i].data());
    }
    thrust::device_vector<DstT *> segments_dptr = segments_hptr;

    size_t sharedMem = segmentKernelSMem<SrcT, DstT>(blockDim);
    timer.start();
    segmentKernel<SrcT, DstT> <<<gridDim, blockDim, sharedMem, timer.stream()>>>(
      thrust::raw_pointer_cast(d_in.data()), d_in.size(),
      thrust::raw_pointer_cast(segments_dptr.data())
    );
    timer.stop();
    checkCudaErrors(cudaGetLastError());

    std::vector<thrust::host_vector<DstT>> h_out(SPLITS);
    for (int i = 0; i < SPLITS; ++i) {
      h_out[i] = segments[i];
    }

    return h_out;
  }
};

TEST_F(CUDATest, RunEmptyKernel) {
  dim3 gridDim(2);
  dim3 blockDim(32);
  emptyKernel<<<gridDim, blockDim>>>();
  checkCudaErrors(cudaGetLastError());
}

TEST_F(CUDATest, ReturnCorrectResults) {
  constexpr int SIZE = 128;
  dim3 gridDim(1);
  dim3 blockDim(1024);
  thrust::host_vector<uint32_t> h_vec(SIZE, 0xAABBCCDD);

  utils::CUDATimer timer("segmentArray small");
  auto h_out = CUDATest::RunSegmentArray<uint32_t, uint8_t>(h_vec, gridDim, blockDim, timer);
  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": kernel takes " << timer.microsecs() << "us" << std::endl;

  uint32_t expected = 0xAABBCCDD;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      auto value = h_out[i][j];
      ASSERT_EQ(value, (uint8_t) (expected & 0xFF)) << "Index at (" << i << ", " << j << ")\n";
    }
    expected >>= 8;
  }
}

TEST_F(CUDATest, ReturnCorrectResults2) {
  constexpr int SIZE = 64*4;
  dim3 gridDim(2);
  dim3 blockDim(32);
  thrust::host_vector<uint32_t> h_vec(SIZE, 0xAABBCCDD);

  utils::CUDATimer timer("segmentArray small");
  auto h_out = CUDATest::RunSegmentArray<uint32_t, uint8_t>(h_vec, gridDim, blockDim, timer);
  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": kernel takes " << timer.microsecs() << "us" << std::endl;

  uint32_t expected = 0xAABBCCDD;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      auto value = h_out[i][j];
      ASSERT_EQ(value, (uint8_t) (expected & 0xFF)) << "Index at (" << i << ", " << j << ")\n";
    }
    expected >>= 8;
  }
}

TEST_F(CUDATest, CanHandleIrregularSize) {
  constexpr int SIZE = 127;
  dim3 gridDim(2);
  dim3 blockDim(512);
  thrust::host_vector<uint32_t> h_vec(SIZE, 0xAABBCCDD);

  utils::CUDATimer timer("segmentArray small");
  auto h_out = CUDATest::RunSegmentArray<uint32_t, uint8_t>(h_vec, gridDim, blockDim, timer);
  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": kernel takes " << timer.microsecs() << "us" << std::endl;

  uint32_t expected = 0xAABBCCDD;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      auto value = h_out[i][j];
      ASSERT_EQ(value, (uint8_t) (expected & 0xFF)) << "Index at (" << i << ", " << j << ")\n";
    }
    expected >>= 8;
  }
}

TEST_F(CUDATest, BenchmarkLargeArray) {
  constexpr int SIZE = 128 * 256 * 32;
  dim3 blockDim(512);
  dim3 gridDim((SIZE - 1) / blockDim.x + 1);
  thrust::host_vector<uint32_t> h_vec(SIZE, 0xAABBCCDD);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("segmentArray Large", stream);
  auto h_out = CUDATest::RunSegmentArray<uint32_t, uint8_t>(h_vec, gridDim, blockDim, timer);
  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": kernel takes " << timer.microsecs() << "us" << std::endl;

  uint32_t expected = 0xAABBCCDD;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      auto value = h_out[i][j];
      ASSERT_EQ(value, (uint8_t) (expected & 0xFF)) << "Index at (" << i << ", " << j << ")\n";
    }
    expected >>= 8;
  }
}

TEST_F(CUDATest, BenchmarkLargeArray_u64_u8) {
  constexpr int SIZE = 128 * 256 * 32;
  dim3 blockDim(512);
  dim3 gridDim((SIZE - 1) / blockDim.x + 1);
  thrust::host_vector<uint64_t> h_vec(SIZE, 0xAABBCCDDEEFF0011);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  utils::CUDATimer timer("segmentArray Large", stream);
  auto h_out = CUDATest::RunSegmentArray<uint64_t, uint8_t>(h_vec, gridDim, blockDim, timer);
  std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
    << ": kernel takes " << timer.microsecs() << "us" << std::endl;

  uint64_t expected = 0xAABBCCDDEEFF0011;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      auto value = h_out[i][j];
      ASSERT_EQ(value, (uint8_t) (expected & 0xFF)) << "Index at (" << i << ", " << j << ")\n";
    }
    expected >>= 8;
  }
}
