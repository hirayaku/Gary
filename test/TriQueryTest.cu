#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "HelperCUDA.h"
#include "gtest/gtest.h"
#include "kernels/GraphQueryHom.cuh"
#include "Timer.h"

class CUDATest : public testing::Test {
protected:
  static void SetUpTestCase() {
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    gpuDeviceInit(0);
  }

  // static DeviceGraphCtx PrepareDataset(std::string )
};

