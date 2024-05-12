#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include "HelperCUDA.h"
#include "ParUtils.h"
#include "gtest/gtest.h"
#include "Graph.h"
#include "DeviceGraph.cuh"
#include "kernels/GraphQueryHom.cuh"
#include "Timer.h"

unsigned cpuTriCount(const GraphCSR &csr) {
  size_t count = 0;
  const auto &deg = csr.getDegreeAll();
  vidT maxD = *std::max_element(deg.begin(), deg.end());
  std::vector<vidT> spad(maxD, 0);

  #pragma omp parallel for schedule(dynamic) firstprivate(spad)
  for (vidT v = 0; v < csr.numV(); ++v) {
    const auto vAdj = csr.N(v);
    const auto last = vAdj.size() - 1;
    unsigned localCount = 0;
    for (auto u : vAdj) {
      const auto uAdj = csr.N(u);
      auto outEnd = std::set_intersection(uAdj.begin(), uAdj.end(),
        vAdj.begin(), vAdj.end(), spad.begin());
      localCount += std::distance(spad.begin(), outEnd);
    }
    utils::atomicAdd(count, localCount);
  }
  return count;
}

class TriQueryTest : public ::testing::Test {
public:
  static DeviceGraphCtx ctx;
  static constexpr int cacheSize = 32;

  static void SetUpTestCase() {
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    gpuDeviceInit(0);

    auto csr = loadFromBinary<GraphCSR>("/mnt/md0/datasets/com-orkut/csr_dag.bin");
    GTEST_LOG_(INFO) << "m=" << csr.numV() << ", nnz=" << csr.numE();
    ctx.loadFromHost(std::make_shared<GraphCSR>(std::move(csr)));
    ctx.populateDevice();
  }

  static void TearDownTestCase() {
    ctx.clear();
  }
};

DeviceGraphCtx TriQueryTest::ctx;

TEST_F(TriQueryTest, RunCPU) {
  utils::Timer timer("CPU counting");
  timer.start();
  unsigned countRef = cpuTriCount(*ctx.getHostCSR());
  timer.stop();
  GTEST_LOG_(INFO) << "ref count: " << countRef;
  GTEST_LOG_(INFO) << "CPU takes " << timer.millisecs() << "ms";
}

TEST_F(TriQueryTest, RunGPULegacy) {
  thrust::device_vector<unsigned> counts(1, 0);
  dim3 gridDim(1024);
  dim3 blockDim(256);

  utils::CUDATimer timerG("Triangle Counting");
  timerG.start();
  graph_query::triangleCountLegacy<<<gridDim, blockDim, 0, timerG.stream()>>>(
    { ctx.getDeviceCSR(), ctx.getDeviceCOO() },
    thrust::raw_pointer_cast(counts.data())
  );
  timerG.stop();
  checkCudaErrors(cudaGetLastError());

  GTEST_LOG_(INFO) << "count: " << thrust::host_vector<unsigned>(counts)[0];
  GTEST_LOG_(INFO) << "legacy kernel takes " << timerG.millisecs() << "ms";
}

TEST_F(TriQueryTest, RunGPU) {
  thrust::device_vector<unsigned> counts(1, 0);
  dim3 gridDim(1024);
  dim3 blockDim(256);

  utils::CUDATimer timerG("Triangle Counting");
  timerG.start();
  graph_query::triangleCountHom<false> <<<gridDim, blockDim, 0, timerG.stream()>>>(
    { ctx.getDeviceCSR(), ctx.getDeviceCOO() },
    thrust::raw_pointer_cast(counts.data())
  );
  timerG.stop();
  checkCudaErrors(cudaGetLastError());
  GTEST_LOG_(INFO) << "count: " << thrust::host_vector<unsigned>(counts)[0];
  GTEST_LOG_(INFO) << "kernel (no index cache) takes " << timerG.millisecs() << "ms";
}

TEST_F(TriQueryTest, RunGPUWithCache) {
  thrust::device_vector<unsigned> counts(1, 0);
  dim3 gridDim(1024);
  dim3 blockDim(256);
  size_t smem = graph_query::getIndexCacheSMem(blockDim, cacheSize);

  utils::CUDATimer timerG("Triangle Counting");
  timerG.start();
  graph_query::triangleCountHom<true, cacheSize> <<<gridDim, blockDim, smem, timerG.stream()>>>(
    { ctx.getDeviceCSR(), ctx.getDeviceCOO() },
    thrust::raw_pointer_cast(counts.data())
  );
  timerG.stop();
  checkCudaErrors(cudaGetLastError());
  GTEST_LOG_(INFO) << "count: " << thrust::host_vector<unsigned>(counts)[0];
  GTEST_LOG_(INFO) << "kernel (with index cache) takes " << timerG.millisecs() << "ms";
}