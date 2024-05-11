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
    for (auto u : vAdj) {
      const auto uAdj = csr.N(u);
      auto outEnd = std::set_intersection(uAdj.begin(), uAdj.end(),
        vAdj.begin(), vAdj.end(), spad.begin());
      utils::atomicAdd(count, std::distance(spad.begin(), outEnd));
    }
  }
  return count;
}

TEST(TriQueryTest, TriangleCountingSmall) {
  utils::Timer timer("CPU counting");
  auto csr = loadFromBinary<GraphCSR>("/mnt/md0/datasets/com-orkut/csr_dag.bin");
  GTEST_LOG_(INFO) << csr.numV() << " " << csr.numE();

  timer.start();
  unsigned countRef = cpuTriCount(csr);
  timer.stop();

  GTEST_LOG_(INFO) << "ref count: " << countRef;
  GTEST_LOG_(INFO) << "CPU takes " << timer.millisecs() << "ms";

  DeviceGraphCtx ctx(std::make_shared<GraphCSR>(std::move(csr)));

  thrust::device_vector<unsigned> counts(1, 0);
  dim3 gridDim(256);
  dim3 blockDim(256);
  size_t smem = graph_query::getIndexCacheSMem(blockDim, 32);
  utils::CUDATimer timerG("Triangle Counting");
  timerG.start();
  graph_query::triangleCountHom<false, 32> <<<gridDim, blockDim, smem, timerG.stream()>>>(
    { ctx.getDeviceCSR(), ctx.getDeviceCOO() },
    thrust::raw_pointer_cast(counts.data())
  );
  timerG.stop();

  GTEST_LOG_(INFO) << "count: " << thrust::host_vector<unsigned>(counts)[0];
  GTEST_LOG_(INFO) << "kernel takes " << timerG.millisecs() << "ms";
}
