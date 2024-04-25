#include <iostream>
#include <cassert>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <fmt/format.h>
#include "HelperString.h"
#include "HelperCUDA.h"
#include "Graph.h"
#include "ops/Search.cuh"

void cudaSetup(int deviceId, bool verbose) {
  // set up GPU device
  gpuDeviceInit(deviceId);
  if (verbose) {
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, deviceId));
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total # SM: %d\n", prop.multiProcessorCount);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max # warps per block: %d\n", (prop.maxThreadsPerBlock + prop.warpSize - 1) / prop.warpSize);
    printf("  Total amount of shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("  Total # registers per block: %d\n", prop.regsPerBlock);
    printf("  Total amount of constant memory: %lu bytes\n", prop.totalConstMem);
    printf("  Total global memory: %.1f GB\n", float(prop.totalGlobalMem)/float(1024*1024*1024));
    printf("  Memory Clock Rate: %.2f GHz\n", float(prop.memoryClockRate)/float(1024*1024));
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth: %.2f GB/s\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

int main(int argc, const char *argv[])
{
  int deviceCount = 0;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  assert(deviceCount > 0 && "No GPU device found!");
  int deviceId = getCmdLineArgumentInt(argc, argv, "device=");
  bool verbose = getCmdLineArgumentInt(argc, argv, "verbose=") != 0;
  cudaSetup(deviceId, verbose);

  return 0;
}
