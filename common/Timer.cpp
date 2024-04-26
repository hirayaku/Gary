#include <cuda_runtime.h>
#include "Timer.h"
#include "Utils.h"
#include "HelperCUDA.h"

namespace utils{

CUDATimer::CUDATimer(const std::string &name)
: Timer{name}, stream_{nullptr} {
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));
}

CUDATimer::CUDATimer(const std::string &name, cudaStream_t stream)
: Timer{name}, stream_{stream} {
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));
}

void CUDATimer::start() {
  utils::cudaNvtxStart(name_);
  utils::cudaHostSync(); // make sure no pending kernels are running
  cudaEventRecord(start_event, stream_);
}

void CUDATimer::stop() {
  cudaEventRecord(stop_event, stream_);
  checkCudaErrors(cudaEventSynchronize(stop_event));
  utils::cudaNvtxStop();
  cudaEventElapsedTime(&millisecs_, start_event, stop_event);
}

CUDATimer::~CUDATimer() {
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
}

}
