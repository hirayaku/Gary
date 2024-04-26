#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include "Utils.h"
#include "HelperCUDA.h"

namespace utils {

void cudaNvtxStart(std::string msg) { nvtxRangePushA(msg.c_str()); }

void cudaNvtxStop() { nvtxRangePop(); }

void cudaHostSync() { checkCudaErrors(cudaDeviceSynchronize()); }

void cudaHostSync(cudaStream_t stream) {
  checkCudaErrors(cudaStreamSynchronize(stream));
}

}