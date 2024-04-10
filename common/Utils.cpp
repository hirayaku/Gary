#include "Utils.h"

namespace utils {

void cudaNvtxStart(std::string msg) { nvtxRangePushA(msg.c_str()); }

void cudaNvtxStop() { nvtxRangePop(); }

void cudaHostSync() { cudaDeviceSynchronize(); }
}