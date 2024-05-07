#pragma once
#include <cstdint>

typedef uint8_t byteT;
typedef uint8_t maskT;
typedef uint8_t labelT;
typedef labelT vlabelT;
typedef labelT elabelT;
typedef int32_t vidT;
typedef int64_t eidT;
typedef int32_t idxT;
typedef int32_t wghT;
typedef uint64_t embIdxT;

constexpr int WARP_SIZE = 32;
constexpr int FULL_MASK = 0xFFFFFFFF;

#ifndef HOST_DEVICE
#define HOST_DEVICE __host__ __device__
#endif

#ifndef HOST_ONLY
#define HOST_ONLY __host__
#endif

#ifndef DEVICE_ONLY
#define DEVICE_ONLY __device__
#endif
