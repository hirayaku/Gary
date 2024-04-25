#pragma once
#include <cstdint>
#include <vector>

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