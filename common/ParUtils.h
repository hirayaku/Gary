#include <numeric>
#include "omp.h"

namespace utils {
template <typename T>
inline void atomicAdd(T &obj, const T &val) {
    #pragma omp atomic update
    obj += val;
}

template <typename T>
inline void atomicAdd(T &obj, const T &val, T &out) {
    #pragma omp atomic capture
    {
        out = obj;
        obj += val;
    }
}

template <typename InTy = unsigned, typename OutTy = unsigned>
OutTy prefixSumPar(size_t length, const InTy *in, OutTy *out) {
    const size_t block_size = 1 << 16;
    const size_t num_blocks = (length + block_size - 1) / block_size;
    std::vector<OutTy> local_sums(num_blocks);
    // count how many bits are set on each thread
    #pragma omp parallel for
    for (size_t block = 0; block < num_blocks; block ++) {
        OutTy lsum = 0;
        size_t block_end = std::min((block + 1) * block_size, length);
        for (size_t i = block * block_size; i < block_end; i++)
            lsum += in[i];
        local_sums[block] = lsum;
    }
    std::vector<OutTy> bulk_prefix(num_blocks + 1);
    OutTy total = 0;
    for (size_t block = 0; block < num_blocks; block++) {
        bulk_prefix[block] = total;
        total += local_sums[block];
    }
    bulk_prefix[num_blocks] = total;
    #pragma omp parallel for
    for (size_t block = 0; block < num_blocks; block ++) {
        OutTy local_total = bulk_prefix[block];
        size_t block_end = std::min((block + 1) * block_size, length);
        for (size_t i = block * block_size; i < block_end; i++) {
            out[i] = local_total;
            local_total += in[i];
        }
    }
    out[length] = bulk_prefix[num_blocks];
    return out[length];
}
}