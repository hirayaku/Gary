/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE_radix_sort file in the LICENSE directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <utility>
#include <stdexcept>

#if !defined(_OPENMP)

namespace utils {

bool inline is_radix_sort_available() {
  return false;
}

template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(K* inp_key_buf,
                                      V* inp_value_buf,
                                      K* tmp_key_buf,
                                      V* tmp_value_buf,
                                      int64_t elements_count,
                                      int64_t max_value) {
  throw std::runtime_error("radix_sort_parallel not compiled: OpenMP not available")
}

}  // namespace utils

#else

#include <optional>
#include <algorithm>
#include <numeric>
#include <limits>
#include <omp.h>

namespace utils {

namespace {

// Leading zeros counter from LLVM
template <typename T, std::size_t SizeOfT> struct LeadingZerosCounter {
  static unsigned count(T Val) {
    if (!Val)
      return std::numeric_limits<T>::digits;
 
    // Bisection method.
    unsigned ZeroBits = 0;
    for (T Shift = std::numeric_limits<T>::digits >> 1; Shift; Shift >>= 1) {
      T Tmp = Val >> Shift;
      if (Tmp)
        Val = Tmp;
      else
        ZeroBits |= Shift;
    }
    return ZeroBits;
  }
};
 
#if defined(__GNUC__) || defined(_MSC_VER)
template <typename T> struct LeadingZerosCounter<T, 4> {
  static unsigned count(T Val) {
    if (Val == 0)
      return 32;
 
#if __has_builtin(__builtin_clz) || defined(__GNUC__)
    return __builtin_clz(Val);
#elif defined(_MSC_VER)
    unsigned long Index;
    _BitScanReverse(&Index, Val);
    return Index ^ 31;
#endif
  }
};
 
#if !defined(_MSC_VER) || defined(_M_X64)
template <typename T> struct LeadingZerosCounter<T, 8> {
  static unsigned count(T Val) {
    if (Val == 0)
      return 64;
 
#if __has_builtin(__builtin_clzll) || defined(__GNUC__)
    return __builtin_clzll(Val);
#elif defined(_MSC_VER)
    unsigned long Index;
    _BitScanReverse64(&Index, Val);
    return Index ^ 63;
#endif
  }
};
#endif
#endif

template <typename T> [[nodiscard]] int countLeadingZeros(T Val) {
  static_assert(std::is_unsigned_v<T>,
                "Only unsigned integral types are allowed.");
  return LeadingZerosCounter<T, sizeof(T)>::count(Val);
}


// Copied from fbgemm implementation available here:
// https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/src/cpu_utils.cpp
//
// `radix_sort_parallel` is only available when pyg-lib is compiled with
// OpenMP, since the algorithm requires sync between omp threads, which can not
// be perfectly mapped to `at::parallel_for` at the current stage.

// histogram size per thread
constexpr int RDX_HIST_SIZE = 256;

template <typename K, typename V>
void radix_sort_kernel(K* input_keys,
                       V* input_values,
                       K* output_keys,
                       V* output_values,
                       int64_t elements_count,
                       int64_t* histogram,
                       int64_t* histogram_ps,
                       int pass) {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  int64_t elements_count_4 = elements_count / 4 * 4;

  int64_t* local_histogram = &histogram[RDX_HIST_SIZE * tid];
  int64_t* local_histogram_ps = &histogram_ps[RDX_HIST_SIZE * tid];

  // Step 1: compute histogram
  for (int i = 0; i < RDX_HIST_SIZE; i++) {
    local_histogram[i] = 0;
  }

#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    K key_1 = input_keys[i];
    K key_2 = input_keys[i + 1];
    K key_3 = input_keys[i + 2];
    K key_4 = input_keys[i + 3];

    local_histogram[(key_1 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_2 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_3 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_4 >> (pass * 8)) & 0xFF]++;
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      K key = input_keys[i];
      local_histogram[(key >> (pass * 8)) & 0xFF]++;
    }
  }
#pragma omp barrier
  // Step 2: prefix sum
  if (tid == 0) {
    int64_t sum = 0, prev_sum = 0;
    for (int bins = 0; bins < RDX_HIST_SIZE; bins++) {
      for (int t = 0; t < nthreads; t++) {
        sum += histogram[t * RDX_HIST_SIZE + bins];
        histogram_ps[t * RDX_HIST_SIZE + bins] = prev_sum;
        prev_sum = sum;
      }
    }
    histogram_ps[RDX_HIST_SIZE * nthreads] = prev_sum;
    assert(prev_sum == elements_count);
  }
#pragma omp barrier

  // Step 3: scatter
#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    K key_1 = input_keys[i];
    K key_2 = input_keys[i + 1];
    K key_3 = input_keys[i + 2];
    K key_4 = input_keys[i + 3];

    int bin_1 = (key_1 >> (pass * 8)) & 0xFF;
    int bin_2 = (key_2 >> (pass * 8)) & 0xFF;
    int bin_3 = (key_3 >> (pass * 8)) & 0xFF;
    int bin_4 = (key_4 >> (pass * 8)) & 0xFF;

    int64_t pos;
    pos = local_histogram_ps[bin_1]++;
    output_keys[pos] = key_1;
    output_values[pos] = input_values[i];
    pos = local_histogram_ps[bin_2]++;
    output_keys[pos] = key_2;
    output_values[pos] = input_values[i + 1];
    pos = local_histogram_ps[bin_3]++;
    output_keys[pos] = key_3;
    output_values[pos] = input_values[i + 2];
    pos = local_histogram_ps[bin_4]++;
    output_keys[pos] = key_4;
    output_values[pos] = input_values[i + 3];
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      K key = input_keys[i];
      int64_t pos = local_histogram_ps[(key >> (pass * 8)) & 0xFF]++;
      output_keys[pos] = key;
      output_values[pos] = input_values[i];
    }
  }
}

}  // namespace

bool inline is_radix_sort_available() {
  return true;
}

// radix_sort only support unsigned intergral types.
// TODO: add support for general integral types
template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(K* inp_key_buf,
                                      V* inp_value_buf,
                                      K* tmp_key_buf,
                                      V* tmp_value_buf,
                                      int64_t elements_count,
                                      K max_value) {
  int maxthreads = omp_get_max_threads();
  std::unique_ptr<int64_t[]> histogram_tmp(
    new int64_t[RDX_HIST_SIZE * maxthreads]);
  std::unique_ptr<int64_t[]> histogram_ps_tmp(
    new int64_t[RDX_HIST_SIZE * maxthreads + 1]);
  int64_t* histogram = histogram_tmp.get();
  int64_t* histogram_ps = histogram_ps_tmp.get();
  if (max_value == 0) {
    return std::make_pair(inp_key_buf, inp_value_buf);
  }

  // __builtin_clz is not portable
  int num_bits =
    sizeof(K) * 8 - countLeadingZeros(static_cast<std::make_unsigned_t<K> >(max_value));
  unsigned int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    K* input_keys = inp_key_buf;
    V* input_values = inp_value_buf;
    K* output_keys = tmp_key_buf;
    V* output_values = tmp_value_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++) {
      radix_sort_kernel(input_keys, input_values, output_keys, output_values,
                        elements_count, histogram, histogram_ps, pass);

      std::swap(input_keys, output_keys);
      std::swap(input_values, output_values);
#pragma omp barrier
    }
  }
  return (num_passes % 2 == 0 ? std::make_pair(inp_key_buf, inp_value_buf)
                              : std::make_pair(tmp_key_buf, tmp_value_buf));
}

// parallel in-place radix sort with an auxiliary array
template <typename K, typename V>
std::tuple<std::vector<K>, std::vector<V>> radixSortInplacePar(
  std::vector<K> &input, std::vector<V> &aux,
  const std::optional<K> max = std::nullopt
) {
  static_assert(std::is_unsigned_v<K>, "radixSort supports only unsigned intergral types");
  const auto elements = input.size();
  if (elements != aux.size())
    throw std::invalid_argument("inputs to radix sort are of different sizes");
  if (elements == 0) return {input, aux};

  K *sorted_keys = nullptr;
  V *sorted_vals = nullptr;
  std::vector<K> tmp_keys(input.size());
  std::vector<V> tmp_vals(input.size());

  const auto maximum = max.value_or(*std::max_element(input.begin(), input.end()));
  std::tie(sorted_keys, sorted_vals) = radix_sort_parallel(
    input.data(), aux.data(),
    tmp_keys.data(), tmp_vals.data(),
    elements, maximum
  );
  const bool sorted_in_place = input.data() == sorted_keys;
  if (!sorted_in_place) {
    // std::copy(sorted_vals, sorted_vals + elements, input.begin());
    // std::copy(sorted_indices, sorted_indices + elements, indices.begin());
    std::swap(input, tmp_keys);
    std::swap(aux, tmp_vals);
  }
  return {input, aux};
}

// parallel in-place radix arg-sort
template <typename K, typename IdxT=int64_t>
std::tuple<std::vector<K>, std::vector<IdxT>> radixArgSortInplacePar(
  std::vector<K> &input,
  const std::optional<K> max = std::nullopt
) {
  static_assert(std::is_unsigned_v<K>, "radixSort supports only unsigned intergral types");
  const auto elements = input.size();
  std::vector<IdxT> indices(elements);
  std::iota(indices.begin(), indices.end(), 0);
  return radixSortInplacePar<K, IdxT>(input, indices, max);
}


}  // namespace utils

#endif