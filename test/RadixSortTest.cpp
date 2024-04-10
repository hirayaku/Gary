#include <vector>
#include <numeric>
#include <random>
#include "gtest/gtest.h"
#include "ParRadixSort.h"

template <typename T>
void checkSorted(const std::vector<T> &vec) {
  if (vec.size() == 0) return;

  size_t idx = 0;
  while (idx + 1 < vec.size()) {
    ASSERT_LE(vec[idx], vec[idx+1]);
    idx++;
  }
}

TEST(RadixSortTest, Handles32bit) {
  constexpr size_t size = 37;
  std::vector<uint32_t> unsorted(size);
  std::iota(unsorted.begin(), unsorted.end(), 0);

  const auto indices = utils::radixSortInplacePar(unsorted);
  checkSorted(unsorted);

  std::vector<uint32_t> copy(size);
  std::reverse_copy(unsorted.begin(), unsorted.end(), copy.begin());

  utils::radixSortInplacePar(copy);
  checkSorted(copy);
}

TEST(RadixSortTest, HandlesRandom32bit) {
  constexpr size_t size = 377;
  std::vector<uint32_t> unsorted(size);

  std::mt19937 gen{std::random_device()()};
  std::uniform_int_distribution<uint32_t> dist{0, 0xFFFFFFFF};
  std::generate(unsorted.begin(), unsorted.end(), [&]() { return dist(gen); });

  std::vector<uint32_t> copy(unsorted);
  const auto indices = utils::radixSortInplacePar(copy);
  checkSorted(copy);
}

TEST(RadixSortTest, Handles64bit) {
  constexpr size_t size = 37;
  std::vector<uint64_t> unsorted(size);
  std::iota(unsorted.begin(), unsorted.end(), 0);
  std::for_each(unsorted.begin(), unsorted.end(), [](auto &a) { a = size - a; });
  std::vector<uint64_t> copy(unsorted);
  const auto indices = utils::radixSortInplacePar(copy);
  checkSorted(copy);
}