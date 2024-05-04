#pragma once

#include <cooperative_groups.h>
#include <cuda/std/array>
#include "Defines.h"

namespace cooperative_groups {
  using thread_warp = thread_block_tile<WARP_SIZE>;
}

static inline int warpsPerBlock(dim3 blockDim) {
  return blockDim.x * blockDim.y * blockDim.z / WARP_SIZE;
}

// array view with compile-time constant sizes
template <typename T, size_t _Size>
struct Array {
private:
  T *data;
public:
  using value_type = T;
  HOST_DEVICE Array(T *data) : data(data) {}
  HOST_DEVICE T &operator[](size_t i) {
    return data[i];
  }
  HOST_DEVICE const T &operator[](size_t i) const {
    return data[i];
  }
  HOST_DEVICE constexpr size_t size() const { return _Size; }
  HOST_DEVICE T *begin() const { return data; }
  HOST_DEVICE T *end() const { return data + size(); }
};

// template <typename T, size_t _Size>
// using Array = cuda::std::array<T, _Size>;

// A span supporting traversal in parallel within CUDA
// IterType should be random accessible. The traversal order is strided by default.
// When initializing the Span, ptr & len should be consistent across all threads in the group
template<typename IterType, typename CudaGroup>
class Span {
private:
  IterType ptr_;
  std::size_t len_;

public:
  const CudaGroup &group;

  using ValueType = decltype(*ptr_);
  using value_type = decltype(*ptr_);

  class ParIter {
  private:
    std::remove_cv_t<IterType> i_;
    const CudaGroup &group;
  public:
    DEVICE_ONLY ParIter(IterType i_, const CudaGroup &group): i_(i_), group(group) {}

    DEVICE_ONLY ValueType operator*() const { return *i_; }

    DEVICE_ONLY bool operator==(const ParIter &other) const { return i_ == other.i_; }

    // NOTE: since we are iterating in **parallel**, we could no longer guarantee
    // each copy of the iterator will reach exactly the value of this->end().i_
    // thus we have to use '<' rather than '!='
    DEVICE_ONLY bool operator!=(const ParIter &other) const { return i_ < other.i_; }

    DEVICE_ONLY ParIter operator++(int) {
      ParIter old {*this};
      i_ += group.size();
      return old;
    }

    DEVICE_ONLY ParIter &operator++() {
      i_ += group.size();
      return *this;
    }
  };

  DEVICE_ONLY Span(IterType ptr, std::size_t len, const CudaGroup &group) noexcept
  : ptr_{ptr}, len_{len}, group(group)
  {}

  DEVICE_ONLY std::size_t size() const noexcept {
    return len_;
  }

  DEVICE_ONLY ParIter begin() noexcept {
    return ParIter(ptr_ + group.thread_rank(), group);
  }

  DEVICE_ONLY ParIter end() noexcept {
    return ParIter(ptr_ + len_, group);
  }

  DEVICE_ONLY ValueType &operator[](std::size_t i) noexcept {
    return *(ptr_ + group.thread_rank() + i);
  }
};

// sequential Span (per-thread, shared or on host)
template<typename IterType>
class Span<IterType, void> {
private:
  IterType ptr_;
  std::size_t len_;

public:
  using ValueType = decltype(*ptr_);
  using value_type = decltype(*ptr_);

  HOST_DEVICE Span(IterType ptr, const std::size_t len) noexcept
  : ptr_{ptr}, len_{len}
  {}

  HOST_DEVICE std::size_t size() const noexcept {
    return len_;
  }

  HOST_DEVICE IterType begin() noexcept {
    return ptr_;
  }

  HOST_DEVICE IterType end() noexcept {
    return ptr_ + len_;
  }

  HOST_DEVICE ValueType &operator[](std::size_t i) noexcept {
    return *(ptr_+i);
  }

  // transform the sequential span into a cooperative span within the `group`
  template <typename CudaGroup>
  Span<IterType, CudaGroup> cooperate(const CudaGroup &group) {
    return Span<IterType, CudaGroup>(ptr_, len_, group);
  }
};

// Parallel version of IdRange for CudaGroup.
// This is useful when we want to iterate over a range within a group cooperatively.
// The traversal order is strided by default.
template <typename IdType, typename CudaGroup>
class IdRange {
public:
  class ParIter {
    private:
      IdType i_;
      const CudaGroup &group;

    public:
      DEVICE_ONLY ParIter(IdType i_, const CudaGroup &group): i_(i_), group(group) {}

      DEVICE_ONLY IdType operator*() const { return i_; }

      DEVICE_ONLY bool operator==(const ParIter &other) const { return i_ == other.i_; }

      // NOTE: since we are iterating in **parallel**, we could no longer guarantee
      // each copy of the iterator will reach exactly the value of this->end().i_
      // thus we have to use '<' rather than '!='
      DEVICE_ONLY bool operator!=(const ParIter &other) const { return i_ < other.i_; }

      DEVICE_ONLY ParIter operator++(int) {
        ParIter old {*this};
        i_ += group.size();
        return old;
      }

      DEVICE_ONLY ParIter &operator++() {
        i_ += group.size();
        return *this;
      }
  };

  DEVICE_ONLY ParIter begin() { return ParIter(min + (IdType)group.thread_rank(), group); }

  DEVICE_ONLY ParIter end() { return ParIter(max, group); }

  // vid ranged from [min, max)
  DEVICE_ONLY IdRange(IdType min, IdType max, const CudaGroup &group)
  : min(min), max(max), group(group) {}

  const CudaGroup &group;

private:
  const IdType min, max;
};

// sequential IdRange (per thread or host)
template <typename IdType>
class IdRange<IdType, void> {
public:
  class Iter {
    private:
      IdType i_;

    public:
      HOST_DEVICE Iter(IdType i_): i_(i_) {}

      HOST_DEVICE IdType operator*() const { return i_; }

      HOST_DEVICE bool operator==(const Iter &other) const { return i_ == other.i_; }

      HOST_DEVICE bool operator!=(const Iter &other) const { return i_ < other.i_; }

      HOST_DEVICE Iter operator++(int) {
        Iter old {*this};
        i_ += 1;
        return old;
      }

      HOST_DEVICE Iter &operator++() {
        i_ += 1;
        return *this;
      }
  };

  HOST_DEVICE Iter begin() { return Iter(min); }

  HOST_DEVICE Iter end() { return Iter(max); }

  // vid ranged from [min, max)
  HOST_DEVICE IdRange(IdType min, IdType max): min(min), max(max) {}

private:
  const IdType min, max;
};
