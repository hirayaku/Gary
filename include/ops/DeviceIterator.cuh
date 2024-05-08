#pragma once

#include <type_traits>
#include <cooperative_groups.h>
#include "Defines.h"

namespace utils {
  using thread_warp = cooperative_groups::thread_block_tile<
    WARP_SIZE, cooperative_groups::thread_block>;

  // TODO: we'd better check if CudaGroup is an instance of thread_block_tile
  template <typename CudaGroup>
  inline constexpr bool is_cooperatative_group_v =
    std::is_base_of_v<cooperative_groups::thread_group, std::remove_cv_t<CudaGroup>> ||
    std::is_same_v<thread_warp, std::remove_cv_t<CudaGroup>>;
}

static inline HOST_DEVICE int warpsPerBlock(dim3 blockDim) {
  return blockDim.x * blockDim.y * blockDim.z / WARP_SIZE;
}

template <typename SubGroup, typename SuperGroup>
static inline DEVICE_ONLY size_t subGroupId(const SubGroup &sub, const SuperGroup &super) {
  return super.thread_rank() / sub.size();
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

// A broadcasting span which generates the same iterator for threads in the same subGroup
template <typename T, typename SubGroup, typename SuperGroup, typename IterType=T*,
          typename = std::enable_if_t<utils::is_cooperatative_group_v<SubGroup>>,
          typename = std::enable_if_t<utils::is_cooperatative_group_v<SuperGroup>>>
class BCastSpan {
public:
  using ValueType = T;
  using value_type = T;

private:
  IterType const ptr_;
  const size_t len_;
  // rank of the subGroup in the superGroup
  const size_t rank_;
  // number of subGroups in the superGroup
  const int numGroups_;

public:
  // possible combination: (block, grid), (warp, grid), (warp, block), (subwarp, ...)
  const SubGroup &subGroup;
  const SuperGroup &superGroup;

  class ParIter {
  private:
    IterType i_;
    const int stepSize;
  public:
    DEVICE_ONLY ParIter(IterType i_, const int stepSize)
    : i_(i_), stepSize(stepSize) {}

    DEVICE_ONLY ValueType operator*() const { return *i_; }

    DEVICE_ONLY bool operator==(const ParIter &other) const { return i_ == other.i_; }

    // NOTE: since we are iterating in **parallel**, we could no longer guarantee
    // each copy of the iterator will reach exactly the value of this->end().i_
    // thus we have to use '<' rather than '!='
    DEVICE_ONLY bool operator!=(const ParIter &other) const { return i_ < other.i_; }

    DEVICE_ONLY ParIter operator++(int) {
      ParIter old {*this};
      i_ += stepSize;
      return old;
    }

    DEVICE_ONLY ParIter &operator++() {
      i_ += stepSize;
      return *this;
    }
  };

  // returns the **total size** of the span, not the per-thread size
  DEVICE_ONLY std::size_t size() const noexcept {
    return len_;
  }

  DEVICE_ONLY ParIter begin() const noexcept {
    return ParIter(ptr_ + rank_, numGroups_);
  }

  DEVICE_ONLY ParIter end() const noexcept {
    return ParIter(ptr_ + len_, numGroups_);
  }

  // `ptr` and `len` should be consistent across the `superGroup`
  DEVICE_ONLY BCastSpan(IterType ptr, const std::size_t len, const SubGroup &subGroup,
                        const SuperGroup &superGroup)
  : ptr_(ptr), len_(len), rank_(subGroupId(subGroup, superGroup)),
    numGroups_(superGroup.size() / subGroup.size()),
    subGroup(subGroup), superGroup(superGroup)
  { }
};


template<typename T, typename CudaGroup, typename Enable=void>
class Span;

// A span supporting traversal in parallel within CUDA
// IterType should be random accessible. The traversal order is strided by default.
// When initializing the Span, ptr & len should be consistent across all threads in the group
// **cooperative span is not assignable.**
template<typename T, typename CudaGroup>
class Span<T, CudaGroup,
           typename std::enable_if_t<utils::is_cooperatative_group_v<CudaGroup>>> {
private:
  T * const ptr_;
  const std::size_t len_;
  // rank of the thread in the group
  const std::size_t rank_;
  const int numThreads_;

public:
  const CudaGroup &group;

  using IterType = T*;
  using ValueType = T;
  using value_type = T;

  class ParIter {
  private:
    IterType i_;
    const int groupSize;
  public:
    DEVICE_ONLY ParIter(IterType i_, const int groupSize): i_(i_), groupSize(groupSize) {}

    DEVICE_ONLY ValueType operator*() const { return *i_; }

    DEVICE_ONLY bool operator==(const ParIter &other) const { return i_ == other.i_; }

    // NOTE: since we are iterating in **parallel**, we could no longer guarantee
    // each copy of the iterator will reach exactly the value of this->end().i_
    // thus we have to use '<' rather than '!='
    DEVICE_ONLY bool operator!=(const ParIter &other) const { return i_ < other.i_; }

    DEVICE_ONLY ParIter operator++(int) {
      ParIter old {*this};
      i_ += groupSize;
      return old;
    }

    DEVICE_ONLY ParIter &operator++() {
      i_ += groupSize;
      return *this;
    }
  };

  DEVICE_ONLY Span(IterType ptr, std::size_t len, const CudaGroup &group) noexcept
  : ptr_{ptr}, len_{len}, rank_(group.thread_rank()), numThreads_(group.size()), group(group)
  {}

  // returns the **total size** of the span, not the per-thread size
  DEVICE_ONLY std::size_t size() const noexcept {
    return len_;
  }

  DEVICE_ONLY ParIter begin() const noexcept {
    return ParIter(ptr_ + rank_, numThreads_);
  }

  DEVICE_ONLY ParIter end() const noexcept {
    return ParIter(ptr_ + len_, numThreads_);
  }

  DEVICE_ONLY ValueType &operator[](std::size_t i) noexcept {
    return ptr_[rank_ + i];
  }
};

// sequential Span (per-thread, shared or on host)
template<typename T>
class Span<T, void> {
private:
  T * ptr_;
  std::size_t len_;

public:
  using IterType = T*;
  using ValueType = T;
  using value_type = T;

  HOST_DEVICE Span(IterType ptr, const std::size_t len) noexcept
  : ptr_{ptr}, len_{len}
  {}

  HOST_DEVICE std::size_t size() const noexcept {
    return len_;
  }

  HOST_DEVICE IterType begin() const noexcept {
    return ptr_;
  }

  HOST_DEVICE IterType end() const noexcept {
    return ptr_ + len_;
  }

  HOST_DEVICE ValueType &operator[](std::size_t i) noexcept {
    return ptr_[i];
  }

  // transform the sequential span into a cooperative span within the `group`
  template <typename CudaGroup>
  DEVICE_ONLY Span<T, CudaGroup> cooperate(const CudaGroup &group) {
    return Span<T, CudaGroup>(ptr_, len_, group);
  }

  // transform the sequential span into a cooperative span broadcasting within `subGroup`
  // across the `superGroup`
  template <typename SubGroup, typename SuperGroup>
  DEVICE_ONLY BCastSpan<T, SubGroup, SuperGroup, IterType>
  cooperate(const SubGroup &subGroup, const SuperGroup &superGroup) {
    return BCastSpan<T, SubGroup, SuperGroup, IterType>(ptr_, len_, subGroup, superGroup);
  }
};

// default template for IdRange has incomplete types (no definition)
// which will cause compilation errors
template<typename T, typename CudaGroup, typename Enable=void>
class IdRange;

// Parallel version of IdRange for CudaGroup.
// This is useful when we want to iterate over a range within a group cooperatively.
// The traversal order is strided by default.
template <typename IdType, typename CudaGroup>
class IdRange<IdType, CudaGroup,
              typename std::enable_if_t<utils::is_cooperatative_group_v<CudaGroup>>> {
public:

  class ParIter {
    private:
      IdType i_;
      // initialize groupSize to avoid recomputing it for each iteration
      const int groupSize;

    public:
      DEVICE_ONLY ParIter(IdType i_, const int groupSize): i_(i_), groupSize(groupSize) {}

      DEVICE_ONLY constexpr IdType operator*() const { return i_; }

      DEVICE_ONLY constexpr bool operator==(const ParIter &other) const { return i_ == other.i_; }

      // NOTE: since we are iterating in **parallel**, we could no longer guarantee
      // each copy of the iterator will reach exactly the value of this->end().i_
      // thus we have to use '<' rather than '!='
      DEVICE_ONLY constexpr bool operator!=(const ParIter &other) const { return i_ < other.i_; }

      DEVICE_ONLY ParIter operator+(int steps) const {
        return ParIter {i_ + steps, groupSize};
      }

      DEVICE_ONLY ParIter &operator+=(int steps) const {
        i_ += steps;
        return *this;
      }

      DEVICE_ONLY ParIter operator++(int) {
        ParIter old {*this};
        i_ += groupSize;
        return old;
      }

      DEVICE_ONLY ParIter &operator++() {
        i_ += groupSize;
        return *this;
      }
  };

  using IterType = ParIter;
  using ValueType = IdType;
  using value_type = IdType;

  DEVICE_ONLY IterType begin() const { return ParIter(min + rank_, numThreads_); }

  DEVICE_ONLY IterType end() const { return ParIter(max, numThreads_); }

  DEVICE_ONLY size_t size() const { return max - min; };

  // cooperative ID range of [min, max)
  DEVICE_ONLY IdRange(IdType min, IdType max, const CudaGroup &group)
  : min(min), max(max), rank_(group.thread_rank()), numThreads_(group.size()), group(group)
  {}

  const CudaGroup &group;

private:
  const IdType min, max;
  const IdType rank_;
  const int numThreads_;
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

      HOST_DEVICE bool operator<(const Iter &other) const { return i_ < other.i_; }

      HOST_DEVICE Iter operator+(int steps) const { return Iter(i_ + steps); }

      HOST_DEVICE Iter &operator+=(int steps) {
        i_ += steps;
        return *this;
      }

      HOST_DEVICE Iter operator++(int) const {
        Iter old {*this};
        i_ += 1;
        return old;
      }

      HOST_DEVICE Iter &operator++() {
        i_ += 1;
        return *this;
      }
  };

  using IterType = Iter;
  using ValueType = IdType;
  using value_type = IdType;

  HOST_DEVICE IterType begin() const { return Iter(min); }

  HOST_DEVICE IterType end() const { return Iter(max); }

  DEVICE_ONLY size_t size() const { return max - min; };

  // vid ranged from [min, max)
  HOST_DEVICE IdRange(IdType min, IdType max): min(min), max(max) {}

  // transform the sequential IdRange into a cooperative IdRange within the `group`
  template <typename CudaGroup> DEVICE_ONLY IdRange<IdType, CudaGroup>
  cooperate(const CudaGroup &group) {
    return IdRange<IdType, CudaGroup>(min, max, group);
  }

  // transform the sequential IdRange into a cooperative IdRange broadcasting within `subGroup`
  // across the `superGroup`
  template <typename SubGroup, typename SuperGroup>
  DEVICE_ONLY BCastSpan<IdType, SubGroup, SuperGroup, IterType>
  cooperate(const SubGroup &subGroup, const SuperGroup &superGroup) {
    return BCastSpan<IdType, SubGroup, SuperGroup, IterType> {
      begin(), size(), subGroup, superGroup};
  }

private:
  const IdType min, max;
};
