#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <memory>
#include <cuda_runtime.h>

#include "Zipf.h"

namespace utils {

template<typename IterType>
class Span {
  IterType ptr_;
  std::size_t len_;

public:
  Span(IterType ptr, std::size_t len) noexcept
      : ptr_{ptr}, len_{len}
  {}

  std::size_t size() const noexcept {
      return len_;
  }

  IterType begin() noexcept {
      return ptr_;
  }

  IterType end() noexcept {
      return ptr_ + len_;
  }
};

template <typename IdType>
class IdRange {
  public:

    class Iter {
      private:
        IdType i_;

      public:
        Iter(IdType i_): i_(i_) {}

        IdType operator*() const { return i_; }

        bool operator==(const Iter &other) const { return i_ == other.i_; }

        bool operator!=(const Iter &other) const { return i_ != other.i_; }

        Iter operator++(int) {
          Iter old {*this};
          ++i_;
          return old;
        }

        Iter &operator++() {
          ++i_;
          return *this;
        }
    };

    Iter begin() { return Iter(min); }

    Iter end() { return Iter(max); }

    // vid ranged from [min, max)
    IdRange(IdType min, IdType max): min(min), max(max) {}

  private:
    const IdType min, max;
};

template <typename T>
T str2int(const char *ptr, char **end_ptr, int base) {
  static_assert(std::is_integral_v<T>);
  if (sizeof(T) <= sizeof(long)) {
    return strtol(ptr, end_ptr, base);
  } else {
    return strtoll(ptr, end_ptr, base);
  }
}

// template <typename ... Args>
// std::string format(const std::string fmt, Args ... args) {
//   int size_s = std::snprintf( nullptr, 0, fmt.c_str(), args ... ) + 1; // Extra space for '\0'
//   if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
//   auto size = static_cast<size_t>( size_s );
//   auto buf = std::make_unique<char[]>( size );
//   std::snprintf( buf.get(), size, fmt.c_str(), args ... );
//   return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
// }

void cudaNvtxStart(std::string msg);

void cudaNvtxStop();

void cudaHostSync();

void cudaHostSync(cudaStream_t stream);

} // namespace utils
