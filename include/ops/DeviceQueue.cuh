#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/std/tuple>
#include <cuda/std/limits>
#include <cuda/atomic>
#include "Defines.h"
#include "ops/DeviceIterator.cuh"

namespace cg = cooperative_groups;

constexpr cuda::memory_order mSeqCst = cuda::memory_order_seq_cst;
constexpr cuda::memory_order mRelease = cuda::memory_order_release;
constexpr cuda::memory_order mAcquire = cuda::memory_order_acquire;
constexpr cuda::memory_order mAcqRel = cuda::memory_order_acq_rel;
constexpr cuda::memory_order mRelaxed = cuda::memory_order_relaxed;

static inline HOST_DEVICE uint64_t &pack(unsigned u32[2]) {
  return *reinterpret_cast<uint64_t *>(u32);
}
static inline HOST_DEVICE cuda::std::tuple<unsigned, unsigned> unpack(uint64_t u64) {
  auto u32_cast = reinterpret_cast<unsigned *>(&u64);
  return { u32_cast[0], u32_cast[1] };
}

static CUDA_INLINE DEVICE_ONLY void pause(unsigned ns) {
  __nanosleep(ns);
}

// A shared queue for threads in a CTA (thread block)
// When `XclEnqDeq` is true, only enq OR deq is supported at the same time
template <typename T = vidT, unsigned N = 128, bool XclEnqDeq=false>
class CTAQueue {
public:
  static constexpr T NIL = cuda::std::numeric_limits<T>::max();

  // the queue data could be in shared memory or global memory
  // requires a subsequent thread block synchronize
  DEVICE_ONLY CTAQueue(T* data, unsigned pos[2])
  : data(data), pos(pos),
    thisBlock(cg::this_thread_block()),
    thisWarp(cg::tiled_partition<WARP_SIZE>(thisBlock)),
    warpRank(thisWarp.meta_group_rank()),
    threadRank(thisBlock.thread_rank()),
    atomicPos{(unsigned &)pos[0], (unsigned &)pos[1]},
    atomicPos2(pack((unsigned *)pos))
  {
    static_assert(XclEnqDeq || cuda::atomic_ref<T, cuda::thread_scope_block>::is_always_lock_free,
      "Data type in queue is not lockfree");
    assert (!__isLocal(data) && "Can't initialize CTAQueue with local memory");
    clear();
  }

  // reset head & tail pointers, as well as the `data` array if necessary
  // CTA cooperative
  DEVICE_ONLY void clear() {
    if (!XclEnqDeq) {
      for (unsigned idx : IdRange<unsigned,void>(N).cooperate(thisBlock))
        data[idx] = NIL;
    }
    if (threadRank == 0) {
      atomicPos2.store(0, mSeqCst);
    }
  }

  // fill the data array with source data at `dataSrc` with size `n`
  // CTA cooperative
  DEVICE_ONLY void fill(T *dataSrc, unsigned n) {
    n = (n < N)? n : N;
    if (threadRank == 0) {
      unsigned array[2] {n, 0};
      atomicPos2.store(pack(array), mSeqCst);
    }
    cg::memcpy_async(thisBlock, data, dataSrc, sizeof(T) * n);
    cg::wait(thisBlock);
  }

  DEVICE_ONLY constexpr int capacity() { return N; }

  CUDA_INLINE DEVICE_ONLY unsigned size() const {
    unsigned head = 0, tail = 0;
    cuda::std::tie(head, tail) = unpack(atomicPos2.load(mRelaxed));
    // tail could overshoot during dequeue, so we take a max of 0
    return std::max(static_cast<int>(head - tail), 0);
  }

  // enqueue one item into the queue
  // returns false if the attempt fails
  // template <typename = std::enable_if_t<!XclEnqDeq, void>>
  CUDA_INLINE DEVICE_ONLY bool tryEnq(T &&item) {
    if (size() >= N) return false;
    unsigned enqPos = head().fetch_add(1, mRelaxed);
    if (enqPos >= tail().load(mRelaxed) + N) {
      // overshoot because of the race condition
      head().fetch_sub(1, mRelaxed);
      return false;
    }
    // atomic enqueue data
    cuda::atomic_ref<T, cuda::thread_scope_block> dataRef(data[enqPos%N]);
    unsigned ns = 8;
    for (T expected = NIL; !dataRef.compare_exchange_weak(expected, item, mRelease, mRelaxed); expected = NIL) {
      do {
        // exponential backoff
        pause(ns);
        ns = (ns < 256) ? ns * 2 : ns;
      } while (dataRef.load(mRelaxed) != NIL);
    }
    return true;
  }

  // dequeue one item form the queue
  // returns false if the attempt fails
  // template <typename = std::enable_if_t<!XclEnqDeq, void>>
  CUDA_INLINE DEVICE_ONLY bool tryDeq(T &item) {
    if (size() == 0) return false;
    unsigned deqPos = tail().fetch_add(1, mRelaxed);
    if (deqPos >= head().load(mRelaxed)) {
      // overshoot because of the race condition
      tail().fetch_sub(1, mRelaxed);;
      return false;
    }
    // atomic dequeue data
    cuda::atomic_ref<T, cuda::thread_scope_block> dataRef(data[deqPos%N]);
    unsigned ns = 8;
    while (true) {
      item = dataRef.exchange(NIL, mAcquire);
      if (item != NIL) return true;
      do {
        // exponential backoff
        pause(ns);
        ns = (ns < 256) ? ns * 2 : ns;
      } while (dataRef.load(mRelaxed) == NIL);
    }
  }

  // enqueue one item into the queue (exclusive, not to be mixed with concurrent dequeue)
  // returns false if the attempt fails
  CUDA_INLINE DEVICE_ONLY bool tryEnqX(T &&item) {
    unsigned enqPos = head().fetch_add(1, mRelaxed);
    if (enqPos >= tail().load(mRelaxed) + N) {
      // overshoot because of the race condition
      head().fetch_sub(1, mRelaxed);
      return false;
    }
    // store to the reserved slot
    data[enqPos % N] = cuda::std::forward<T>(item);
    return true;
  }

  // dequeue one item form the queue (exclusive, not to be mixed with concurrent enqueue)
  // returns false if the attempt fails
  CUDA_INLINE DEVICE_ONLY bool tryDeqX(T &item) {
    unsigned deqPos = tail().fetch_add(1, mRelaxed);
    if (deqPos >= head().load(mRelaxed)) {
      // overshoot because of the race condition
      tail().fetch_sub(1, mRelaxed);;
      return false;
    }
    // load from the reserved slot
    item = data[deqPos % N];
    return true;
  }

private:
  T* const data;
  // { head, tail } are absolute positions (mod N not taken)
  // head is exclusive, tail is inclusive
  // we could always take mod N when we read/write from `data` array
  unsigned *pos;

  const cg::thread_block &thisBlock;
  const utils::thread_warp &thisWarp;
  const int warpRank;
  const int threadRank;

  cuda::atomic_ref<unsigned, cuda::thread_scope_block> atomicPos[2];
  cuda::atomic_ref<uint64_t, cuda::thread_scope_block> atomicPos2;

  DEVICE_ONLY cuda::atomic_ref<unsigned, cuda::thread_scope_block> &head() { return atomicPos[0];}
  DEVICE_ONLY cuda::atomic_ref<unsigned, cuda::thread_scope_block> &tail() { return atomicPos[1];}
};
