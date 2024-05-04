#pragma once
#include <sys/time.h>
#include <string>

namespace utils {
class Timer {
public:
  Timer() : Timer("null") {}
  Timer(const std::string &name) : name_(name) {}
  virtual void start() {
    gettimeofday(&start_time_, NULL);
  }
  virtual void stop() {
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
  }
  virtual double seconds() const {
    return (double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1e6;
  }
  virtual double millisecs() const {
    return (double)1000*(double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/(double)1000;
  }
  virtual double microsecs() const {
    return 1e6*(double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec;
  }
  std::string getName() { return name_; }
  // std::cout << "runtime[" << t.getName() << "] = " << t.seconds() << " sec\n";
protected:
  std::string name_;
private:
  struct timeval start_time_;
  struct timeval elapsed_time_;
};

#include <cuda_runtime.h>

class CUDATimer: public Timer {
 public:
  CUDATimer(): CUDATimer("null") {}
  CUDATimer(const std::string &name);
  CUDATimer(const std::string &name, cudaStream_t stream);
  ~CUDATimer();

  void start() override;
  void stop() override;

  double seconds() const override {
    return (double)millisecs_ / 1000;
  }
  double millisecs() const override {
    return millisecs_;
  }
  double microsecs() const override {
    return 1e3 * (double)millisecs_;
  }

  cudaStream_t stream() const { return stream_; }

private:
  cudaEvent_t start_event, stop_event;
  cudaStream_t stream_ = 0;
  float millisecs_ = 0;
};

// Times op's execution using the timer t
#define TIME_IT(t, op) \
  t.start(); (op); t.stop();

template <typename T, typename F>
T timeIt(const F& f, const std::string name) {
  T t(name);
  t.start();
  f();
  t.stop();
  return t;
}

#ifdef __x86_64__
static inline uint64_t readTs() {
  volatile uint64_t t = 0;
  asm __volatile__(
      "lfence\n"
      // Guaranteed to clear the high-order 32 bits of RAX and RDX.
      "rdtsc\n"
      "shlq $32, %%rdx\n"
      "orq %%rdx, %%rax\n"
      : "=a" (t)
      :
      : "%rdx"
      );
  return t;
}
#endif

} // namespace utils
