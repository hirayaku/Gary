#pragma once
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <string>
#include "Utils.h"

class Timer {
public:
  Timer() : Timer("null") {}
  Timer(const std::string &name) : name_(name) {}
  void start() {
    gettimeofday(&start_time_, NULL);
  }
  void stop() {
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
  }
  double seconds() const {
    return (double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1e6;
  }
  double millisecs() const {
    return (double)1000*(double)elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/(double)1000;
  }
  double microsecs() const {
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
  CUDATimer(const std::string& name) : Timer{name} {
    cudaDeviceSynchronize();
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
  }
  void start() {
    utils::cudaNvtxStart(name_);
    cudaEventRecord(start_event);
  }
  void stop() {
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&millisecs_, start_event, stop_event);
    utils::cudaNvtxStop();
  }
  double seconds() const {
    return (double)millisecs_ / 1000;
  }
  double millisecs() const {
    return millisecs_;
  }
  double microsecs() const {
    return 1e3 * (double)millisecs_;
  }

  ~CUDATimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }
private:
  cudaEvent_t start_event, stop_event;
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

inline uint64_t readTs() {
  volatile uint64_t t = 0;
#ifdef PROFILE_LATENCY
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
#endif
  return t;
}

