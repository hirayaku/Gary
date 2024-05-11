#include <stdexcept>
#include <memory>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "HelperCUDA.h"
#include "ParUtils.h"
#include "Graph.h"
#include "DeviceGraph.cuh"


DeviceGraphCtx::DeviceGraphCtx(std::shared_ptr<GraphCSR> graph) {
  this->loadFromHost(graph);
}

void DeviceGraphCtx::loadFromHost(std::shared_ptr<GraphCSR> graph) {
  makeDeviceCurrent_();

  hostCSR = std::move(graph);
  // make hostCOO consistent with hostCSR
  if (hostCOO != nullptr)
    hostCOO = std::make_shared<GraphCOO>(hostCSR->toCOO());
  this->devPtr = std::make_shared<thrust::device_vector<eidT>>(hostCSR->getPtr());
  this->devColIdx = std::make_shared<thrust::device_vector<vidT>>(hostCSR->getIdx());
  this->devDeg = std::make_shared<thrust::device_vector<vidT>>(hostCSR->getDegreeAll());
}

void DeviceGraphCtx::loadFromHost(std::shared_ptr<GraphCOO> graph, bool sorted) {
  hostCSR = std::make_shared<GraphCSR>(graph->toCSR_(sorted));
  hostCOO = std::move(graph);
  loadFromHost(hostCSR);
}

void DeviceGraphCtx::populateDevice() {
  if (hostCSR == nullptr)
    throw std::runtime_error("DeviceGraphCtx is not initialized");
  // populate CSR data
  if (devPtr == nullptr) {
    loadFromHost(hostCSR);
  }
  // populate the remaining COO data
  if (devRowIdx == nullptr) {
    devRowIdx = std::make_shared<thrust::device_vector<vidT>>(getHostCOO()->getRow());  
  }
}

void DeviceGraphCtx::detachFromHost() {
  hostCSR.reset();
  hostCOO.reset();
}

void DeviceGraphCtx::clear() {
  makeDeviceCurrent_();
  this->detachFromHost();
  this->devPtr.reset();
  this->devRowIdx.reset();
  this->devColIdx.reset();
  this->devDeg.reset();
}

std::shared_ptr<GraphCSR> DeviceGraphCtx::getHostCSR() {
  if (this->hostCSR == nullptr) {
    if (this->devPtr == nullptr) {
      throw std::runtime_error("DeviceGraphCtx is not initialized");
    }
    makeDeviceCurrent_();
    vidT m = devPtr->size() - 1;
    eidT nnz = devColIdx->size();
    std::vector<eidT> hostPtr(m);
    std::vector<vidT> hostColIdx(nnz);
    thrust::copy_n(devPtr->begin(), m, hostPtr.begin());
    thrust::copy_n(devColIdx->begin(), nnz, hostColIdx.begin());
    this->hostCSR = std::make_shared<GraphCSR>(m, nnz,
      std::make_shared<std::vector<eidT>>(std::move(hostPtr)),
      std::make_shared<std::vector<vidT>>(std::move(hostColIdx))
    );
  }
  return this->hostCSR;
}

std::shared_ptr<GraphCOO> DeviceGraphCtx::getHostCOO() {
  if (this->hostCOO == nullptr) {
    auto csr = getHostCSR();
    this->hostCOO = std::make_shared<GraphCOO>(csr->toCOO());
  }
  return this->hostCOO;
}

void DeviceGraphCtx::makeDeviceCurrent_() {
  int devId = -1;
  checkCudaErrors(cudaGetDevice(&devId));
  if (this->devId != devId)
    checkCudaErrors(cudaSetDevice(devId));
}
