#include <stdexcept>
#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "ParUtils.h"
#include "Graph.h"
#include "DeviceGraph.cuh"


DeviceGraphCtx::DeviceGraphCtx(std::shared_ptr<GraphCSR> graph) {
  this->loadFromHost(graph);
}

void DeviceGraphCtx::loadFromHost(std::shared_ptr<GraphCSR> graph) {
  hostCSR = std::move(graph);
  // we must make new COO to enable sharing of the column idx on device 
  hostCOO = std::make_shared<GraphCOO>(hostCSR->toCOO());
  this->devPtr = std::make_shared<thrust::device_vector<eidT>>(hostCSR->getPtr());
  this->devColIdx = std::make_shared<thrust::device_vector<vidT>>(hostCSR->getIdx());
  this->devDeg = std::make_shared<thrust::device_vector<vidT>>(hostCSR->getDegreeAll());
}

void DeviceGraphCtx::loadFromHost(std::shared_ptr<GraphCOO> graph, bool sorted=false) {
  hostCSR = std::make_shared<GraphCSR>(graph->toCSR_(sorted));
  hostCOO = std::move(graph);
}

void DeviceGraphCtx::detachFromHost() {
  hostCSR.reset();
  hostCOO.reset();
}

void DeviceGraphCtx::clear() {
  this->detachFromHost();
  this->devPtr.reset();
  this->devRowIdx.reset();
  this->devColIdx.reset();
  this->devDeg.reset();
}

std::shared_ptr<GraphCSR> DeviceGraphCtx::getHostCSR() {
  if (this->hostCSR == nullptr) {
    vidT m = devPtr->size() - 1;
    eidT nnz = devColIdx->size();
    std::vector<eidT> hostPtr(m);
    std::vector<vidT> hostColIdx(nnz);
    thrust::copy_n(devPtr->begin(), m, hostPtr.begin());
    thrust::copy_n(devColIdx->begin(), nnz, hostColIdx.begin());
    loadFromHost(std::make_shared<GraphCSR>(m, nnz,
      std::make_shared<std::vector<eidT>>(std::move(hostPtr)),
      std::make_shared<std::vector<vidT>>(std::move(hostColIdx))
      )
    );
  }

  return this->hostCSR;
}
