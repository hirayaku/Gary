#include <stdexcept>
#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "Graph.h"
#include "DeviceGraph.cuh"


DeviceGraphCtx::DeviceGraphCtx(std::shared_ptr<GraphCSR> graph) {
  this->loadFromHost(hostGraph);
}

void DeviceGraphCtx::loadFromHost(std::shared_ptr<GraphCSR> graph) {
  this->hostGraph = std::move(graph);
  this->devPtr = hostGraph->getPtr();
  this->devIdx = hostGraph->getIdx();
  this->devDeg = hostGraph->getDegreeAll();
}

void DeviceGraphCtx::detachFromHost() {
  this->hostGraph = nullptr;
}

void DeviceGraphCtx::clear() {
  this->detachFromHost();
  this->devPtr.clear();
  this->devIdx.clear();
  this->devDeg.clear();
}

std::shared_ptr<GraphCSR> DeviceGraphCtx::getHostGraph() {
  if (this->hostGraph == nullptr) {
    vidT m = devPtr.size() - 1;
    eidT nnz = devIdx.size();
    std::vector<eidT> hostPtr(m);
    std::vector<vidT> hostIdx(nnz);
    thrust::copy_n(devPtr.begin(), m, hostPtr.begin());
    thrust::copy_n(devIdx.begin(), nnz, hostIdx.begin());
    return std::make_shared<GraphCSR>(m, nnz,
      std::make_shared<std::vector<eidT>>(std::move(hostPtr)),
      std::make_shared<std::vector<vidT>>(std::move(hostIdx))
    );
  } else {
    return this->hostGraph;
  }
}

thrust::host_vector<eidT> DeviceGraphCtx::getDevicePtr() const { return devPtr; }
thrust::host_vector<vidT> DeviceGraphCtx::getDeviceCol() const { return devIdx; }
thrust::host_vector<vidT> DeviceGraphCtx::getDeviceDeg() const { return devDeg; }
