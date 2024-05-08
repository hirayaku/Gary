#pragma once

#include <vector>
#include <memory>
#include <tuple>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include "Defines.h"
#include "Graph.h"

#include "ops/DeviceIterator.cuh"

// Raw CSR representation of a graph (friendly to device code)
struct DeviceCSR {
  // members describing graph structures
  vidT m = 0;
  eidT nnz = 0;
  eidT *indPtr = nullptr;
  vidT *colIdx = nullptr;
  vidT *deg = nullptr;

  using VType = vidT;
  using EType = thrust::tuple<vidT, vidT>;

  HOST_DEVICE vidT numV() const { return m; }
  HOST_DEVICE eidT numE() const { return nnz; }
  HOST_DEVICE vidT getDegree(vidT v) const {
    return deg[v];
  }
  // vid: the source vertex ID of the requested edge
  // eno: the index of the edge in the adjacency list of vid; requires 0 <= eno < degree(vid)
  HOST_DEVICE EType getEdge(vidT vid, vidT eno) const { return {vid, colIdx[indPtr[vid]+eno]}; }

  // get all vertices
  HOST_DEVICE IdRange<vidT, void> V() const { return IdRange<vidT, void>(0, m); }

  // get neighbors of a vertex
  HOST_DEVICE Span<vidT, void> N(vidT v) const {
    return Span<vidT, void>(colIdx+indPtr[v], deg[v]);
  }
};

// Raw COO representation of a graph (friendly to device code)
struct DeviceCOO {
  // members describing graph structures
  vidT m = 0;
  eidT nnz = 0;
  vidT *rowIdx = nullptr;
  vidT *colIdx = nullptr;
  vidT *deg = nullptr;

  using VType = vidT;
  using EType = thrust::tuple<vidT, vidT>;

  HOST_DEVICE vidT numV() const { return m; }
  HOST_DEVICE eidT numE() const { return nnz; }
  HOST_DEVICE vidT getDegree(vidT v) const {
    return deg[v];
  }

  HOST_DEVICE EType getEdge(eidT eid) const { return {rowIdx[eid], colIdx[eid]}; }

  HOST_DEVICE IdRange<vidT, void> V() const { return IdRange<vidT, void>(0, m); }

  HOST_DEVICE IdRange<eidT, void> E() const { return IdRange<eidT, void>(0, nnz); }
};

struct DeviceGraph {
  DeviceCSR csr;
  DeviceCOO coo;
};

// Manages the graph data between host and device memory.
class DeviceGraphCtx {
public:
  DeviceGraphCtx(std::shared_ptr<GraphCSR> graph);

  // load host graph data (replace the current one if it exists)
  void loadFromHost(std::shared_ptr<GraphCSR> graph);

  // load host graph data (replace the current one if it exists)
  void loadFromHost(std::shared_ptr<GraphCOO> graph, bool sorted);

  // detach the host graph data
  void detachFromHost();

  // clear device data and hostGraph
  void clear();

  // create a new host graph copy from on-device data
  // if already detached, create new host graph objects
  std::shared_ptr<GraphCSR> getHostCSR();

  DeviceCSR getDeviceCSR() {
    return DeviceCSR {
      hostCSR->numV(), hostCSR->numE(),
      thrust::raw_pointer_cast(devPtr->data()),
      thrust::raw_pointer_cast(devColIdx->data()),
      thrust::raw_pointer_cast(devDeg->data())
    };
  }

  DeviceCOO getDeviceCOO() {
    if (devRowIdx == nullptr)
      this->devRowIdx = std::make_shared<thrust::device_vector<vidT>>(hostCOO->getRow());
    return DeviceCOO {
      hostCOO->numV(), hostCOO->numE(),
      thrust::raw_pointer_cast(devRowIdx->data()),
      thrust::raw_pointer_cast(devColIdx->data()),
      thrust::raw_pointer_cast(devDeg->data())
    };
  }

protected:
  std::shared_ptr<thrust::device_vector<eidT>> devPtr;
  std::shared_ptr<thrust::device_vector<vidT>> devRowIdx;
  std::shared_ptr<thrust::device_vector<vidT>> devColIdx;
  std::shared_ptr<thrust::device_vector<vidT>> devDeg;
  std::shared_ptr<GraphCSR> hostCSR;
  std::shared_ptr<GraphCOO> hostCOO;
};
