#pragma once

#include <vector>
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include "Defines.h"
#include "Graph.h"

#include "ops/DeviceIterator.cuh"

// Device-side CSR representation of a graph.
HOST_DEVICE struct DeviceGraph {
  // members describing graph structures
  vidT m = 0;
  eidT nnz = 0;
  eidT *indPtr = nullptr;
  // vidT *rowIdx = nullptr;
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

  // get all vertices and traverse in parallel
  template <typename CudaGroup>
  DEVICE_ONLY IdRange<vidT, CudaGroup> V(const CudaGroup &group) const {
    return IdRange<vidT, CudaGroup>(0, m, group);
  }

  HOST_DEVICE IdRange<vidT, void> V() const { return IdRange<vidT, void>(0, m); }

  // get neighbors of a vertex
  template <typename CudaGroup>
  DEVICE_ONLY Span<vidT *, CudaGroup> N(vidT v, const CudaGroup &group) const {
    return Span<vidT *, CudaGroup>(colIdx+indPtr[v], deg[v], group);
  }

  HOST_DEVICE Span<vidT *, void> N(vidT v) const {
    return Span<vidT *, void>(colIdx+indPtr[v], deg[v]);
  }
};

// Manages the graph data between host and device memory.
class DeviceGraphCtx {
public:
  DeviceGraphCtx(std::shared_ptr<GraphCSR> graph);

  // load host graph data (replace the current one if it exists)
  void loadFromHost(std::shared_ptr<GraphCSR> graph);

  // detach the host graph data
  void detachFromHost();

  // clear device data and hostGraph
  void clear();

  // create a new host graph copy from on-device data
  std::shared_ptr<GraphCSR> getHostGraph();
  thrust::host_vector<eidT> getDevicePtr() const;
  thrust::host_vector<vidT> getDeviceCol() const;
  thrust::host_vector<vidT> getDeviceDeg() const;

  DeviceGraph getDeviceGraph() {
    return DeviceGraph {
      hostGraph->numV(), hostGraph->numE(),
      thrust::raw_pointer_cast(devPtr.data()),
      thrust::raw_pointer_cast(devIdx.data()),
      thrust::raw_pointer_cast(devDeg.data())
    };
  }

protected:
  thrust::device_vector<eidT> devPtr;
  thrust::device_vector<vidT> devIdx;
  thrust::device_vector<vidT> devDeg;
  std::shared_ptr<GraphCSR> hostGraph;
};
