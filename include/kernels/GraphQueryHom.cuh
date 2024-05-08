#pragma once
#include "DeviceGraph.cuh"

namespace graph_query {
  __global__ void triangleQueryHom(DeviceGraph dataGraph);
  __global__ void rectQueryHom(DeviceGraph dataGraph);
}
