#pragma once

#include <vector>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>

using PinMemResource = thrust::cuda::universal_host_pinned_memory_resource;

template <typename T>
using PinMemAlloc = thrust::mr::stateless_resource_allocator<T, PinMemResource>;

template <typename T>
using PinMemVector = std::vector<T, PinMemAlloc<T>>;
