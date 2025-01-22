/* Copyright 2025 ComputerGraphics Tuebingen. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. Lensch
// converted to GGNN library by: Lukas Ruppert, Deborah Kornwolf

#include <ggnn/base/data.cuh>

#include <ggnn/cuda_utils/check.cuh>

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>

#include <cuda_runtime.h>

namespace ggnn {

namespace detail {

DataLocation disown(DataLocation location)
{
  switch (location) {
    case DataLocation::FOREIGN_GPU:
      DLOG(WARNING) << "ownership has already been removed";
    case DataLocation::GPU:
    case DataLocation::MANAGED:
      return DataLocation::FOREIGN_GPU;

    case DataLocation::FOREIGN_CPU:
      DLOG(WARNING) << "ownership has already been removed";
    case DataLocation::CPU_PINNED:
    case DataLocation::CPU_MALLOC:
      return DataLocation::FOREIGN_CPU;
    default:;
  }
  return DataLocation::UNKNOWN;
};

size_t dataSize(DataType type)
{
  switch (type) {
    case DataType::BYTE:
    case DataType::UINT8:
      return 1;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FLOAT:
      return 4;
    default:
      LOG(FATAL) << "size for data type " << type << " is unknown.";
  }
  return 0;
}

};  // namespace detail

std::byte* Allocator::cudaMallocChecked(const size_t size)
{
  CHECK_CUDA(cudaPeekAtLastError());
  std::byte* placeholder;
  const cudaError_t result = cudaMalloc(&placeholder, size);
  CHECK_EQ(result, cudaSuccess) << "failed to allocate " << size << " bytes of GPU memory.";
  return placeholder;
}
std::byte* Allocator::cudaMallocManagedChecked(const size_t size)
{
  CHECK_CUDA(cudaPeekAtLastError());
  std::byte* placeholder;
  const cudaError_t result = cudaMallocManaged(&placeholder, size);
  CHECK_EQ(result, cudaSuccess) << "failed to allocate " << size << " bytes of managed GPU memory.";
  return placeholder;
}
std::byte* Allocator::cudaMallocHostChecked(const size_t size, const unsigned int flags)
{
  CHECK_CUDA(cudaPeekAtLastError());
  std::byte* placeholder;
  const cudaError_t result = cudaMallocHost(&placeholder, size, flags);
  CHECK_EQ(result, cudaSuccess) << "failed to allocate " << size << " bytes of pinned CPU memory.";
  return placeholder;
}

std::byte* Allocator::mallocChecked(const size_t size)
{
  CHECK_CUDA(cudaPeekAtLastError());
  std::byte* placeholder = reinterpret_cast<std::byte*>(std::malloc(size));
  CHECK_NOTNULL(placeholder);
  return placeholder;
}

void Allocator::allocateData(Allocation& alloc, uint32_t flags)
{
  alloc.mem = [&alloc, &flags]() -> void* {
    switch (alloc.location) {
      case DataLocation::GPU:
        CHECK_CUDA(cudaSetDevice(alloc.gpu_id));
        return cudaMallocChecked(alloc.required_size_bytes());
        break;
      case DataLocation::MANAGED:
        CHECK_CUDA(cudaSetDevice(alloc.gpu_id));
        return cudaMallocManagedChecked(alloc.required_size_bytes());
        break;
      case DataLocation::CPU_PINNED:
        return cudaMallocHostChecked(alloc.required_size_bytes(), flags);
        break;
      case DataLocation::CPU_MALLOC:
        return mallocChecked(alloc.required_size_bytes());
        break;
      case DataLocation::FOREIGN_GPU:
      case DataLocation::FOREIGN_CPU:
      case DataLocation::UNKNOWN:
        LOG(ERROR) << "cannot allocate data to unknown or foreign location.";
    }
    return nullptr;
  }();
}

void Allocator::freeData(Allocation& alloc)
{
  if (alloc.mem) {
    switch (alloc.location) {
      case DataLocation::GPU:
      case DataLocation::MANAGED:
        CHECK_CUDA(cudaSetDevice(alloc.gpu_id));
        CHECK_CUDA(cudaFree(alloc.mem));
        break;
      case DataLocation::CPU_PINNED:
        CHECK_CUDA(cudaFreeHost(alloc.mem));
        break;
      case DataLocation::CPU_MALLOC:
        std::free(alloc.mem);
        break;
      case DataLocation::FOREIGN_CPU:
      case DataLocation::FOREIGN_GPU:
        break;  // noop
      default:
        LOG(WARNING) << "cannot free data from unknown origin.";
    }
    alloc.mem = nullptr;
  }
}

std::ostream& operator<<(std::ostream& stream, DataType type)
{
  switch (type) {
    case DataType::UNKNOWN:
      stream << "unknown";
      break;
    case DataType::BYTE:
      stream << "byte";
      break;
    case DataType::UINT8:
      stream << "uint8";
      break;
    case DataType::INT32:
      stream << "int32";
      break;
    case DataType::UINT32:
      stream << "uint32";
      break;
    case DataType::FLOAT:
      stream << "float";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, DataLocation location)
{
  switch (location) {
    case DataLocation::UNKNOWN:
      stream << "unknown";
      break;
    case DataLocation::GPU:
      stream << "GPU";
      break;
    case DataLocation::MANAGED:
      stream << "managed";
      break;
    case DataLocation::CPU_MALLOC:
      stream << "CPU (malloc)";
      break;
    case DataLocation::CPU_PINNED:
      stream << "CPU (pinned)";
      break;
    case DataLocation::FOREIGN_CPU:
      stream << "CPU (foreign)";
      break;
    case DataLocation::FOREIGN_GPU:
      stream << "GPU (foreign)";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, Allocation alloc)
{
  stream << "[alloc: " << alloc.N << "x" << alloc.D << ", " << alloc.type << ", " << alloc.location
         << "]";
  return stream;
}

};  // namespace ggnn
