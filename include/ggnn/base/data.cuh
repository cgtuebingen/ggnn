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

#ifndef INCLUDE_GGNN_DATA_CUH
#define INCLUDE_GGNN_DATA_CUH

#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace ggnn {

enum class DataType : uint16_t {
  UNKNOWN,
  BYTE,
  UINT8,
  INT32,
  UINT32,
  FLOAT
};

enum class DataLocation : uint16_t {
  UNKNOWN,      //< unset
  GPU,          //< regular GPU memory
  MANAGED,      //< managed GPU memory
  CPU_PINNED,   //< pinned CPU memory
  CPU_MALLOC,   //< regular/pageable CPU memory
  FOREIGN_GPU,  //< foreign GPU memory - do not free
  FOREIGN_CPU,  //< foreign CPU memory - do not free
};

std::ostream& operator<<(std::ostream& stream, DataType type);
std::ostream& operator<<(std::ostream& stream, DataLocation location);

namespace detail {

template <typename T>
struct DataTypeAssignment;

template <>
struct DataTypeAssignment<std::byte> {
  static constexpr DataType value{DataType::BYTE};
};
template <>
struct DataTypeAssignment<uint8_t> {
  static constexpr DataType value{DataType::UINT8};
};
template <>
struct DataTypeAssignment<int32_t> {
  static constexpr DataType value{DataType::INT32};
};
template <>
struct DataTypeAssignment<uint32_t> {
  static constexpr DataType value{DataType::UINT32};
};
template <>
struct DataTypeAssignment<float> {
  static constexpr DataType value{DataType::FLOAT};
};

DataLocation disown(DataLocation location);
size_t dataSize(DataType type);

};  // namespace detail

template <typename T>
constexpr DataType DataType_v = detail::DataTypeAssignment<T>::value;

struct Allocation {
  uint64_t N{};
  uint32_t D{};

  DataType type{};
  DataLocation location{};
  int32_t gpu_id{};

  void* mem{nullptr};

  // transfer ownership away from this object
  // when this object is to be deallocated by the GPUContext, deallocation will be skipped
  void releaseOwnership()
  {
    location = detail::disown(location);
  }

  bool isCPUAccessible() const
  {
    switch (location) {
      case DataLocation::CPU_MALLOC:
      case DataLocation::CPU_PINNED:
      case DataLocation::FOREIGN_CPU:
      case DataLocation::MANAGED:
        return true;
      default:;
    }
    return false;
  }

  bool isGPUAccessible() const
  {
    switch (location) {
      case DataLocation::GPU:
      case DataLocation::FOREIGN_GPU:
      case DataLocation::MANAGED:
        return true;
      default:;
    }
    return false;
  }

  size_t element_size() const
  {
    return detail::dataSize(type);
  }

  size_t numel() const
  {
    return static_cast<size_t>(N) * D;
  }

  size_t required_size_bytes() const
  {
    return element_size() * numel();
  }

  explicit operator void*()
  {
    return mem;
  }

  explicit operator const void*() const
  {
    return mem;
  }
};

struct Allocator {
  static std::byte* cudaMallocChecked(const size_t size);
  static std::byte* cudaMallocManagedChecked(const size_t size);
  static std::byte* cudaMallocHostChecked(const size_t size, const unsigned int flags);
  static std::byte* mallocChecked(const size_t size);
  static void allocateData(Allocation& alloc, uint32_t flags = 0);
  static void freeData(Allocation& alloc);
};

std::ostream& operator<<(std::ostream& stream, Allocation alloc);

};  // namespace ggnn

#endif  // INCLUDE_GGNN_DATA_CUH
