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

#ifndef INCLUDE_GGNN_DATASET_CUH
#define INCLUDE_GGNN_DATASET_CUH

#include <ggnn/base/data.cuh>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace ggnn {

struct GenericDataset : protected Allocation {
  GenericDataset() : Allocation{} {}
  GenericDataset(const Allocation& alloc) : Allocation{alloc} {}
  GenericDataset(const GenericDataset& other) = delete;
  GenericDataset& operator=(const GenericDataset& other) = delete;
  explicit GenericDataset(GenericDataset&& other) noexcept;
  GenericDataset& operator=(GenericDataset&& other) noexcept;
  virtual ~GenericDataset();

  using Allocation::D;
  using Allocation::element_size;
  using Allocation::gpu_id;
  using Allocation::location;
  using Allocation::N;
  using Allocation::numel;
  using Allocation::type;

  GenericDataset reference() const;
  GenericDataset referenceRange(uint64_t from, uint64_t num) const;

  using Allocation::isCPUAccessible;
  using Allocation::isGPUAccessible;
  using Allocation::releaseOwnership;

  using Allocation::required_size_bytes;

  template <typename T>
  std::span<T> reinterpret()
  {
    return {reinterpret_cast<T*>(mem), numel()};
  }
  template <typename T>
  std::span<const T> reinterpret() const
  {
    return {reinterpret_cast<const T*>(mem), numel()};
  }

  template <typename T>
  std::span<T> access()
  {
    assert(DataType_v<T> == type);
    return reinterpret<T>();
  }
  template <typename T>
  std::span<const T> access() const
  {
    assert(DataType_v<T> == type);
    return reinterpret<T>();
  }

  static GenericDataset load(const std::filesystem::path& file, uint32_t from = 0,
                             uint32_t num = std::numeric_limits<uint32_t>::max(),
                             bool pin_memory = false);
};

template <typename T>
struct Dataset : public GenericDataset, public std::span<T> {
  Dataset() : GenericDataset{}, std::span<T>(reinterpret_cast<T*>(0), 0UL) {}
  Dataset(GenericDataset&& data)
      : GenericDataset{std::move(data)}, std::span<T>{GenericDataset::access<T>()}
  {
  }
  Dataset(Dataset& other) = delete;
  Dataset& operator=(Dataset& other) = delete;
  Dataset(Dataset&& other) noexcept = default;
  Dataset& operator=(Dataset&& other) noexcept = default;
  virtual ~Dataset() = default;

  using GenericDataset::D;
  using GenericDataset::element_size;
  using GenericDataset::gpu_id;
  using GenericDataset::location;
  using GenericDataset::N;
  using GenericDataset::numel;
  using GenericDataset::type;

  using GenericDataset::isCPUAccessible;
  using GenericDataset::isGPUAccessible;
  using GenericDataset::reference;
  using GenericDataset::referenceRange;
  using GenericDataset::releaseOwnership;

  operator T*()
  {
    return std::span<T>::data();
  }
  operator const T*() const
  {
    return std::span<T>::data();
  }

#if __cpp_lib_span < 202311L
  T& at(size_t index)
  {
    if (index >= this->size())
      throw std::out_of_range("Index " + std::to_string(index) + " is out of bounds (size " +
                              std::to_string(this->size()) + ").");
    return (*this)[index];
  }
  const T& at(size_t index) const
  {
    if (index >= this->size())
      throw std::out_of_range("Index " + std::to_string(index) + " is out of bounds (size " +
                              std::to_string(this->size()) + ").");
    return (*this)[index];
  }
#endif

  static Dataset empty(const uint64_t N, const uint32_t D, bool pin_memory = false);
  static Dataset emptyOnGPU(const uint64_t N, const uint32_t D, int32_t gpu_id);
  static Dataset copy(const std::span<const T>& data, uint32_t D, bool pin_memory = false);
  static Dataset load(const std::filesystem::path& file, uint32_t from = 0,
                      uint32_t num = std::numeric_limits<uint32_t>::max(), bool pin_memory = false);
  static Dataset referenceCPUData(T* data, const uint64_t N, const uint32_t D);
  static Dataset referenceGPUData(T* data, const uint64_t N, const uint32_t D, int32_t gpu_id);

  void store(const std::filesystem::path& path) const;
  void copyTo(Dataset& other, cudaStream_t stream = 0) const;
  void copyRangeTo(uint64_t from, uint64_t num, Dataset& other, cudaStream_t stream = 0) const;

  Dataset clone(cudaStream_t stream = 0) const;
  Dataset referenceOnGPU(int gpu_id, cudaStream_t stream = 0) const;
};

template <typename KeyT, typename ValueT>
struct Results {
  Dataset<KeyT> ids{};
  Dataset<ValueT> dists{};
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_DATASET_CUH
