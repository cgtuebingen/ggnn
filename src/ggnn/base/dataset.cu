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
#include <ggnn/base/dataset.cuh>

#include <ggnn/base/lib.h>

#include <ggnn/cuda_utils/check.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <span>
#include <type_traits>
#include <vector>

#include <glog/logging.h>

#include <cuda_runtime.h>

namespace ggnn {

GenericDataset::GenericDataset(GenericDataset&& other) noexcept : Allocation{other}
{
  other.releaseOwnership();
}

GenericDataset& GenericDataset::operator=(GenericDataset&& other) noexcept
{
  Allocator::freeData(*this);
  Allocation::operator=(other);
  other.releaseOwnership();
  return *this;
}

GenericDataset::~GenericDataset()
{
  Allocator::freeData(*this);
}

GenericDataset GenericDataset::reference() const
{
  return Allocation{.N = N,
                    .D = D,
                    .type = type,
                    .location = detail::disown(location),
                    .gpu_id = gpu_id,
                    .mem = mem};
}
GenericDataset GenericDataset::referenceRange(uint64_t from, uint64_t num) const
{
  CHECK_LE(from + num, N);
  return Allocation{.N = num,
                    .D = D,
                    .type = type,
                    .location = detail::disown(location),
                    .gpu_id = gpu_id,
                    .mem = reinterpret_cast<std::byte*>(mem) + from * D * detail::dataSize(type)};
}

template <typename T>
Dataset<T> Dataset<T>::empty(const uint64_t N, const uint32_t D, bool pin_memory)
{
  Allocation alloc{.N = N,
                   .D = D,
                   .type = DataType_v<T>,
                   .location = pin_memory ? DataLocation::CPU_PINNED : DataLocation::CPU_MALLOC};
  Allocator::allocateData(alloc);

  Dataset<T> result{alloc};
  CHECK_NOTNULL(result.data());
  return result;
}

template <typename T>
Dataset<T> Dataset<T>::emptyOnGPU(const uint64_t N, const uint32_t D, int32_t gpu_id)
{
  Allocation alloc{
      .N = N, .D = D, .type = DataType_v<T>, .location = DataLocation::GPU, .gpu_id = gpu_id};
  Allocator::allocateData(alloc);

  Dataset<T> result{alloc};
  CHECK_NOTNULL(result.data());
  return result;
}

template <typename T>
Dataset<T> Dataset<T>::copy(const std::span<const T>& data, uint32_t D, bool pin_memory)
{
  const uint32_t N = data.size() / D;
  CHECK_EQ(N * D, data.size());

  Dataset<T> result{empty(N, D, pin_memory)};
  std::copy(data.begin(), data.end(), result.data());

  return result;
}

GenericDataset GenericDataset::load(const std::filesystem::path& path, uint32_t from, uint32_t num,
                                    bool pin_memory)
{
  if (path.string().ends_with(".fvecs"))
    return GenericDataset{Dataset<float>::load(path, from, num, pin_memory)};
  else if (path.string().ends_with(".bvecs"))
    return GenericDataset{Dataset<uint8_t>::load(path, from, num, pin_memory)};
  else if (path.string().ends_with(".ivecs"))
    return GenericDataset{Dataset<int32_t>::load(path, from, num, pin_memory)};
  LOG(FATAL) << "Could not guess file type from " << path
             << ". fvecs, bvecs, or ivecs file required.";
  // to avoid "missing return" - the above LOG(FATAL) will already call abort
  abort();
}

template <typename T>
Dataset<T> Dataset<T>::load(const std::filesystem::path& path, uint32_t from, uint32_t num,
                            bool pin_memory)
{
  std::ifstream file{path, std::ios_base::in | std::ios_base::binary};
  CHECK(file) << "Unable to open file " << path << " for reading.";

  uint32_t D{};
  // read dimension
  file.read(reinterpret_cast<char*>(&D), sizeof(D));

  const size_t in_vec_size = sizeof(uint32_t) + D * sizeof(T);

  // calc file size
  file.seekg(0, std::ios::end);
  std::streampos fsize = file.tellg();

  uint32_t N = fsize / in_vec_size;
  file.seekg(0, std::ios::beg);

  VLOG(1) << "Opened dataset file " << path << " containing " << N << "x" << D << " vectors.";

  if (num != -1U) {
    N = std::min(N - from, num);
    CHECK_EQ(N, num) << "Dataset contains fewer vectors than requested.";
  }

  // read in blocks of 1'000 vectors from the file
  const uint32_t blocksize = 1'000;
  const uint32_t num_blocks = (N + blocksize - 1) / blocksize;
  const bool report_progress = N >= 100'000'000;
  const uint32_t report_every = 1'000;

  if (report_progress)
    std::cout << "\n[CPU] Allocating memory..." << std::flush;
  Dataset<T> result{empty(N, D, pin_memory)};
  std::vector<std::byte> buffer(blocksize * in_vec_size);

  for (uint32_t block = 0; block < num_blocks; ++block) {
    const size_t out_pos = static_cast<size_t>(block) * blocksize;
    const size_t in_pos = from + out_pos;

    file.seekg(in_vec_size * in_pos);
    file.read(reinterpret_cast<char*>(buffer.data()),
              std::min(blocksize, N - block * blocksize) * in_vec_size);

    for (size_t i = 0; i < blocksize && out_pos + i < N; ++i)
      std::copy_n(buffer.data() + in_vec_size * i + sizeof(uint32_t), D * sizeof(T),
                  reinterpret_cast<std::byte*>(&result[(out_pos + i) * D]));

    if (report_progress) {
      if (block % (report_every * 10) == 0) {
        std::cout << "\r[";
        std::cout.fill('0');
        std::cout.width(2);
        std::cout << block * 100 / num_blocks;
        std::cout << "%] Loading...\033[K" << std::flush;
      }
      else if (block % report_every == 0)
        std::cout << '.' << std::flush;
    }
  }

  if (report_progress)
    std::cout << "\r\033[K" << std::flush;

  CHECK(file) << "Failed to read vectors from " << path << ".";

  return result;
}

template <typename T>
Dataset<T> Dataset<T>::referenceCPUData(T* data, const uint64_t N, const uint32_t D)
{
  return GenericDataset{Allocation{
      .N = N, .D = D, .type = DataType_v<T>, .location = DataLocation::FOREIGN_CPU, .mem = data}};
}

template <typename T>
Dataset<T> Dataset<T>::referenceGPUData(T* data, const uint64_t N, const uint32_t D, int32_t gpu_id)
{
  return GenericDataset{Allocation{.N = N,
                                   .D = D,
                                   .type = DataType_v<T>,
                                   .location = DataLocation::FOREIGN_GPU,
                                   .gpu_id = gpu_id,
                                   .mem = data}};
}

template <typename T>
void Dataset<T>::store(const std::filesystem::path& path) const
{
  std::ofstream file{path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc};
  CHECK(file) << "Unable to open file " << path << " for writing.";

  static_assert(std::is_same_v<decltype(D), uint32_t>);
  for (size_t n = 0; n < N; ++n) {
    file.write(reinterpret_cast<const char*>(&D), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(this->data() + n * D), sizeof(T) * D);
  }
}

template <typename T>
void Dataset<T>::copyTo(Dataset& other, cudaStream_t stream) const
{
  CHECK_NOTNULL(this->data());
  CHECK_NOTNULL(other.data());
  CHECK_GE(other.size_bytes(), this->size_bytes());
  if (this->isCPUAccessible() && other.isCPUAccessible()) {
    std::copy(this->begin(), this->end(), other.begin());
  }
  else {
    cudaMemcpyKind copyType = cudaMemcpyDefault;
    if (this->isGPUAccessible() && other.isGPUAccessible()) {
      if (this->gpu_id == other.gpu_id) {
        copyType = cudaMemcpyDeviceToDevice;
      }
      else {
        // TODO: cudaMemcpyPeer where possible
        VLOG(4) << "Copying between different GPUs is not supported - copying through CPU instead.";
        Dataset<T> temp = Dataset<T>::empty(N, D, true);
        copyTo(temp, stream);
        temp.copyTo(other, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        return;
      }
    }
    else if (this->isCPUAccessible() && other.isGPUAccessible())
      copyType = cudaMemcpyHostToDevice;
    else if (this->isGPUAccessible() && other.isCPUAccessible())
      copyType = cudaMemcpyDeviceToHost;
    CHECK_CUDA(cudaMemcpyAsync(other.data(), this->data(), this->size_bytes(), copyType, stream));
  }
}

template <typename T>
void Dataset<T>::copyRangeTo(uint64_t from, uint64_t num, Dataset& other, cudaStream_t stream) const
{
  CHECK_NOTNULL(this->data());
  CHECK_NOTNULL(other.data());
  const size_t copySize = num * D * sizeof(T);
  CHECK_GE(other.size_bytes(), copySize);
  if (this->isCPUAccessible() && other.isCPUAccessible()) {
    std::copy_n(this->begin() + from * D, num * D, other.begin());
  }
  else {
    cudaMemcpyKind copyType = cudaMemcpyDefault;
    if (this->isGPUAccessible() && other.isGPUAccessible()) {
      if (this->gpu_id == other.gpu_id) {
        copyType = cudaMemcpyDeviceToDevice;
      }
      else {
        // TODO: cudaMemcpyPeer where possible
        VLOG(4) << "Copying between different GPUs is not supported - copying through CPU instead.";
        Dataset<T> temp = Dataset<T>::empty(num, D, true);
        copyRangeTo(from, num, temp, stream);
        temp.copyTo(other, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        return;
      }
    }
    else if (this->isCPUAccessible() && other.isGPUAccessible())
      copyType = cudaMemcpyHostToDevice;
    else if (this->isGPUAccessible() && other.isCPUAccessible())
      copyType = cudaMemcpyDeviceToHost;
    CHECK_CUDA(cudaMemcpyAsync(other.data(), this->data() + from * D, copySize, copyType, stream));
  }
}

template <typename T>
Dataset<T> Dataset<T>::clone(cudaStream_t stream) const
{
  Dataset<T> result;
  switch (location) {
    case DataLocation::FOREIGN_GPU:
    case DataLocation::GPU:
    case DataLocation::MANAGED:
      result = Dataset<T>::emptyOnGPU(N, D, gpu_id);
      break;
    case DataLocation::FOREIGN_CPU:
    case DataLocation::CPU_MALLOC:
    case DataLocation::CPU_PINNED:
      result = Dataset<T>::empty(N, D, location == DataLocation::CPU_PINNED);
      break;
    default:
      break;
  }
  CHECK_NOTNULL(result.data());
  copyTo(result, stream);
  return result;
}

template <typename T>
Dataset<T> Dataset<T>::referenceOnGPU(int gpu_id, cudaStream_t stream) const
{
  if (this->isGPUAccessible() && this->gpu_id == gpu_id)
    return Dataset{reference()};

  Dataset result = emptyOnGPU(N, D, gpu_id);
  copyTo(result, stream);
  return result;
}

template struct Dataset<std::byte>;
template struct Dataset<uint8_t>;
template struct Dataset<int32_t>;
template struct Dataset<uint32_t>;
template struct Dataset<float>;

};  // namespace ggnn
