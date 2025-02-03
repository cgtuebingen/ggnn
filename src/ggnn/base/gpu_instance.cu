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

#include <ggnn/base/gpu_instance.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/graph.h>
#include <ggnn/base/graph_config.h>
#include <ggnn/base/lib.h>
#include <ggnn/base/dataset.cuh>

#include <ggnn/cuda_utils/check.cuh>

#include <ggnn/construction/graph_buffer.cuh>
#include <ggnn/construction/graph_construction.cuh>
#include <ggnn/query/query_kernels.cuh>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cub/cub.cuh>

namespace ggnn {

void CUDAStreamDeleter::operator()(cudaStream_t stream)
{
  if (stream)
    CHECK_CUDA(cudaStreamDestroy(stream));
}
void CUDAEventDeleter::operator()(cudaEvent_t event)
{
  if (event)
    CHECK_CUDA(cudaEventDestroy(event));
}

int GPUContext::getCurrentGPUId()
{
  int device;
  cudaGetDevice(&device);
  return device;
}

void GPUContext::activate() const
{
  CHECK_CUDA(cudaSetDevice(gpu_id));
}

CudaStream GPUContext::createStream()
{
  activate();
  cudaStream_t new_stream;
  CHECK_CUDA(cudaStreamCreate(&new_stream));
  return CudaStream{new_stream};
}
CudaEvent GPUContext::createEvent()
{
  activate();
  cudaEvent_t new_event;
  CHECK_CUDA(cudaEventCreate(&new_event));
  return CudaEvent{new_event};
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::CPUGraphBuffer::load(
    const std::filesystem::path& part_filename, const uint32_t global_shard_id)
{
  std::ifstream inFile{part_filename, std::ifstream::in | std::ifstream::binary};
  CHECK(inFile.is_open()) << "Unable to open " << part_filename;

  inFile.seekg(0, std::ifstream::end);
  size_t filesize = inFile.tellg();
  inFile.seekg(0, std::ifstream::beg);
  CHECK_EQ(filesize, graph.memory.size_bytes())
      << "Error on loading" << part_filename
      << ". File size of GGNNGraph does not match the expected size.";

  inFile.read(reinterpret_cast<char*>(graph.memory.data()), graph.memory.size_bytes());
  inFile.close();

  this->global_shard_id = global_shard_id;
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::CPUGraphBuffer::store(
    const std::filesystem::path& part_filename) const
{
  std::ofstream outFile{part_filename,
                        std::ofstream::out | std::ofstream::binary | std::ofstream::trunc};
  CHECK(outFile.is_open()) << "Unable to open " << part_filename;
  outFile.write(reinterpret_cast<const char*>(graph.memory.data()), graph.memory.size_bytes());
  outFile.close();
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::CPUGraphBuffer::upload(GPUGraphBuffer& gpu_buffer) const
{
  gpu_buffer.global_shard_id = global_shard_id;
  const cudaStream_t stream = gpu_buffer.stream.get();
  graph.memory.copyTo(gpu_buffer.graph.memory, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::CPUGraphBuffer::download(const GPUGraphBuffer& gpu_buffer)
{
  global_shard_id = gpu_buffer.global_shard_id;
  const cudaStream_t stream = gpu_buffer.stream.get();
  gpu_buffer.graph.memory.copyTo(graph.memory, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::allocateGraph(const GraphConfig& config,
                                                     const size_t reserved_gpu_memory)
{
  gpu_ctx.activate();

  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_ctx.gpu_id);
    LOG(INFO) << "[GPU: " << shard_config.device_index
              << "] GPUInstance(): CUDA device id: " << gpu_ctx.gpu_id << " " << prop.name;
  }

  // deallocate old shards
  h_buffers.clear();
  d_buffers.clear();
  d_base_buffers.clear();

  graph_config = config;

  using GraphPartSizes = typename ggnn::Graph<KeyT, ValueT>::PartSizes;
  const size_t graph_size = GraphPartSizes{graph_config}.getGraphSize();

  const uint32_t max_gpu_buffers = [this, graph_size, reserved_gpu_memory]() -> uint32_t {
    size_t free, total;
    CHECK_CUDA(cudaMemGetInfo(&free, &total));

    if (reserved_gpu_memory) {
      CHECK_GT(free, reserved_gpu_memory)
          << "GPU memory does not suffice for the reserved amount.\n"
          << "(Graph construction requires a minimum amount of reserved memory.)";
      VLOG(4) << "Reserving " << sizeInGB(reserved_gpu_memory) << " GB of device memory.";
      free -= reserved_gpu_memory;
    }

    const size_t size_per_shard = graph_config.getBaseSize(sizeof(BaseT)) + graph_size;

    const uint32_t max_shards = static_cast<uint32_t>(free / size_per_shard);
    VLOG(4) << "Remaining device memory (" << sizeInGB(free) << " GB) suffices for " << max_shards
            << " shards (" << sizeInGB(size_per_shard) << " GB each).";

    CHECK_GT(max_shards, 0)
        << "GPU memory does not suffice for a single shard. use smaller shards.";

    return max_shards;
  }();

  const uint32_t num_gpu_buffers = std::min(max_gpu_buffers, shard_config.num_shards);

  d_buffers.reserve(num_gpu_buffers);
  d_base_buffers.reserve(num_gpu_buffers);

  // allocate CPU memory first (fail early if out of memory)
  const uint32_t max_cpu_buffers = [this, graph_size]() -> uint32_t {
    if (shard_config.cpu_memory_limit == std::numeric_limits<uint32_t>::max())
      return std::numeric_limits<uint32_t>::max();

    const size_t size_per_shard = graph_size;

    const uint32_t max_shards =
        static_cast<uint32_t>(shard_config.cpu_memory_limit / size_per_shard);
    VLOG(4) << "assigned CPU memory (" << sizeInGB(shard_config.cpu_memory_limit)
            << " GB) suffices for " << max_shards << " shards (" << sizeInGB(size_per_shard)
            << " GB each).";

    CHECK_GT(max_shards, 0)
        << "CPU memory does not suffice for a single shard. use smaller shards.";

    return max_shards;
  }();

  // NOTE: even a single buffer is not strictly necessary when running everything purely on the GPU
  const bool all_shards_on_gpu = num_gpu_buffers == shard_config.num_shards;
  const uint32_t num_cpu_buffers =
      all_shards_on_gpu ? 1 : std::min(max_cpu_buffers, shard_config.num_shards);
  VLOG_IF(4, all_shards_on_gpu)
      << "GPU memory suffices for all shards. Allocating only a single CPU buffer.";

  allocateCPUBuffers(num_cpu_buffers);

  for (uint32_t i = 0; i < num_gpu_buffers; i++) {
    d_buffers.push_back(GPUGraphBuffer{
        .graph = Graph{graph_config, Dataset<std::byte>::emptyOnGPU(graph_size, 1, gpu_ctx.gpu_id)},
        .global_shard_id = -1U,
        .stream = gpu_ctx.createStream()});
    d_base_buffers.push_back(GPUBaseBuffer{
        .base = Dataset<BaseT>::emptyOnGPU(graph_config.N, graph_config.D, gpu_ctx.gpu_id),
        .global_shard_id = -1U,
    });
  }

  io_threads.resize(std::min(num_cpu_buffers, num_gpu_buffers));
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::allocateCPUBuffers(const uint32_t num_cpu_buffers)
{
  CHECK(h_buffers.empty());

  using GraphPartSizes = typename ggnn::Graph<KeyT, ValueT>::PartSizes;
  const size_t graph_size = GraphPartSizes{graph_config}.getGraphSize();

  h_buffers.reserve(num_cpu_buffers);
  for (uint32_t i = 0; i < num_cpu_buffers; i++) {
    h_buffers.push_back(CPUGraphBuffer{
        .graph = Graph{graph_config, Dataset<std::byte>::empty(graph_size, 1)},
        .global_shard_id = -1U,
    });
  }
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::prefetchBase()
{
  CHECK_NOTNULL(h_base_ref.data());

  for (uint32_t i = 0; i < d_base_buffers.size(); i++) {
    const uint32_t global_shard_id = shard_config.num_shards * shard_config.device_index + i;
    loadBasePart(global_shard_id);
  }
}

template <typename KeyT, typename ValueT, typename BaseT>
const GPUInstance<KeyT, ValueT, BaseT>::CPUGraphBuffer&
GPUInstance<KeyT, ValueT, BaseT>::getCPUGraphShard(const std::filesystem::path& graph_dir,
                                                   const uint32_t global_shard_id)
{
  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  CHECK_GE(global_shard_id, num_previous_shards);
  const uint32_t on_gpu_shard_id = global_shard_id - num_previous_shards;
  CHECK_LT(on_gpu_shard_id, shard_config.num_shards);

  const CPUGraphBuffer& cpu_buffer = getCPUGraphBuffer(on_gpu_shard_id);
  if (cpu_buffer.global_shard_id != global_shard_id) {
    if (d_buffers.size() < shard_config.num_shards) {
      // need to load from file
      swapInPart(graph_dir, global_shard_id);
    }
    else {
      // need to download from GPU
      CHECK_EQ(d_buffers.size(), shard_config.num_shards);
      swapOutPart(graph_dir, global_shard_id, true);
    }
    waitForPart(global_shard_id);
  }
  CHECK_EQ(cpu_buffer.global_shard_id, global_shard_id);
  return cpu_buffer;
}

template <typename KeyT, typename ValueT, typename BaseT>
const GPUInstance<KeyT, ValueT, BaseT>::GPUGraphBuffer&
GPUInstance<KeyT, ValueT, BaseT>::getGPUGraphShard(const std::filesystem::path& graph_dir,
                                                   const uint32_t global_shard_id,
                                                   const bool sync_stream)
{
  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  CHECK_GE(global_shard_id, num_previous_shards);
  const uint32_t on_gpu_shard_id = global_shard_id - num_previous_shards;
  CHECK_LT(on_gpu_shard_id, shard_config.num_shards);

  const GPUGraphBuffer& gpu_buffer = getGPUGraphBuffer(on_gpu_shard_id);
  if (gpu_buffer.global_shard_id != global_shard_id) {
    swapInPart(graph_dir, global_shard_id);
    if (sync_stream)
      waitForPart(global_shard_id);
  }
  CHECK_EQ(gpu_buffer.global_shard_id, global_shard_id);
  return gpu_buffer;
}

template <typename KeyT, typename ValueT, typename BaseT>
const GPUInstance<KeyT, ValueT, BaseT>::GPUBaseBuffer&
GPUInstance<KeyT, ValueT, BaseT>::getGPUBaseShard(const uint32_t global_shard_id,
                                                  const bool sync_stream)
{
  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  CHECK_GE(global_shard_id, num_previous_shards);
  const uint32_t on_gpu_shard_id = global_shard_id - num_previous_shards;
  CHECK_LT(on_gpu_shard_id, shard_config.num_shards);
  CHECK(hasPart(global_shard_id)) << "part " << global_shard_id
                                  << " does not belong to GPU instance "
                                  << shard_config.device_index;

  const GPUBaseBuffer& gpu_base_buffer = getGPUBaseBuffer(on_gpu_shard_id);
  if (gpu_base_buffer.global_shard_id != global_shard_id) {
    loadBasePart(global_shard_id);
    if (sync_stream)
      cudaStreamSynchronize(getStreamForPart(global_shard_id));
  }
  CHECK_EQ(gpu_base_buffer.global_shard_id, global_shard_id);
  return gpu_base_buffer;
}

template <typename KeyT, typename ValueT, typename BaseT>
bool GPUInstance<KeyT, ValueT, BaseT>::hasPart(const uint32_t global_shard_id) const
{
  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  return global_shard_id >= num_previous_shards &&
         global_shard_id < num_previous_shards + shard_config.num_shards;
}

template <typename KeyT, typename ValueT, typename BaseT>
cudaStream_t GPUInstance<KeyT, ValueT, BaseT>::getStreamForPart(
    const uint32_t global_shard_id) const
{
  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  CHECK_GE(global_shard_id, num_previous_shards);
  const uint32_t on_gpu_shard_id = global_shard_id - num_previous_shards;
  CHECK_LT(on_gpu_shard_id, shard_config.num_shards);

  const GPUGraphBuffer& gpu_buffer = getGPUGraphBuffer(on_gpu_shard_id);
  return gpu_buffer.stream.get();
}

// io

template <typename KeyT, typename ValueT, typename BaseT>
std::thread& GPUInstance<KeyT, ValueT, BaseT>::getThreadForPart(const uint32_t global_shard_id)
{
  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  CHECK_GE(global_shard_id, num_previous_shards);
  const uint32_t on_gpu_shard_id = global_shard_id - num_previous_shards;
  CHECK_LT(on_gpu_shard_id, shard_config.num_shards);

  return io_threads.at(on_gpu_shard_id % io_threads.size());
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::waitForPart(const uint32_t global_shard_id)
{
  std::thread& io_thread = getThreadForPart(global_shard_id);
  if (io_thread.joinable())
    io_thread.join();
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::swapOutPart(const std::filesystem::path& graph_dir,
                                                   const uint32_t global_shard_id,
                                                   bool force_to_ram, bool force_to_file)
{
  const uint32_t num_gpu_buffers = static_cast<uint32_t>(d_buffers.size());
  const uint32_t num_cpu_buffers = static_cast<uint32_t>(h_buffers.size());
  const bool swap_to_ram = num_gpu_buffers < shard_config.num_shards;
  const bool swap_to_disk = swap_to_ram && num_cpu_buffers < shard_config.num_shards;

  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  CHECK_GE(global_shard_id, num_previous_shards);
  const uint32_t on_gpu_shard_id = global_shard_id - num_previous_shards;
  CHECK_LT(on_gpu_shard_id, shard_config.num_shards);

  std::thread& io_thread = getThreadForPart(global_shard_id);
  if (io_thread.joinable())
    io_thread.join();

  io_thread = std::thread([=, this]() -> void {
    CPUGraphBuffer& cpu_buffer = getCPUGraphBuffer(on_gpu_shard_id);

    if (cpu_buffer.global_shard_id == global_shard_id) {
      VLOG(4) << "[GPU: " << shard_config.device_index << "] part " << global_shard_id
              << " is already downloaded";
    }
    else if (swap_to_ram || force_to_ram || force_to_file) {
      GPUGraphBuffer& gpu_buffer = getGPUGraphBuffer(on_gpu_shard_id);
      gpu_ctx.activate();

      CHECK_EQ(gpu_buffer.global_shard_id, global_shard_id);
      cpu_buffer.download(gpu_buffer);
      VLOG(3) << "[GPU: " << shard_config.device_index << "] downloaded part " << global_shard_id;
    }
    else {
      // TODO: is this a good idea? (in this case, there is enough space on the GPU, so we don't
      // need to copy back to CPU)
      VLOG(4) << "[GPU: " << shard_config.device_index << "] skipped downloading part "
              << global_shard_id;
    }

    if (swap_to_disk || force_to_file) {
      CHECK_EQ(cpu_buffer.global_shard_id, global_shard_id);
      const std::filesystem::path part_filename =
          graph_dir / std::string{"part_" + std::to_string(global_shard_id) + ".ggnn"};
      cpu_buffer.store(part_filename);
      VLOG(2) << "[GPU: " << shard_config.device_index << "] stored part " << global_shard_id
              << " to " << part_filename.c_str();
    }
  });
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::swapInPart(const std::filesystem::path& graph_dir,
                                                  const uint32_t global_shard_id,
                                                  bool force_load_from_file)
{
  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  CHECK_GE(global_shard_id, num_previous_shards);
  const uint32_t on_gpu_shard_id = global_shard_id - num_previous_shards;
  CHECK_LT(on_gpu_shard_id, shard_config.num_shards);

  GPUGraphBuffer& gpu_buffer = getGPUGraphBuffer(on_gpu_shard_id);
  if (!force_load_from_file && gpu_buffer.global_shard_id == global_shard_id) {
    VLOG(4) << "[GPU: " << shard_config.device_index << "] part " << global_shard_id
            << " is already loaded on GPU buffer " << on_gpu_shard_id % d_buffers.size();
    return;
  }

  std::thread& io_thread = getThreadForPart(global_shard_id);
  if (io_thread.joinable())
    io_thread.join();

  io_thread = std::thread([=, this]() -> void {
    gpu_ctx.activate();

    CPUGraphBuffer& cpu_buffer = getCPUGraphBuffer(on_gpu_shard_id);
    GPUGraphBuffer& gpu_buffer = getGPUGraphBuffer(on_gpu_shard_id);

    if (!force_load_from_file && cpu_buffer.global_shard_id == global_shard_id) {
      VLOG(4) << "[GPU: " << shard_config.device_index << "] part " << global_shard_id
              << " is already loaded on CPU buffer " << on_gpu_shard_id % h_buffers.size();
    }
    else {
      const std::filesystem::path part_filename =
          graph_dir / std::string{"part_" + std::to_string(global_shard_id) + ".ggnn"};
      VLOG(4) << "[GPU: " << shard_config.device_index << "] loading part " << global_shard_id
              << " from " << part_filename.c_str();
      cpu_buffer.load(part_filename, global_shard_id);
      VLOG(3) << "[GPU: " << shard_config.device_index << "] loaded part " << global_shard_id
              << " from " << part_filename.c_str();
    }
    CHECK_EQ(cpu_buffer.global_shard_id, global_shard_id);
    cpu_buffer.upload(gpu_buffer);
    CHECK_EQ(gpu_buffer.global_shard_id, global_shard_id);
    VLOG(3) << "[GPU: " << shard_config.device_index << "] uploaded part " << global_shard_id;
  });
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::loadBasePart(const uint32_t global_shard_id)
{
  const uint32_t num_previous_shards = shard_config.num_shards * shard_config.device_index;
  CHECK_GE(global_shard_id, num_previous_shards);
  const uint32_t on_gpu_shard_id = global_shard_id - num_previous_shards;
  CHECK_LT(on_gpu_shard_id, shard_config.num_shards);

  CHECK_NOTNULL(h_base_ref.data());

  GPUBaseBuffer& gpu_base_buffer = getGPUBaseBuffer(on_gpu_shard_id);
  if (gpu_base_buffer.global_shard_id == global_shard_id) {
    VLOG(4) << "[GPU: " << shard_config.device_index << "] base part " << global_shard_id
            << " is already loaded on shard " << on_gpu_shard_id;
    return;
  }
  const size_t N_shard = graph_config.N;

  if (h_base_ref.isGPUAccessible() && h_base_ref.gpu_id == gpu_ctx.gpu_id) {
    gpu_base_buffer.base = h_base_ref.referenceRange(N_shard * global_shard_id, N_shard);
  }
  else {
    h_base_ref.copyRangeTo(N_shard * global_shard_id, N_shard, gpu_base_buffer.base,
                           getStreamForPart(global_shard_id));
  }
  gpu_base_buffer.global_shard_id = global_shard_id;
  VLOG(3) << "[GPU: " << shard_config.device_index << "] base part " << global_shard_id
          << " loaded on shard " << on_gpu_shard_id;
}

template <typename KeyT, typename ValueT, typename BaseT>
float GPUInstance<KeyT, ValueT, BaseT>::build(const Dataset<BaseT>& base,
                                              const std::filesystem::path& graph_dir,
                                              const GraphConfig& config, const float tau_build,
                                              const uint32_t refinement_iterations,
                                              const DistanceMeasure measure,
                                              const size_t reserved_gpu_memory)
{
  using GraphBufferPartSizes = typename ggnn::GraphBuffer<KeyT, ValueT>::PartSizes;

  h_base_ref = base.reference();
  const size_t construction_size = GraphBufferPartSizes{graph_config}.getBufferSize();
  allocateGraph(config, std::max(construction_size, reserved_gpu_memory));
  prefetchBase();

  using GraphConstruction = ggnn::GraphConstruction<KeyT, ValueT, BaseT>;

  GraphConstruction construction{*this, tau_build, measure};
  float total_build_time = 0.0f;

  gpu_ctx.activate();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  VLOG(1) << "[GPU: " << shard_config.device_index << "] build(): N: " << graph_config.N;

  for (uint32_t i = 0; i < shard_config.num_shards; i++) {
    const uint32_t global_shard_id = shard_config.device_index * shard_config.num_shards + i;
    GPUGraphBuffer& gpu_buffer = getGPUGraphBuffer(i);
    const GPUBaseBuffer& gpu_base_buffer = getGPUBaseBuffer(i);
    const cudaStream_t stream = gpu_buffer.stream.get();

    uint32_t refinement_step = 0;
    float milliseconds = 0;

    auto sync_and_report_progress = [&]() {
      cudaEventRecord(stop, stream);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&milliseconds, start, stop);
      VLOG(0) << "[GPU: " << shard_config.device_index << "] build(): part: " << global_shard_id
              << " refinement step: " << refinement_step << " => seconds: " << milliseconds / 1000.f
              << " [" << graph_config.N << " points build -> "
              << milliseconds * 1000.0f / static_cast<float>(graph_config.N) << " us/point] \n";
    };

    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    construction.build(gpu_buffer.graph, gpu_base_buffer.base, stream);

    for (; refinement_step < refinement_iterations; ++refinement_step) {
      sync_and_report_progress();
      construction.refine(gpu_buffer.graph, gpu_base_buffer.base, stream);
    }

    sync_and_report_progress();
    total_build_time += milliseconds;

    getGPUGraphBuffer(i).global_shard_id = global_shard_id;

    swapOutPart(graph_dir, global_shard_id);

    // prefetch base for following in-memory shards
    if (i + d_base_buffers.size() < shard_config.num_shards)
      loadBasePart(global_shard_id + d_base_buffers.size());
  }

  // wait for all parts to be swapped out
  for (uint32_t i = 0; i < shard_config.num_shards; i++) {
    const uint32_t global_shard_id = shard_config.device_index * shard_config.num_shards + i;
    waitForPart(global_shard_id);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  VLOG(0) << "[GPU: " << shard_config.device_index << "] build() done.";

  // process the shards in reverse order during the next query for improved cache utilization
  process_shards_back_to_front = true;

  return total_build_time;
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::load(const Dataset<BaseT>& base,
                                            const std::filesystem::path& graph_dir,
                                            const GraphConfig& config,
                                            const size_t reserved_gpu_memory)
{
  h_base_ref = base.reference();
  allocateGraph(config, reserved_gpu_memory);
  prefetchBase();

  for (uint32_t i = 0; i < shard_config.num_shards; i++) {
    const uint32_t global_shard_id = shard_config.device_index * shard_config.num_shards + i;
    swapInPart(graph_dir, global_shard_id, true);
  }
  for (uint32_t i = 0; i < shard_config.num_shards; i++) {
    const uint32_t global_shard_id = shard_config.device_index * shard_config.num_shards + i;
    waitForPart(global_shard_id);
  }

  // process the shards in reverse order during the next query for improved cache utilization
  process_shards_back_to_front = true;

  VLOG(0) << "[GPU: " << shard_config.device_index << "] load() done.";
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::store(const std::filesystem::path& graph_dir)
{
  for (uint32_t i = 0; i < shard_config.num_shards; i++) {
    const uint32_t global_shard_id = shard_config.device_index * shard_config.num_shards + i;
    swapOutPart(graph_dir, global_shard_id, true, true);
  }
  for (uint32_t i = 0; i < shard_config.num_shards; i++) {
    const uint32_t global_shard_id = shard_config.device_index * shard_config.num_shards + i;
    waitForPart(global_shard_id);
  }

  VLOG(0) << "[GPU: " << shard_config.device_index << "] store() done.";
}

template <typename KeyT, typename ValueT, typename BaseT>
typename GPUInstance<KeyT, ValueT, BaseT>::Results GPUInstance<KeyT, ValueT, BaseT>::query(
    const Dataset<BaseT>& query, const std::filesystem::path& graph_dir, const uint32_t KQuery,
    const uint32_t max_iterations, const float tau_query, const DistanceMeasure measure)
{
  if (d_buffers.empty() || getGPUGraphBuffer(0).global_shard_id == -1U) {
    LOG(ERROR) << "no graph available for query. did you forget to build one?";
    return {};
  }

  const uint32_t num_previous_shards = shard_config.device_index * shard_config.num_shards;

  Dataset<BaseT> d_query = query.referenceOnGPU(
      gpu_ctx.gpu_id,
      getStreamForPart(num_previous_shards +
                       (process_shards_back_to_front ? shard_config.num_shards - 1 : 0)));

  Results d_results = {
      Dataset<KeyT>::emptyOnGPU(d_query.N, KQuery * shard_config.num_shards, gpu_ctx.gpu_id),
      Dataset<ValueT>::emptyOnGPU(d_query.N, KQuery * shard_config.num_shards, gpu_ctx.gpu_id)};

  QueryKernels<KeyT, ValueT, BaseT> query_kernels{measure};
  const uint32_t N_query = d_query.N;
  const size_t num_gpu_buffers = d_buffers.size();
  const size_t num_cpu_buffers = h_buffers.size();
  const size_t prefetch_amount = std::min(num_cpu_buffers, num_gpu_buffers);

  gpu_ctx.activate();

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  float milliseconds = 0;

  // prefetch as many shards onto the GPU as possible
  for (uint32_t i = 0; i < num_gpu_buffers; i++) {
    const uint32_t j = process_shards_back_to_front ? shard_config.num_shards - i - 1 : i;
    const uint32_t global_shard_id = num_previous_shards + j;
    loadBasePart(global_shard_id);
    swapInPart(graph_dir, global_shard_id);
  }

  // query all shards
  for (uint32_t i = 0; i < shard_config.num_shards; i++) {
    const uint32_t j = process_shards_back_to_front ? shard_config.num_shards - i - 1 : i;
    const uint32_t global_shard_id = num_previous_shards + j;

    const cudaStream_t stream = getStreamForPart(global_shard_id);

    {
      const auto begin = std::chrono::high_resolution_clock::now();
      waitForPart(global_shard_id);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto cpu_us =
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
      VLOG(2) << "[GPU: " << shard_config.device_index
              << "] shard-swap delay: " << static_cast<float>(cpu_us) * 0.001f << " ms.";
    }

    // CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventRecord(start, stream));
    query_kernels.query(*this, global_shard_id, d_query, KQuery, max_iterations, tau_query,
                        d_results);
    CHECK_CUDA(cudaEventRecord(stop, stream));

    // start the upload for the next shard after starting the current query
    // then, it should be able to overlap
    // prefetch only as much in parallel as there are cpu buffers
    if (process_shards_back_to_front) {
      if (j >= prefetch_amount && j - prefetch_amount < shard_config.num_shards - num_gpu_buffers) {
        loadBasePart(global_shard_id - prefetch_amount);
        swapInPart(graph_dir, global_shard_id - prefetch_amount);
      }
    }
    else if (j + prefetch_amount < shard_config.num_shards &&
             j + prefetch_amount >= num_gpu_buffers) {
      loadBasePart(global_shard_id + prefetch_amount);
      swapInPart(graph_dir, global_shard_id + prefetch_amount);
    }

    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    VLOG(0) << "[GPU: " << shard_config.device_index << "] query part: " << global_shard_id
            << " => ms: " << milliseconds << " [" << N_query << " points query -> "
            << milliseconds * 1000.0f / static_cast<float>(N_query) << " us/point] \n";
  }

  // sort results from multiple parts
  if (shard_config.num_shards > 1) {
    cudaStream_t lastShardStream = getStreamForPart(
        num_previous_shards + (process_shards_back_to_front ? 0 : shard_config.num_shards - 1));

    CHECK_CUDA(cudaEventRecord(start, lastShardStream));

    sortQueryResults(d_results, lastShardStream);

    CHECK_CUDA(cudaEventRecord(stop, lastShardStream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    VLOG(0) << "[GPU: " << shard_config.device_index
            << "] query sort: " << " => ms: " << milliseconds << " [" << N_query
            << " points query -> " << milliseconds * 1000.0f / static_cast<float>(N_query)
            << " us/point] \n";

    VLOG(0) << "[GPU: " << shard_config.device_index << "] query() done.";
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  // process the shards in reverse order during the next query for improved cache utilization
  process_shards_back_to_front = !process_shards_back_to_front;

  return d_results;
}

template <typename KeyT, typename ValueT, typename BaseT>
void GPUInstance<KeyT, ValueT, BaseT>::sortQueryResults(Results& d_results, cudaStream_t stream)
{
  if (shard_config.num_shards <= 1)
    return;

  CHECK_NOTNULL(d_results.ids.data());

  Results d_results_sorted = {
      Dataset<KeyT>::emptyOnGPU(d_results.ids.N, d_results.ids.D, gpu_ctx.gpu_id),
      Dataset<ValueT>::emptyOnGPU(d_results.dists.N, d_results.dists.D, gpu_ctx.gpu_id)};

  Dataset<uint32_t> d_offsets =
      Dataset<uint32_t>::emptyOnGPU(d_results.ids.N + 1, 1, gpu_ctx.gpu_id);

  // The results are stored sequentially for all parts per query.
  // CUB needs to know where these sequences begin and end.
  // The previous end always serves as the next beginning.
  Dataset<uint32_t> h_offsets{Dataset<uint32_t>::empty(d_results.ids.N + 1, 1, true)};
  for (uint32_t i = 0; i < (d_results.ids.N + 1); i++) {
    h_offsets[i] = i * d_results.ids.D;
  }
  h_offsets.copyTo(d_offsets, stream);

  size_t temp_storage_bytes = 0;

  cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr, temp_storage_bytes, d_results.dists.data(), d_results_sorted.dists.data(),
      d_results.ids.data(), d_results_sorted.ids.data(), static_cast<int>(d_results.ids.numel()),
      static_cast<int>(d_results.ids.N), d_offsets.data(), d_offsets.data() + 1, 0,
      sizeof(ValueT) * 8, stream);

  Dataset<std::byte> d_temp_storage =
      Dataset<std::byte>::emptyOnGPU(temp_storage_bytes, 1, gpu_ctx.gpu_id);

  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage.data(), temp_storage_bytes, d_results.dists.data(),
      d_results_sorted.dists.data(), d_results.ids.data(), d_results_sorted.ids.data(),
      static_cast<int>(d_results.ids.numel()), static_cast<int>(d_results.ids.N), d_offsets.data(),
      d_offsets.data() + 1, 0, sizeof(ValueT) * 8, stream);

  // wait for CUB to finish using d_temp_storage before deleting
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::swap(d_results, d_results_sorted);
}

#define GGNN_GPU_INSTANCE(KeyT, ValueT, BaseT)                  \
  extern template struct Dataset<BaseT>;                        \
  extern template struct Graph<KeyT, ValueT>;                   \
  extern template struct Results<KeyT, ValueT>;                 \
  extern template class GraphConstruction<KeyT, ValueT, BaseT>; \
  extern template class QueryKernels<KeyT, ValueT, BaseT>;      \
                                                                \
  template class GPUInstance<KeyT, ValueT, BaseT>;

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_GPU_INSTANCE);

};  // namespace ggnn
