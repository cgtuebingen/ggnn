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

#ifndef INCLUDE_GGNN_GPU_INSTANCE_CUH
#define INCLUDE_GGNN_GPU_INSTANCE_CUH

#include <ggnn/base/def.h>
#include <ggnn/base/graph.h>
#include <ggnn/base/graph_config.h>
#include <ggnn/base/dataset.cuh>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

namespace ggnn {

struct ShardingConfiguration {
  /// Number of base data points per shard.
  uint32_t N_shard{0};
  /// Sequential index for sharding: GPU i/N - independent of the CUDA device index.
  uint32_t device_index{0};
  /// Number of shards to process on this GPU.
  uint32_t num_shards{1};
  /// Memory limit for swapping shards to CPU.
  size_t cpu_memory_limit{std::numeric_limits<size_t>::max()};
};

struct CUDAStreamDeleter {
  void operator()(cudaStream_t stream);
};
struct CUDAEventDeleter {
  void operator()(cudaEvent_t event);
};

using CudaStream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CUDAStreamDeleter>;
using CudaEvent = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, CUDAEventDeleter>;

struct GPUContext {
  const int gpu_id{getCurrentGPUId()};
  static int getCurrentGPUId();

  void activate() const;
  CudaStream createStream();
  CudaEvent createEvent();
};

/**
 * GGNN core operations (shared between single-GPU and multi-GPU version)
 *
 * @param KeyT datatype of dataset indices (needs to be able to represent
 * N_base, signed integer required)
 * @param ValueT distance value type
 * @param BaseT datatype of dataset vector elements
 */
template <typename KeyT, typename ValueT, typename BaseT>
class GPUInstance {
 public:
  using Graph = ggnn::Graph<KeyT, ValueT>;
  using Results = ggnn::Results<KeyT, ValueT>;

  GPUInstance(const GPUContext& gpu_ctx, const ShardingConfiguration& shard_config,
              const GraphConfig& graph_config)
      : gpu_ctx{gpu_ctx}, shard_config{shard_config}, graph_config{graph_config}
  {
  }

  GPUContext gpu_ctx{};
  ShardingConfiguration shard_config{};
  GraphConfig graph_config{};

  float build(const Dataset<BaseT>& base, const std::filesystem::path& graph_dir,
              const GraphConfig& graph_config, const float tau_build,
              const uint32_t refinement_iterations, const DistanceMeasure measure);
  void load(const Dataset<BaseT>& base, const std::filesystem::path& graph_dir,
            const GraphConfig& graph_config);
  void store(const std::filesystem::path& graph_dir);

  [[nodiscard]] Results query(const Dataset<BaseT>& query, const std::filesystem::path& graph_dir,
                              const uint32_t KQuery, const uint32_t max_iterations,
                              const float tau_query, const DistanceMeasure measure);

  struct GPUGraphBuffer {
    Graph graph;
    uint32_t global_shard_id;
    CudaStream stream;
  };

  struct GPUBaseBuffer {
    Dataset<BaseT> base;
    uint32_t global_shard_id;
  };

  struct CPUGraphBuffer {
    Graph graph;
    uint32_t global_shard_id;

    void load(const std::filesystem::path& part_filename, const uint32_t global_shard_id);
    void store(const std::filesystem::path& part_filename) const;
    void upload(GPUGraphBuffer& gpu_buffer) const;
    void download(const GPUGraphBuffer& gpu_buffer);
  };

  [[nodiscard]] const CPUGraphBuffer& getCPUGraphShard(const std::filesystem::path& graph_dir,
                                                       const uint32_t global_shard_id);
  [[nodiscard]] const GPUGraphBuffer& getGPUGraphShard(const std::filesystem::path& graph_dir,
                                                       const uint32_t global_shard_id,
                                                       const bool sync_stream = true);
  [[nodiscard]] const GPUBaseBuffer& getGPUBaseShard(const uint32_t global_shard_id,
                                                     const bool sync_stream = true);
  [[nodiscard]] bool hasPart(const uint32_t global_shard_id) const;

  [[nodiscard]] cudaStream_t getStreamForPart(const uint32_t global_shard_id) const;

  /// Get the GPU buffer responsible for the given \c on_gpu_shard_id (const version).
  const GPUGraphBuffer& getGPUGraphBuffer(const uint32_t on_gpu_shard_id) const
  {
    return d_buffers.at(on_gpu_shard_id % d_buffers.size());
  }
  /// Get the GPU base buffer responsible for the given \c on_gpu_shard_id (const version).
  const GPUBaseBuffer& getGPUBaseBuffer(const uint32_t on_gpu_shard_id) const
  {
    return d_base_buffers.at(on_gpu_shard_id % d_base_buffers.size());
  }

 private:
  /// in-memory GPU shards (some shards might be swapped out to CPU)
  std::vector<GPUGraphBuffer> d_buffers;
  /// in-memory GPU base shards (some shards might be swapped out to CPU)
  std::vector<GPUBaseBuffer> d_base_buffers;
  /// in-memory CPU shards (some shards might be swapped out to disk)
  std::vector<CPUGraphBuffer> h_buffers;
  /// threads for performing i/o tasks (number is min of CPU/GPU shards)
  std::vector<std::thread> io_threads;

  Dataset<BaseT> h_base_ref{};

  bool process_shards_back_to_front{false};

  void allocateGraph(const GraphConfig& graph_config,
                     const bool reserve_construction_memory = false);
  void allocateCPUBuffers(const uint32_t num_cpu_buffers);

  /// Get the GPU buffer responsible for the given \c on_gpu_shard_id.
  GPUGraphBuffer& getGPUGraphBuffer(const uint32_t on_gpu_shard_id)
  {
    return d_buffers.at(on_gpu_shard_id % d_buffers.size());
  }
  /// Get the GPU base buffer responsible for the given \c on_gpu_shard_id.
  GPUBaseBuffer& getGPUBaseBuffer(const uint32_t on_gpu_shard_id)
  {
    return d_base_buffers.at(on_gpu_shard_id % d_base_buffers.size());
  }
  /// Get the CPU buffer responsible for the given \c on_gpu_shard_id.
  CPUGraphBuffer& getCPUGraphBuffer(const uint32_t on_gpu_shard_id)
  {
    return h_buffers.at(on_gpu_shard_id % h_buffers.size());
  }

  [[nodiscard]] std::thread& getThreadForPart(const uint32_t global_shard_id);

  // NOTE: global_shard_id refers to the global index in the number of shards the dataset has been
  // split into. NOTE: on_gpu_shard_id follows the same index but starts at 0 per GPU.

  // io

  /**
   * Swap out a newly constructed graph shard from GPU to CPU.
   * If necessary, or requested by \c force_store, store into a file.
   */
  void swapOutPart(const std::filesystem::path& graph_dir, const uint32_t global_shard_id,
                   bool force_to_ram = false, bool force_to_file = false);
  /**
   * Swap in a previously constructed graph shard from CPU to GPU.
   * If necessary, or requested by \c force_load, load from a file.
   */
  void swapInPart(const std::filesystem::path& graph_dir, const uint32_t global_shard_id,
                  bool force_load_from_file = false);
  /**
   * Wait for swap in / swap out of the given part to complete.
   */
  void waitForPart(const uint32_t global_shard_id);
  /**
   * Swap in the base data for the given \c global_shard_id.
   */
  void loadBasePart(const uint32_t global_shard_id);
  /**
   * Load the first N base parts which fit into the \c d_base_buffers.
   */
  void prefetchBase();

  /**
   * Sort the results from querying multiple parts.
   * Results are expected to be concatenated per query vector.
   */
  void sortQueryResults(Results& d_results, cudaStream_t stream);
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_GPU_INSTANCE_CUH
