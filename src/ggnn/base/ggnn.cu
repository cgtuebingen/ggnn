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

#include <ggnn/base/ggnn.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/eval.h>
#include <ggnn/base/graph.h>
#include <ggnn/base/graph_config.h>
#include <ggnn/base/lib.h>
#include <ggnn/base/result_merger.h>
#include <ggnn/base/data.cuh>

#include <ggnn/base/dataset.cuh>
#include <ggnn/base/gpu_instance.cuh>

#include <ggnn/cuda_utils/check.cuh>

#include <ggnn/query/query_kernels.cuh>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <glog/logging.h>

namespace ggnn {

struct GGNNConfig {
  std::filesystem::path graph_dir{};
  size_t cpu_memory_limit{-1UL};
  size_t reserved_gpu_memory{0UL};
  std::vector<int> gpu_ids{};
  uint32_t N_shard{};
  bool return_results_on_gpu{};
};

template <typename KeyT, typename ValueT>
struct GGNNImplBase : public GGNNConfig, public GGNN<KeyT, ValueT> {
  using GGNN = ggnn::GGNN<KeyT, ValueT>;
  using Graph = ggnn::Graph<KeyT, ValueT>;

  GGNNImplBase(const GGNNConfig& config = {}) : GGNNConfig{config}, GGNN{1} {}

  void setWorkingDirectory(const std::filesystem::path& dir) override
  {
    graph_dir = dir.empty() ? std::filesystem::current_path() : std::filesystem::absolute(dir);
    if (std::filesystem::create_directories(graph_dir))
      VLOG(1) << "Created working directory " << graph_dir << ".";
    else
      VLOG(1) << "Using working directory " << graph_dir;
  }

  void setCPUMemoryLimit(const size_t memory_limit) override
  {
    cpu_memory_limit = memory_limit;
    VLOG(1) << "Set CPU memory limit to " << sizeInGB(cpu_memory_limit) << " GiB.";
  }
  void setReservedGPUMemory(const size_t reserved_memory) override
  {
    reserved_gpu_memory = reserved_memory;
    VLOG(1) << "Set reserved GPU memory to " << sizeInGB(reserved_gpu_memory) << " GiB.";
  }

  void setGPUs(const std::span<const int>& gpu_ids) override
  {
    int num_physical_gpus;
    cudaGetDeviceCount(&num_physical_gpus);

    for (int gpu_id : gpu_ids) {
      if (gpu_id < 0 || gpu_id > num_physical_gpus)
        throw std::out_of_range("Invalid GPU index " + std::to_string(gpu_id) + " given.");
    }

    this->gpu_ids.assign(gpu_ids.begin(), gpu_ids.end());
  }

  void setShardSize(const uint32_t N_shard) override
  {
    this->N_shard = N_shard;
  }

  void setReturnResultsOnGPU(const bool return_results_on_gpu) override
  {
    this->return_results_on_gpu = return_results_on_gpu;
  }

  void build(const uint32_t /*KBuild*/, const float /*tau_build*/,
             const uint32_t /*refinement_iterations*/, const DistanceMeasure /*measure*/) override
  {
    throw std::runtime_error("The base needs to be set before building a graph.");
  }

  const Graph& getGraph(const uint32_t /*global_shard_id*/) override
  {
    throw std::runtime_error("No graph has been built or loaded yet.");
  }
};

template <typename KeyT, typename ValueT, typename BaseT>
struct GGNNImpl : public GGNNImplBase<KeyT, ValueT> {
  using GGNN = ggnn::GGNN<KeyT, ValueT>;
  using GGNNImplBase = ggnn::GGNNImplBase<KeyT, ValueT>;
  using GPUInstance = ggnn::GPUInstance<KeyT, ValueT, BaseT>;
  using Results = ggnn::Results<KeyT, ValueT>;
  using Graph = ggnn::Graph<KeyT, ValueT>;

  using GGNNImplBase::cpu_memory_limit;
  using GGNNImplBase::gpu_ids;
  using GGNNImplBase::graph_dir;
  using GGNNImplBase::N_shard;
  using GGNNImplBase::return_results_on_gpu;

  GGNNImpl(const GGNNConfig& config) : GGNNImplBase{config} {}

  /// base data or reference to it
  Dataset<BaseT> base{};

  /// one instance per GPU
  std::vector<GPUInstance> gpu_instances{};

  void setBaseImpl(Dataset<BaseT>&& base)
  {
    // TODO: clear all base copies on GPU instances instead?
    if (!gpu_instances.empty())
      throw std::runtime_error("The base cannot be changed once the GPU instances are setup.");
    this->base = std::move(base);
  }

  void prepare(uint32_t KBuild)
  {
    // TODO: remove the existing GPU instances instead?
    if (!gpu_instances.empty())
      throw std::runtime_error("A graph has already been built or loaded.");

    GraphParameters graph_params{.N = static_cast<uint32_t>(base.N), .D = base.D, .KBuild = KBuild};

    if (N_shard > 0) {
      CHECK_EQ(base.N % N_shard, 0)
          << "The base dataset needs to be evenly divisible by the shard size.";
      graph_params.N = N_shard;
    }

    CHECK_GT(graph_params.N, 0);
    CHECK_GE(graph_params.D, GGNN::MIN_D);
    CHECK_LE(graph_params.D, GGNN::MAX_D);
    CHECK_GE(graph_params.KBuild, GGNN::MIN_KBUILD);
    CHECK_LE(graph_params.KBuild, GGNN::MAX_KBUILD);

    if (gpu_ids.empty()) {
      gpu_ids.resize(1);
      cudaGetDevice(&gpu_ids.at(0));
      VLOG(3) << "Auto-selecting current GPU: " << gpu_ids.at(0) << ".";
    }

    const size_t num_gpus = gpu_ids.size();
    const uint32_t num_shards_per_gpu = base.N / (static_cast<uint64_t>(graph_params.N) * num_gpus);
    CHECK_EQ(graph_params.N * num_gpus * num_shards_per_gpu, base.N)
        << "base.N needs to be evenly divisible by (N_shard x num_gpus).";

    const size_t cpu_memory_per_gpu = cpu_memory_limit / num_gpus;

    const GraphConfig graph_config{graph_params};

    gpu_instances.reserve(num_gpus);
    for (uint32_t device_i = 0; device_i < num_gpus; ++device_i) {
      const int gpu_id = gpu_ids[device_i];
      CHECK_CUDA(cudaSetDevice(gpu_id));

      gpu_instances.emplace_back(GPUContext{gpu_id},
                                 ShardingConfiguration{.N_shard = graph_config.N,
                                                       .device_index = device_i,
                                                       .num_shards = num_shards_per_gpu,
                                                       .cpu_memory_limit = cpu_memory_per_gpu},
                                 graph_config);
    }

    VLOG(2) << "GGNN multi-GPU setup configured.";
  }

  void build(const uint32_t KBuild, const float tau_build, const uint32_t refinement_iterations,
             const DistanceMeasure measure) override
  {
    if (!base.data())
      throw std::runtime_error("The base needs to be set before building a graph.");

    prepare(KBuild);

    VLOG(0) << "GGNN::build() started.";

    const auto begin = std::chrono::steady_clock::now();

    const size_t num_gpus = gpu_ids.size();
    std::vector<std::thread> build_threads;
    build_threads.reserve(num_gpus);
    float build_time_ms = 0.0f;

    for (auto& gpu_instance : gpu_instances)
      build_threads.emplace_back([&]() -> void {
        const float build_time = gpu_instance.build(
            base, graph_dir, GraphParameters{gpu_instance.shard_config.N_shard, base.D, KBuild},
            tau_build, refinement_iterations, measure, this->reserved_gpu_memory);
        build_time_ms += build_time;
      });
    for (auto& build_thread : build_threads)
      build_thread.join();

    const auto end = std::chrono::steady_clock::now();

    const auto wall_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    VLOG(0) << "GGNN::build() completed.";
    VLOG(0) << "Sum of shard build times: " << build_time_ms * 0.001f << " s";
    VLOG(0) << "Wall time: " << static_cast<float>(wall_time_ms) * 0.001f << " s";
  }

  void store() override
  {
    if (gpu_instances.empty())
      throw std::runtime_error("There is no graph to store.");

    const uint32_t num_gpus = static_cast<uint32_t>(gpu_instances.size());
    std::vector<std::thread> store_threads;
    store_threads.reserve(num_gpus);

    for (auto& gpu_instance : gpu_instances)
      store_threads.emplace_back([&]() -> void { gpu_instance.store(graph_dir); });
    for (auto& thread : store_threads)
      thread.join();
  }

  void load(const uint32_t KBuild) override
  {
    if (!base.data())
      throw std::runtime_error("The base needs to be set before loading a graph.");

    prepare(KBuild);

    const uint32_t num_gpus = static_cast<uint32_t>(gpu_instances.size());
    std::vector<std::thread> load_threads;
    load_threads.reserve(num_gpus);

    for (auto& gpu_instance : gpu_instances)
      load_threads.emplace_back([&]() -> void {
        gpu_instance.load(base, graph_dir,
                          GraphParameters{gpu_instance.shard_config.N_shard, base.D, KBuild},
                          this->reserved_gpu_memory);
      });
    for (auto& thread : load_threads)
      thread.join();
  }

  Results queryImpl(const Dataset<BaseT>& query, const uint32_t KQuery, const float tau_query,
                    const uint32_t max_iterations, const DistanceMeasure measure)
  {
    if (gpu_instances.empty())
      throw std::runtime_error("There is no graph to query.");

    GPUInstance& gpu_instance_0 = gpu_instances.at(0);

    const uint32_t N_query = query.N;
    const uint32_t num_gpus = static_cast<uint32_t>(gpu_instances.size());
    const uint32_t N_shard = gpu_instance_0.shard_config.N_shard;
    const uint32_t num_shards_per_gpu = gpu_instance_0.shard_config.num_shards;

    using ResultMerger = ggnn::ResultMerger<KeyT, ValueT>;
    ResultMerger result_merger{static_cast<uint32_t>(query.N), KQuery, num_gpus,
                               num_shards_per_gpu};

    VLOG(0) << "GGNN::query()" << " | N_query: " << N_query << " | tau_query: " << tau_query
            << " | num_gpus: " << num_gpus << " | N_shard: " << N_shard
            << " | num_iterations: " << num_shards_per_gpu;

    if (return_results_on_gpu) {
      if (gpu_instances.size() > 1)
        throw std::runtime_error(
            "Returning query results on GPU is only possible when using a single GPU.");
      Results d_results =
          gpu_instance_0.query(query, graph_dir, KQuery, max_iterations, tau_query, measure);
      return d_results;
    }

    std::vector<std::thread> query_threads;
    query_threads.reserve(num_gpus);

    auto run_query = [&](uint32_t device_i) -> void {
      GPUInstance& gpu_instance = gpu_instances.at(device_i);
      Results d_results =
          gpu_instance.query(query, graph_dir, KQuery, max_iterations, tau_query, measure);
      auto& h_results = result_merger.partial_results_per_gpu.at(device_i);
      // TODO: use a stream assigned to this GPU?
      // cudaStream_t shard0Stream = gpu_instance_0.getGPUGraphBuffer(0).stream.get();
      d_results.ids.copyTo(h_results.ids, 0);
      d_results.dists.copyTo(h_results.dists, 0);
      cudaStreamSynchronize(0);
    };

    for (uint32_t device_i = 0; device_i < num_gpus; device_i++)
      query_threads.emplace_back(run_query, device_i);
    for (auto& thread : query_threads)
      thread.join();

    // CPU Zone:
    return std::move(result_merger).merge(N_shard);
  }

  Results bfQueryImpl(const Dataset<BaseT>& query, const uint32_t KGT,
                      const DistanceMeasure measure)
  {
    if (!base.data())
      throw std::runtime_error("There is no base dataset loaded which could be queried.");

    if (gpu_instances.size() > 1)
      throw std::runtime_error("The brute-force query only supports a single GPU.");

    // TODO: use a stream assigned to this GPU?
    const cudaStream_t stream = 0;
    const int32_t gpu_id = gpu_instances.empty() ? gpu_ids.empty() ? 0 : gpu_ids.at(0)
                                                 : gpu_instances.at(0).gpu_ctx.gpu_id;
    cudaSetDevice(gpu_id);

    Dataset<BaseT> d_base =
        gpu_instances.empty()
            ? base.referenceOnGPU(gpu_id, stream)
            : Dataset<BaseT>{gpu_instances.at(0).getGPUBaseShard(0).base.reference()};
    Dataset<BaseT> d_query = query.referenceOnGPU(gpu_id, stream);

    Results d_results = {Dataset<KeyT>::emptyOnGPU(query.N, KGT, gpu_id),
                         Dataset<ValueT>::emptyOnGPU(query.N, KGT, gpu_id)};

    QueryKernels<KeyT, ValueT, BaseT> query_kernels{measure};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start, stream);
    query_kernels.bruteForceQuery(d_base, d_query, KGT, d_results, stream);
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    VLOG(0) << "[GPU: " << 0 << "] brute-force query: => ms: " << milliseconds << " [" << query.N
            << " points query -> " << milliseconds * 1000.0f / static_cast<float>(query.N)
            << " us/point] \n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (return_results_on_gpu)
      return d_results;

    using ResultMerger = ggnn::ResultMerger<KeyT, ValueT>;
    ResultMerger result_merger{static_cast<uint32_t>(query.N), KGT};

    auto& h_results = result_merger.partial_results_per_gpu.at(0);
    d_results.ids.copyTo(h_results.ids, stream);
    d_results.dists.copyTo(h_results.dists, stream);

    cudaStreamSynchronize(stream);

    return std::move(result_merger).merge(base.N);
  }

  const Graph& getGraph(const uint32_t global_shard_id) override
  {
    if (gpu_instances.empty())
      throw std::runtime_error("There is no graph.");

    for (auto& gpu_instance : gpu_instances) {
      if (gpu_instance.hasPart(global_shard_id)) {
        if (return_results_on_gpu) {
          const auto& gpu_graph_shard = gpu_instance.getGPUGraphShard(graph_dir, global_shard_id);
          CHECK_EQ(gpu_graph_shard.global_shard_id, global_shard_id);
          return gpu_graph_shard.graph;
        }
        else {
          const auto& cpu_graph_shard = gpu_instance.getCPUGraphShard(graph_dir, global_shard_id);
          CHECK_EQ(cpu_graph_shard.global_shard_id, global_shard_id);
          return cpu_graph_shard.graph;
        }
      }
    }
    throw std::runtime_error("Shard " + std::to_string(global_shard_id) + " does not exist.");
  }
};

template <typename KeyT, typename ValueT>
GGNN<KeyT, ValueT>::GGNN() : pimpl{new GGNNImplBase<KeyT, ValueT>{}}
{
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::setWorkingDirectory(const std::filesystem::path& dir)
{
  pimpl->setWorkingDirectory(dir);
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::setCPUMemoryLimit(const size_t memory_limit)
{
  pimpl->setCPUMemoryLimit(memory_limit);
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::setReservedGPUMemory(const size_t reserved_memory)
{
  pimpl->setReservedGPUMemory(reserved_memory);
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::setGPUs(const std::span<const int>& gpu_ids)
{
  pimpl->setGPUs(gpu_ids);
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::setShardSize(const uint32_t N_shard)
{
  pimpl->setShardSize(N_shard);
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::setReturnResultsOnGPU(const bool gpu_only)
{
  pimpl->setReturnResultsOnGPU(gpu_only);
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::setBase(GenericDataset&& base)
{
  // TODO: check pimpl, set new one based on data type
  GGNNImpl<KeyT, ValueT, uint8_t>* impl_uint8_t =
      dynamic_cast<GGNNImpl<KeyT, ValueT, uint8_t>*>(pimpl.get());
  GGNNImpl<KeyT, ValueT, float>* impl_float =
      dynamic_cast<GGNNImpl<KeyT, ValueT, float>*>(pimpl.get());
  GGNNImplBase<KeyT, ValueT>* impl_base = dynamic_cast<GGNNImplBase<KeyT, ValueT>*>(pimpl.get());
  CHECK_NOTNULL(impl_base);

  switch (base.type) {
    case DataType::FLOAT:
      CHECK_EQ(impl_uint8_t, nullptr)
          << "base has already been set with a different data type (uint8_t)";
      if (!impl_float) {
        impl_float = new GGNNImpl<KeyT, ValueT, float>{*static_cast<GGNNConfig*>(impl_base)};
        pimpl.reset(impl_float);
      }
      impl_float->setBaseImpl(std::move(base));
      return;
    case DataType::UINT8:
      CHECK_EQ(impl_float, nullptr)
          << "base has already been set with a different data type (float)";
      if (!impl_uint8_t) {
        impl_uint8_t = new GGNNImpl<KeyT, ValueT, uint8_t>{*static_cast<GGNNConfig*>(impl_base)};
        pimpl.reset(impl_uint8_t);
      }
      impl_uint8_t->setBaseImpl(std::move(base));
      return;
    default:
      break;
  }

  throw std::runtime_error("unsupported datatype for base");
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::setBaseReference(const GenericDataset& base)
{
  setBase(base.reference());
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::build(const uint32_t KBuild, const float tau_build,
                               const uint32_t refinement_iterations, const DistanceMeasure measure)
{
  pimpl->build(KBuild, tau_build, refinement_iterations, measure);
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::store()
{
  pimpl->store();
}

template <typename KeyT, typename ValueT>
void GGNN<KeyT, ValueT>::load(const uint32_t KBuild)
{
  pimpl->load(KBuild);
}

template <typename KeyT, typename ValueT>
Results<KeyT, ValueT> GGNN<KeyT, ValueT>::query(const GenericDataset& query, const uint32_t KQuery,
                                                const float tau_query,
                                                const uint32_t max_iterations,
                                                const DistanceMeasure measure)
{
  switch (query.type) {
    case DataType::FLOAT: {
      GGNNImpl<KeyT, ValueT, float>* impl =
          dynamic_cast<GGNNImpl<KeyT, ValueT, float>*>(pimpl.get());
      CHECK_NOTNULL(impl);  // query data type does not mach base data type or base not set
      return impl->queryImpl(query.reference(), KQuery, tau_query, max_iterations, measure);
    }
    case DataType::UINT8: {
      GGNNImpl<KeyT, ValueT, uint8_t>* impl =
          dynamic_cast<GGNNImpl<KeyT, ValueT, uint8_t>*>(pimpl.get());
      CHECK_NOTNULL(impl);  // query data type does not mach base data type or base not set
      return impl->queryImpl(query.reference(), KQuery, tau_query, max_iterations, measure);
    }
    default:
      break;
  }
  throw std::runtime_error("unsupported datatype for query");
}

template <typename KeyT, typename ValueT>
Results<KeyT, ValueT> GGNN<KeyT, ValueT>::bfQuery(const GenericDataset& query, const uint32_t KGT,
                                                  const DistanceMeasure measure)
{
  switch (query.type) {
    case DataType::FLOAT: {
      GGNNImpl<KeyT, ValueT, float>* impl =
          dynamic_cast<GGNNImpl<KeyT, ValueT, float>*>(pimpl.get());
      CHECK_NOTNULL(impl);  // query data type does not mach base data type or base not set
      return impl->bfQueryImpl(query.reference(), KGT, measure);
    }
    case DataType::UINT8: {
      GGNNImpl<KeyT, ValueT, uint8_t>* impl =
          dynamic_cast<GGNNImpl<KeyT, ValueT, uint8_t>*>(pimpl.get());
      CHECK_NOTNULL(impl);  // query data type does not mach base data type or base not set
      return impl->bfQueryImpl(query.reference(), KGT, measure);
    }
    default:
      break;
  }
  throw std::runtime_error("unsupported datatype for brute-force query");
}

template <typename KeyT, typename ValueT>
const Graph<KeyT, ValueT>& GGNN<KeyT, ValueT>::getGraph(const uint32_t on_gpu_shard_id)
{
  return pimpl->getGraph(on_gpu_shard_id);
}

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_INSTANTIATE_STRUCT, GGNNImpl);
GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_INSTANTIATE_STRUCT, GGNNImplBase);
GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_INSTANTIATE_CLASS, GGNN);

};  // namespace ggnn
