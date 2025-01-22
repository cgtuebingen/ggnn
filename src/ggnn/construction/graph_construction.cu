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

#include <ggnn/construction/graph_construction.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/graph.h>
#include <ggnn/base/graph_config.h>
#include <ggnn/base/lib.h>
#include <ggnn/base/dataset.cuh>
#include <ggnn/base/gpu_instance.cuh>

#include <ggnn/construction/graph_buffer.cuh>
#include <ggnn/construction/merge_layer.cuh>
#include <ggnn/construction/sym_buffer_merge_layer.cuh>
#include <ggnn/construction/sym_query_layer.cuh>
#include <ggnn/construction/top_merge_layer.cuh>
#include <ggnn/construction/wrs_select_layer.cuh>

#include <ggnn/cuda_utils/check.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cub/cub.cuh>

#include <glog/logging.h>

namespace ggnn {

template <typename T>
void time_launcher(const int log_level, T& kernel, uint32_t N, cudaStream_t stream = 0)
{
  cudaEvent_t start, stop;
  if (VLOG_IS_ON(log_level)) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
  }
  kernel.launch(N, stream);
  if (VLOG_IS_ON(log_level)) {
    cudaEventRecord(stop, stream);
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    VLOG(log_level) << milliseconds << " ms for " << N << " queries -> "
                    << milliseconds * 1000.0f / static_cast<float>(N) << " us/query \n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
}

template <typename ValueT>
__global__ void divide(ValueT* res, ValueT* input, ValueT N)
{
  res[threadIdx.x] = input[threadIdx.x] / N;
}

struct CurandGeneratorDeleter {
  void operator()(curandGenerator_t gen)
  {
    if (gen)
      curandDestroyGenerator(gen);
  }
};

using CurandGenerator =
    std::unique_ptr<std::remove_pointer_t<curandGenerator_t>, CurandGeneratorDeleter>;

CurandGenerator createPRNG()
{
  curandGenerator_t gen_tmp;
  curandCreateGenerator(&gen_tmp, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen_tmp, 1234ULL);
  return CurandGenerator{gen_tmp};
}

template <typename KeyT, typename ValueT, typename BaseT>
class GraphConstructionImpl : public GraphConstruction<KeyT, ValueT, BaseT> {
 public:
  using Graph = ggnn::Graph<KeyT, ValueT>;
  using GraphBuffer = ggnn::GraphBuffer<KeyT, ValueT>;
  using GPUInstance = ggnn::GPUInstance<KeyT, ValueT, BaseT>;

  GraphConstructionImpl(GPUInstance& gpu_instance, const float tau_build,
                        const DistanceMeasure measure)
      : gpu_instance{gpu_instance}, tau_build{tau_build}, measure{measure}
  {
  }

 private:
  GPUInstance& gpu_instance;
  const GraphConfig& graph_config{gpu_instance.graph_config};
  float tau_build{};
  DistanceMeasure measure;

  const size_t buffer_size = typename GraphBuffer::PartSizes{graph_config}.getBufferSize();
  GraphBuffer buffer{graph_config,
                     Dataset<std::byte>::emptyOnGPU(buffer_size, 1, gpu_instance.gpu_ctx.gpu_id)};
  CurandGenerator gen{createPRNG()};

  void build(Graph& graph, const Dataset<BaseT>& base, const cudaStream_t stream) override
  {
    for (uint32_t layer_top = 0; layer_top < GraphConfig::L; layer_top++) {
      for (uint32_t layer_btm = layer_top; layer_btm != -1U; layer_btm--) {
        merge(layer_top, layer_btm, graph, base, stream);

        if (layer_top < (GraphConfig::L - 1) && layer_top == layer_btm)
          select(layer_top, graph, stream);

        sym(layer_btm, graph, base, stream);
      }
    }
  }
  void refine(Graph& graph, const Dataset<BaseT>& base, const cudaStream_t stream) override
  {
    for (uint32_t layer = GraphConfig::L - 2; layer != -1U; layer--) {
      merge(GraphConfig::L - 1, layer, graph, base, stream);
      sym(layer, graph, base, stream);
    }
  }

  struct ConstructionKernelConfig {
    uint32_t block_dim_x;
    uint32_t dist_items_per_thread;
  };

  ConstructionKernelConfig selectKernelConfig(uint32_t D, uint32_t min_block_dim_x)
  {
    const uint32_t dist_items_per_thread = D <= 1024 ? 4U : 8U;
    const uint32_t block_dim_x = std::max(
        min_block_dim_x, ggnn::bit_ceil((D + dist_items_per_thread - 1) / dist_items_per_thread));

    return {block_dim_x, dist_items_per_thread};
  }

  void select(const uint32_t layer, Graph& graph, cudaStream_t stream)
  {
    gpu_instance.gpu_ctx.activate();

    /* Generate n floats on device */
    curandSetStream(gen.get(), stream);
    curandGenerateUniform(gen.get(), buffer.rng, graph_config.Ns[layer]);

    using SelectionKernel = ggnn::WRSSelectionKernel<KeyT, ValueT>;

    SelectionKernel select_kernel{.d_selection = graph.selection[layer + 1].data(),
                                  .d_translation = graph.translation[layer + 1].data(),
                                  .d_translation_layer = graph.translation[layer].data(),
                                  .d_nn1_dist_buffer = buffer.nn1_dist_buffer,
                                  .d_rng = buffer.rng,
                                  .Sglob = graph_config.S,
                                  .S = layer ? graph_config.S : graph_config.S0,
                                  .S_offset = layer ? 0 : graph_config.S0_off,
                                  .G = graph_config.G,
                                  .SG = graph_config.SG,
                                  .SG_offset = graph_config.SG_off,
                                  .layer = layer};

    time_launcher(2, select_kernel, graph_config.Bs[layer], stream);
  }

  void merge(const uint32_t layer_top, const uint32_t layer_btm, Graph& graph,
             const Dataset<BaseT>& base, const cudaStream_t stream)
  {
    if (layer_top == layer_btm)
      top(layer_btm, graph, base, stream);
    else
      mergeLayer(layer_top, layer_btm, graph, base, stream);

    if (!layer_btm)
      computeNN1Stats(graph, stream);
  }

  void top(const uint32_t layer, Graph& graph, const Dataset<BaseT>& base,
           const cudaStream_t stream)
  {
    gpu_instance.gpu_ctx.activate();

    auto run_top_merge = [&]<uint32_t BLOCK_DIM_X, uint32_t DIST_ITEMS_PER_THREAD>() -> void {
      using TopMergeKernel =
          ggnn::TopMergeKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;

      TopMergeKernel top_kernel{.D = base.D,
                                .measure = measure,
                                .KBuild = graph_config.KBuild,
                                .d_base = base.data(),
                                .d_translation = graph.translation[layer].data(),
                                .d_graph = graph.graph[layer].data(),
                                .d_nn1_dist_buffer = buffer.nn1_dist_buffer,
                                .S = layer ? graph_config.S : graph_config.S0,
                                .S_offset = layer ? 0 : graph_config.S0_off,
                                .layer = layer};

      time_launcher(2, top_kernel, graph_config.Ns[layer], stream);
    };

    static constexpr uint32_t MIN_BLOCK_DIM_X = 128;
    auto [block_dim_x, dist_items_per_thread] = selectKernelConfig(base.D, MIN_BLOCK_DIM_X);

    if (block_dim_x == 128 && dist_items_per_thread == 4)
      run_top_merge.template operator()<128, 4>();
    else if (block_dim_x == 256 && dist_items_per_thread == 4)
      run_top_merge.template operator()<256, 4>();
    else if (block_dim_x == 256 && dist_items_per_thread == 8)
      run_top_merge.template operator()<256, 8>();
    else if (block_dim_x == 512 && dist_items_per_thread == 8)
      run_top_merge.template operator()<512, 8>();
    else
      CHECK(false) << "configuration " << block_dim_x << " " << dist_items_per_thread
                   << " not supported for top merge kernel.";
  }

  void mergeLayer(const uint32_t layer_top, const uint32_t layer_btm, Graph& graph,
                  const Dataset<BaseT>& base, const cudaStream_t stream)
  {
    gpu_instance.gpu_ctx.activate();

    auto run_merge = [&]<uint32_t BLOCK_DIM_X, uint32_t DIST_ITEMS_PER_THREAD>() -> void {
      using MergeKernel =
          ggnn::MergeKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;

      MergeKernel merge_kernel{
          .D = base.D,
          .measure = measure,
          .KBuild = graph_config.KBuild,
          .S = graph_config.S,
          .d_base = base.data(),
          .d_selection = graph.selection[1].data(),      // the entire selection starts at layer 1
          .d_translation = graph.translation[1].data(),  // the entire translation starts at layer 1
          .d_graph = graph.graph[0].data(),
          .d_graph_buffer = buffer.graph_buffer,
          .d_nn1_stats = graph.nn1_stats.data(),
          .d_nn1_dist_buffer = buffer.nn1_dist_buffer,
          .layer_top = layer_top,
          .layer_btm = layer_btm,
          .G = graph_config.G,
          .S0 = graph_config.S0,
          .S0_offset = graph_config.S0_off,
          .Ns_offsets = graph_config.Ns_offsets,
          .STs_offsets = graph_config.STs_offsets,
          .tau_build = tau_build};

      time_launcher(2, merge_kernel, graph_config.Ns[layer_btm], stream);
    };

    static constexpr uint32_t MIN_BLOCK_DIM_X = 32;
    auto [block_dim_x, dist_items_per_thread] = selectKernelConfig(base.D, MIN_BLOCK_DIM_X);

    if (block_dim_x == 32 && dist_items_per_thread == 4)
      run_merge.template operator()<32, 4>();
    else if (block_dim_x == 64 && dist_items_per_thread == 4)
      run_merge.template operator()<64, 4>();
    else if (block_dim_x == 128 && dist_items_per_thread == 4)
      run_merge.template operator()<128, 4>();
    else if (block_dim_x == 256 && dist_items_per_thread == 4)
      run_merge.template operator()<256, 4>();
    else if (block_dim_x == 256 && dist_items_per_thread == 8)
      run_merge.template operator()<256, 8>();
    else if (block_dim_x == 512 && dist_items_per_thread == 8)
      run_merge.template operator()<512, 8>();
    else
      CHECK(false) << "configuration " << block_dim_x << " " << dist_items_per_thread
                   << " not supported for merge kernel.";

    const size_t graph_buffer_size =
        static_cast<size_t>(graph_config.Ns[layer_btm]) * graph_config.KBuild * sizeof(KeyT);
    CHECK_CUDA(cudaMemcpyAsync(graph.graph[layer_btm].data(), buffer.graph_buffer,
                               graph_buffer_size, cudaMemcpyDeviceToDevice, stream));
  }

  void sym(const uint32_t layer, Graph& graph, const Dataset<BaseT>& base,
           const cudaStream_t stream)
  {
    gpu_instance.gpu_ctx.activate();

    cudaMemsetAsync(buffer.sym_buffer, -1,
                    static_cast<size_t>(graph_config.Ns[layer]) * graph_config.KF * sizeof(KeyT),
                    stream);

    cudaMemsetAsync(buffer.sym_atomic, 0, graph_config.Ns[layer] * sizeof(uint32_t), stream);
    auto run_sym = [&]<uint32_t BLOCK_DIM_X, uint32_t DIST_ITEMS_PER_THREAD>() -> void {
      using SymQueryKernel =
          ggnn::SymQueryKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;

      SymQueryKernel sym_kernel{
          .D = base.D,
          .measure = measure,
          .KBuild = graph_config.KBuild,
          .d_base = base.data(),
          .d_graph = graph.graph[layer].data(),
          .d_translation = graph.translation[layer].data(),
          .d_nn1_stats = graph.nn1_stats.data(),
          .tau_build = tau_build,
          .d_sym_buffer = buffer.sym_buffer,
          .d_sym_atomic = buffer.sym_atomic,
      };

      time_launcher(2, sym_kernel, graph_config.Ns[layer], stream);
    };

    static constexpr uint32_t MIN_BLOCK_DIM_X = 64;
    auto [block_dim_x, dist_items_per_thread] = selectKernelConfig(base.D, MIN_BLOCK_DIM_X);

    if (block_dim_x == 64 && dist_items_per_thread == 4)
      run_sym.template operator()<64, 4>();
    else if (block_dim_x == 128 && dist_items_per_thread == 4)
      run_sym.template operator()<128, 4>();
    else if (block_dim_x == 256 && dist_items_per_thread == 4)
      run_sym.template operator()<256, 4>();
    else if (block_dim_x == 256 && dist_items_per_thread == 8)
      run_sym.template operator()<256, 8>();
    else if (block_dim_x == 512 && dist_items_per_thread == 8)
      run_sym.template operator()<512, 8>();
    else
      CHECK(false) << "configuration " << block_dim_x << " " << dist_items_per_thread
                   << " not supported for sym kernel.";

    using SymBufferMergeKernel = ggnn::SymBufferMergeKernel<KeyT, ValueT>;

    SymBufferMergeKernel sym_buffer_merge_kernel{.KBuild = graph_config.KBuild,
                                                 .d_sym_buffer = buffer.sym_buffer,
                                                 .d_sym_atomic = buffer.sym_atomic,
                                                 .d_graph = graph.graph[layer].data()};

    time_launcher(3, sym_buffer_merge_kernel, graph_config.Ns[layer], stream);

    if (VLOG_IS_ON(2)) {
      Dataset<uint32_t> h_sym_atomic = Dataset<uint32_t>::empty(graph_config.Ns[layer], 1, true);
      // Dataset<uint32_t> h_statistics = Dataset<uint32_t>::empty(graph_config.Ns[layer], 1, true);

      CHECK_CUDA(cudaMemcpyAsync(h_sym_atomic.data(), buffer.sym_atomic, h_sym_atomic.size_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
      // cudaMemcpyAsync(h_statistics.data(), buffer.statistics, h_statistics.size_bytes(),
      // cudaMemcpyDeviceToHost, stream);

      CHECK_CUDA(cudaStreamSynchronize(stream));

      uint32_t c = 0;
      uint32_t m = 0;
      // int unconnected = 0;
      for (uint32_t i = 0; i < graph_config.Ns[layer]; i++) {
        if (h_sym_atomic[i] > graph_config.KF)
          c++;
        m += (h_sym_atomic[i] > graph_config.KF) ? graph_config.KF : h_sym_atomic[i];
        // unconnected += h_statistics[i];
      }
      VLOG(2) << "Layer " << layer << " [N: " << graph_config.Ns[layer] << "] | overflow: " << c
              << " (" << static_cast<float>(c) / static_cast<float>(graph_config.Ns[layer])
              << ") | added_links: " << m << " ("
              << static_cast<float>(m) / static_cast<float>(graph_config.Ns[layer]) << ")\n";
    }
  }

  void computeNN1Stats(Graph& graph, const cudaStream_t stream)
  {
    gpu_instance.gpu_ctx.activate();

    CHECK_CUDA(cub::DeviceReduce::Sum(buffer.temp_storage_cub, buffer.temp_storage_bytes_cub,
                                      buffer.nn1_dist_buffer, &graph.nn1_stats[0],
                                      static_cast<int>(graph_config.N), stream));
    CHECK_CUDA(cub::DeviceReduce::Max(buffer.temp_storage_cub, buffer.temp_storage_bytes_cub,
                                      buffer.nn1_dist_buffer, &graph.nn1_stats[1],
                                      static_cast<int>(graph_config.N), stream));

    divide<ValueT><<<1, 1, 0, stream>>>(graph.nn1_stats.data(), graph.nn1_stats.data(),
                                        ValueT(graph_config.N));

    if (VLOG_IS_ON(2)) {
      ValueT h_nn1_stats[2];
      CHECK_CUDA(cudaMemcpyAsync(h_nn1_stats, graph.nn1_stats.data(), 2 * sizeof(ValueT),
                                 cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
      VLOG(2) << "mean: " << h_nn1_stats[0] << " | max: " << h_nn1_stats[1] << std::endl;
    }
  }
};

template <typename KeyT, typename ValueT, typename BaseT>
GraphConstruction<KeyT, ValueT, BaseT>::GraphConstruction(GPUInstance& gpu_instance,
                                                          const float tau_build,
                                                          const DistanceMeasure measure)
{
  pimpl.reset(new GraphConstructionImpl<KeyT, ValueT, BaseT>{gpu_instance, tau_build, measure});
}

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_INSTANTIATE_CLASS, GraphConstruction);
GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_INSTANTIATE_CLASS, GraphConstructionImpl);

};  // namespace ggnn
