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

#include <ggnn/query/query_kernels.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/graph_config.h>
#include <ggnn/base/lib.h>
#include <ggnn/base/dataset.cuh>
#include <ggnn/base/gpu_instance.cuh>

#include <ggnn/query/bf_query_layer.cuh>
#include <ggnn/query/query_layer.cuh>

#include <ggnn/cuda_utils/check.cuh>

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ggnn {

template <typename KeyT, typename ValueT, typename BaseT>
class QueryKernelsImpl : public QueryKernels<KeyT, ValueT, BaseT> {
 public:
  using GPUInstance = ggnn::GPUInstance<KeyT, ValueT, BaseT>;
  using Results = ggnn::Results<KeyT, ValueT>;

  QueryKernelsImpl(const DistanceMeasure measure) : measure{measure} {}

 private:
  const DistanceMeasure measure;

  virtual void query(const GPUInstance& gpu_instance, const uint32_t shard_id,
                     const Dataset<BaseT>& query, const uint32_t KQuery, const uint32_t max_iters,
                     const float tau_query, Results& results) override
  {
    // number of cache entries per thread
    static constexpr uint32_t CACHE_ITEMS_PER_THREAD = 16;
    // number of query dimensions cached per thread in the distance computation
    static constexpr uint32_t DIST_ITEMS_PER_THREAD = 4;
    // CUDA warp size
    static constexpr uint32_t WARP_SIZE = 32;
    // CUDA block size limit
    static constexpr uint32_t MAX_BLOCK_DIM_X = 1024;

    static constexpr uint32_t MIN_PRIOQ_SIZE = 16;
    static constexpr uint32_t MIN_VISITED_SIZE = 32;
    static constexpr uint32_t MIN_CACHE_SIZE = 256;
    static constexpr uint32_t MAX_K_QUERY =
        6000;  // making it larger would exceed the 48k shared memory limit
    static constexpr uint32_t MAX_CACHE_SIZE =
        8192;  // the next larger one would exceed the 48k shared memory limit

    static constexpr uint32_t MAX_ITERS =
        std::min(MAX_BLOCK_DIM_X * CACHE_ITEMS_PER_THREAD, MAX_CACHE_SIZE);
    static constexpr uint32_t MAX_DIM = MAX_BLOCK_DIM_X * DIST_ITEMS_PER_THREAD;

    CHECK_LE(KQuery, MAX_K_QUERY);

    const uint32_t required_sorted_size = next_multiple<uint32_t, 32>(KQuery + 1 + MIN_PRIOQ_SIZE);

    const uint32_t cache_size =
        std::max({MIN_CACHE_SIZE, required_sorted_size + MIN_VISITED_SIZE, bit_ceil(max_iters)});
    /** number of threads required for tracking visited points
     * (ignoring cache entries used for best list and sorted size)
     * (iterations beyond that length typically result in cycles)
     * 512 ==> 32 threads
     * 1024 ==> 64 threads
     * 2048 ==> 128 threads
     * 4096 ==> 256 threads
     * 8192 ==> 512 threads
     */
    const uint32_t cache_size_block_dim_x =
        bit_ceil((cache_size + CACHE_ITEMS_PER_THREAD - 1) / CACHE_ITEMS_PER_THREAD);
    /** number of threads required for processing data with a certain dimension:
     * 128D ==> 32 threads
     * 256D ==> 64 threads
     * 512D ==> 128 threads
     * 1024D ==> 256 threads
     * 2048D ==> 512 threads
     * 4096D ==> 1024 threads
     */
    const uint32_t dimension_block_dim_x =
        bit_ceil((query.D + DIST_ITEMS_PER_THREAD - 1) / DIST_ITEMS_PER_THREAD);
    const uint32_t block_dim_x =
        std::max({WARP_SIZE, cache_size_block_dim_x, dimension_block_dim_x});

    CHECK_LE(max_iters, MAX_ITERS);
    CHECK_LE(query.D, MAX_DIM);
    CHECK_LE(cache_size, MAX_CACHE_SIZE);
    CHECK_LE(block_dim_x, MAX_BLOCK_DIM_X);

    const uint32_t sorted_size = std::max(cache_size < 512U ? 64U : 32U, required_sorted_size);

    gpu_instance.gpu_ctx.activate();
    const uint32_t on_gpu_shard_id =
        shard_id - gpu_instance.shard_config.num_shards * gpu_instance.shard_config.device_index;
    const auto& gpu_buffer = gpu_instance.getGPUGraphBuffer(on_gpu_shard_id);
    const auto& gpu_base_buffer = gpu_instance.getGPUBaseBuffer(on_gpu_shard_id);
    const auto& graph = gpu_buffer.graph;
    const Dataset<BaseT>& base = gpu_base_buffer.base;
    const cudaStream_t stream = gpu_buffer.stream.get();

    static constexpr bool WRITE_DISTS = true;
    static constexpr bool DIST_STATS = false;

    uint32_t* m_dist_statistics = nullptr;

    if constexpr (DIST_STATS)
      cudaMallocAsync(&m_dist_statistics, query.N * sizeof(uint32_t), stream);

    CHECK_LT(base.N, std::numeric_limits<uint32_t>::max());

    auto run_query = [&]<uint32_t BLOCK_DIM_X>() {
      using QueryKernel =
          ggnn::QueryKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, WRITE_DISTS, DIST_STATS>;

      QueryKernel query_kernel{
          .D = query.D,
          .measure = measure,
          .KQuery = KQuery,
          .sorted_size = sorted_size,
          .cache_size = cache_size,
          .tau_query = tau_query,
          .max_iterations = max_iters,
          .N_base = static_cast<KeyT>(base.N),
          .KBuild = gpu_instance.graph_config.KBuild,
          .num_starting_points = gpu_instance.graph_config.S,
          .d_base = base.data(),
          .d_query = query.data(),
          .d_graph = graph.graph[0].data(),
          .d_starting_points = graph.translation[GraphConfig::L - 1].data(),
          .d_nn1_stats = graph.nn1_stats.data(),
          .d_query_results = results.ids.data(),
          .d_query_results_dists = results.dists.data(),
          .d_dist_stats = m_dist_statistics,
          .shards_per_gpu = gpu_instance.shard_config.num_shards,
          .on_gpu_shard_id = on_gpu_shard_id,
      };

      query_kernel.launch(query.N, gpu_buffer.stream.get());
    };

    if constexpr (DIST_STATS)
      CHECK_CUDA(cudaFreeAsync(m_dist_statistics, gpu_buffer.stream.get()));

    switch (block_dim_x) {
      case 32:
        run_query.template operator()<32>();
        break;
      case 64:
        run_query.template operator()<64>();
        break;
      case 128:
        run_query.template operator()<128>();
        break;
      case 256:
        run_query.template operator()<256>();
        break;
      case 512:
        run_query.template operator()<512>();
        break;
      case 1024:
        run_query.template operator()<1024>();
        break;
      default:
        LOG(DFATAL) << "The query has not been compiled for BLOCK_DIM_X == " << block_dim_x << ".";
    }
  }

  virtual void bruteForceQuery(const Dataset<BaseT>& base, const Dataset<BaseT>& query,
                               const uint32_t KQuery, Results& results,
                               cudaStream_t stream) override
  {
    // number of query dimensions cached per thread in the distance computation
    static constexpr uint32_t DIST_ITEMS_PER_THREAD = 4;
    // CUDA warp size
    static constexpr uint32_t WARP_SIZE = 32;
    // CUDA block size limit
    static constexpr uint32_t MAX_BLOCK_DIM_X = 1024;
    static constexpr uint32_t MAX_DIM = MAX_BLOCK_DIM_X * DIST_ITEMS_PER_THREAD;
    static constexpr uint32_t MAX_K_QUERY =
        6000;  // making it larger would exceed the 48k shared memory limit

    CHECK_LE(KQuery, MAX_K_QUERY);

    /** number of threads required for processing data with a certain dimension:
     * 128D ==> 32 threads
     * 256D ==> 64 threads
     * 512D ==> 128 threads
     * 1024D ==> 256 threads
     * 2048D ==> 512 threads
     * 4096D ==> 1024 threads
     */
    const uint32_t dimension_block_dim_x =
        bit_ceil((query.D + DIST_ITEMS_PER_THREAD - 1) / DIST_ITEMS_PER_THREAD);
    const uint32_t block_dim_x = std::max({WARP_SIZE, dimension_block_dim_x});

    CHECK_LE(query.D, MAX_DIM);
    CHECK_LE(block_dim_x, MAX_BLOCK_DIM_X);

    static constexpr bool WRITE_DISTS = true;

    CHECK_LT(base.N, std::numeric_limits<KeyT>::max());

    auto run_bf_query = [&]<uint32_t BLOCK_DIM_X>() {
      using BFQueryKernel =
          ggnn::BruteForceQueryKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, WRITE_DISTS>;

      BFQueryKernel query_kernel{
          .D = query.D,
          .measure = measure,
          .KQuery = KQuery,
          .N_base = static_cast<KeyT>(base.N),  // this applies to potential subsets
          .d_base = base.data(),
          .d_query = query.data(),
          .d_query_results = results.ids.data(),
          .d_query_results_dists = results.dists.data(),
      };

      query_kernel.launch(query.N, stream);
    };

    switch (block_dim_x) {
      case 32:
        run_bf_query.template operator()<32>();
        break;
      case 64:
        run_bf_query.template operator()<64>();
        break;
      case 128:
        run_bf_query.template operator()<128>();
        break;
      case 256:
        run_bf_query.template operator()<256>();
        break;
      case 512:
        run_bf_query.template operator()<512>();
        break;
      case 1024:
        run_bf_query.template operator()<1024>();
        break;
      default:
        LOG(DFATAL) << "The brute-force query has not been compiled for BLOCK_DIM_X == "
                    << block_dim_x << ".";
    }
  }
};

template <typename KeyT, typename ValueT, typename BaseT>
QueryKernels<KeyT, ValueT, BaseT>::QueryKernels(const DistanceMeasure measure)
{
  pimpl.reset(new QueryKernelsImpl<KeyT, ValueT, BaseT>{measure});
}

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_INSTANTIATE_CLASS, QueryKernels);
GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_INSTANTIATE_CLASS, QueryKernelsImpl);
};  // namespace ggnn
