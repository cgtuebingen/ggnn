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

#include <ggnn/query/query_layer.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/lib.h>
#include <ggnn/cuda_utils/distance.cuh>
#include <ggnn/cuda_utils/simple_knn_cache.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ggnn {

template <typename T>
__global__ void __launch_bounds__(T::BLOCK_DIM_X) query(const T kernel)
{
  kernel();
}

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE, bool WRITE_DISTS,
          bool DIST_STATS>
__device__ __forceinline__ void
QueryKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, WRITE_DISTS, DIST_STATS>::operator()() const
{
  static constexpr uint32_t K_BLOCK = 32;

  using Cache = SimpleKNNCache<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD, DIST_STATS>;

  const float xi = (measure == DistanceMeasure::Euclidean)
                       ? (d_nn1_stats[1] * d_nn1_stats[1]) * tau_query * tau_query
                       : d_nn1_stats[1] * tau_query;

  const KeyT n = static_cast<KeyT>(blockIdx.x);

  Cache cache(D, measure, KQuery, sorted_size, cache_size, d_base, d_query, n, xi);
  cache.fetch_unfiltered(d_starting_points, nullptr, num_starting_points);

  for (uint32_t ite = 0; ite < max_iterations; ++ite) {
    if (measure == DistanceMeasure::Euclidean) {
      cache.r_xi = min(xi, cache.s_dists[0] * tau_query * tau_query);
    }
    else if (measure == DistanceMeasure::Cosine) {
      cache.r_xi = min(xi, cache.s_dists[0] * tau_query);
    }

    const KeyT anchor = cache.pop();
    if (anchor == Cache::EMPTY_KEY)
      break;

    __shared__ KeyT s_knn[K_BLOCK];
    for (uint32_t i = 0; i < KBuild; i += K_BLOCK) {
      if (threadIdx.x < K_BLOCK) {
        s_knn[threadIdx.x] = (i + threadIdx.x < KBuild)
                                 ? d_graph[static_cast<size_t>(anchor) * KBuild + i + threadIdx.x]
                                 : Cache::EMPTY_KEY;
      }
      cache.fetch(s_knn, nullptr, K_BLOCK);
    }

  }  // end iterations

  __syncthreads();
  cache.write_best(d_query_results, n * shards_per_gpu + on_gpu_shard_id, KQuery,
                   on_gpu_shard_id * N_base);

  if constexpr (WRITE_DISTS) {
#pragma unroll
    for (uint32_t k = threadIdx.x; k < KQuery; k += BLOCK_DIM_X) {
      d_query_results_dists[(n * shards_per_gpu + on_gpu_shard_id) * KQuery + k] = cache.s_dists[k];
    }
  }

  if constexpr (DIST_STATS) {
    if (!threadIdx.x) {
      d_dist_stats[n] = cache.get_dist_stats();
    }
  }
}

#define GGNN_QUERY(KeyT, ValueT, BaseT, BLOCK_DIM_X, WRITE_DISTS, DIST_STATS)    \
  template __global__ void                                                       \
  query<QueryKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, WRITE_DISTS, DIST_STATS>>( \
      const QueryKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, WRITE_DISTS, DIST_STATS>);

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_QUERYS, GGNN_WRITE_DISTS, GGNN_DIST_STATS,
          GGNN_QUERY);

};  // namespace ggnn
