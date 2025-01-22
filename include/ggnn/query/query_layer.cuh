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

#ifndef INCLUDE_GGNN_QUERY_LAYER_CUH
#define INCLUDE_GGNN_QUERY_LAYER_CUH

#include <ggnn/base/def.h>
#include <glog/logging.h>

#include <cstdint>

namespace ggnn {

template <typename T>
__global__ void query(const T kernel);

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          bool WRITE_DISTS = false, bool DIST_STATS = false>
struct QueryKernel {
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;
  static constexpr uint32_t DIST_ITEMS_PER_THREAD = 4;

  static constexpr uint32_t MAX_SM = 48 * 1024;

  void launch(const uint32_t N, const cudaStream_t stream = 0)
  {
    VLOG(1) << "QueryKernel -- BLOCK_DIM_X: " << BLOCK_DIM_X << " || KQuery: " << KQuery
            << " MAX_ITERATIONS: " << max_iterations << " CACHE_SIZE: " << cache_size
            << " SORTED_SIZE: " << sorted_size;
    uint32_t sm_size = cache_size * sizeof(KeyT) + sorted_size * sizeof(ValueT);
    CHECK_LT(KQuery, sorted_size);
    CHECK_LT(sorted_size, cache_size);
    CHECK_LT(sm_size, MAX_SM);

    CHECK_LE(D, BLOCK_DIM_X * DIST_ITEMS_PER_THREAD);

    query<<<N, BLOCK_DIM_X, sm_size, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const;

  const uint32_t D;
  const DistanceMeasure measure;

  const uint32_t KQuery;
  const uint32_t sorted_size;
  const uint32_t cache_size;

  const float tau_query;
  const uint32_t max_iterations;

  KeyT N_base;  // number of points in the dataset

  const uint32_t KBuild;
  const uint32_t num_starting_points;

  const BaseT* d_base;   // [Nall,D]
  const BaseT* d_query;  // [Nq,D]

  const KeyT* d_graph;            // [Nall,K]
  const KeyT* d_starting_points;  // [S]

  const float* d_nn1_stats;  // [sum,max]

  KeyT* d_query_results;          // [Nq,KQuery]
  ValueT* d_query_results_dists;  // [Nq,KQuery]

  uint32_t* d_dist_stats;  // [Nq]

  const uint32_t shards_per_gpu{1};
  const uint32_t on_gpu_shard_id{0};
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_QUERY_LAYER_CUH
