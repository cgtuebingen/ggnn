/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.

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

#ifndef INCLUDE_GGNN_QUERY_CUDA_KNN_NO_SLACK_QUERY_LAYER_CUH_
#define INCLUDE_GGNN_QUERY_CUDA_KNN_NO_SLACK_QUERY_LAYER_CUH_

#include <algorithm>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// #include "ggnn/cache/cuda_knn_sorted_buffer_cache.cuh"
#include "ggnn/cache/cuda_simple_knn_cache_no_slack.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"


template <typename T>
__global__ void
no_slack_query(const T kernel) {
  kernel();
}

template <DistanceMeasure measure,
          typename ValueT, typename KeyT, int D, int K, int KF, int KQuery,
          int S, int BLOCK_DIM_X, typename BaseT = ValueT,
          typename BAddrT = int32_t, typename GAddrT = int32_t,
          bool DIST_STATS = false, bool OVERFLOW_STATS = false,
          int MAX_ITERATIONS = 400, int CACHE_SIZE = 512, int SORTED_SIZE = 256,
          int BEST_SIZE = 128, bool WRITE_DISTS = false>
struct NoSlackQueryKernel {
  static constexpr int KL = K - KF;
  static constexpr int KS = (K > S) ? K : S;

  static_assert(BEST_SIZE >= KQuery, "best size needs to be at least KQuery");
  static constexpr int VISITED_SIZE = CACHE_SIZE - SORTED_SIZE;
  static constexpr int PRIOQ_SIZE = SORTED_SIZE - BEST_SIZE;

  static constexpr int ITERATIONS_FOR_K = (K+BLOCK_DIM_X-1)/BLOCK_DIM_X;
  static constexpr int ITERATIONS_FOR_S = (S+BLOCK_DIM_X-1)/BLOCK_DIM_X;

  typedef SimpleKNNCacheNoSlack<measure, ValueT, KeyT, KQuery, D, BLOCK_DIM_X,
                            VISITED_SIZE, PRIOQ_SIZE, BEST_SIZE,
                            BaseT, BAddrT, DIST_STATS, OVERFLOW_STATS>
      Cache;

  void launch(const cudaStream_t stream = 0) {
    VLOG(1) << "NoSlackQueryKernel -- BLOCK_DIM_X: " << BLOCK_DIM_X
               << " || KQuery: " << KQuery
               << " MAX_ITERATIONS: " << MAX_ITERATIONS
               << " CACHE_SIZE: " << CACHE_SIZE
               << " SORTED_SIZE: " << SORTED_SIZE
               << " || BEST_SIZE: " << BEST_SIZE
               << " PRIOQ_SIZE: " << PRIOQ_SIZE
               << " VISITED_SIZE: " << VISITED_SIZE;
    no_slack_query<<<N, BLOCK_DIM_X, 0, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    const KeyT n = N_offset + static_cast<int>(blockIdx.x);

    Cache cache(d_base, d_query, n);
    __syncthreads();


    __shared__ KeyT s_knn[KS];
    for (int i=0; i < ITERATIONS_FOR_S; ++i) {
      const int s = i*BLOCK_DIM_X+threadIdx.x;
      if (s < S)
        s_knn[s] = d_translation[c_STs_offsets[c_L - 1]+s];
    }
    __syncthreads();

    cache.fetch(s_knn, nullptr, S);
    __syncthreads();

    for (int ite = 0; ite < MAX_ITERATIONS; ++ite) {
      __syncthreads();

      const KeyT anchor = cache.pop();
      if (anchor == Cache::EMPTY_KEY) {
        break;
      }
      __syncthreads();

      for (int i=0; i < ITERATIONS_FOR_K; ++i) {
        const int k = i*BLOCK_DIM_X+threadIdx.x;
        if (k < K)
          s_knn[k] =
              d_graph[static_cast<GAddrT>(anchor) * K + k];
      }

      __syncthreads();
      cache.fetch(s_knn, nullptr, K);

    }  // end iterations

    __syncthreads();
    cache.write_best(d_query_results, n, KQuery);

    if (WRITE_DISTS) {
      if (threadIdx.x < KQuery) {
        d_query_results_dists[n * KQuery + threadIdx.x] =
            cache.s_dists[threadIdx.x];
      }
    }

    if (DIST_STATS) {
      if (!threadIdx.x) {
        d_dist_stats[n] = cache.get_dist_stats();
      }
    }
  }

  const BaseT* d_base;        // [Nall,D]
  const BaseT* d_query;       // [Nq,D]
  const KeyT* d_translation;  // [Nall]

  const KeyT* d_graph;            // [Nall,K]
  KeyT* d_query_results;          // [Nq,KQuery]
  ValueT* d_query_results_dists;  // [Nq,KQuery]

  const float* d_nn1_stats;  // [sum,max]

  int* d_dist_stats;          // [Nq]

  int N;         // number of points to query for -> Nq
  int N_offset;  // gpu offset in N
};

#endif  // INCLUDE_GGNN_QUERY_CUDA_KNN_NO_SLACK_QUERY_LAYER_CUH_
