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

#ifndef CUDA_KNN_QUERY_LAYER_CUH_
#define CUDA_KNN_QUERY_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/cache/cuda_knn_multi_worked_dists_cache.cuh"
#include "ggnn/cache/cuda_knn_sorted_buffer_cache.cuh"
#include "ggnn/config.hpp"
#include "ggnn/cuda_knn_config.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename ValueT, typename KeyT, int D, int K, int KF, int KQuery,
          int S, int BLOCK_DIM_X, typename BaseT = ValueT,
          typename BAddrT = int32_t, typename GAddrT = int32_t,
          bool DIST_STATS = false, bool OVERFLOW_STATS = false,
          int MAX_ITERATIONS = 400, int CACHE_SIZE = 512, int PRIOQ_SIZE = 256,
          bool WRITE_DISTS = false>
struct QueryKernel {
  static constexpr int KL = K - KF;
  static constexpr int KS = (K > S) ? K : S;

  static constexpr int BEST_SIZE = KQuery;

  static constexpr bool DIRECT_BTM = true;

  typedef SortedBufferCache<ValueT, KeyT, KQuery, D, BLOCK_DIM_X, CACHE_SIZE,
                            PRIOQ_SIZE, BEST_SIZE, BaseT, BAddrT, DIST_STATS,
                            OVERFLOW_STATS>
      Cache;

  void launch() {
    lprintf(0,
            "QueryKernel: KQuery %d, "
            "MAX_ITERATIONS "
            "%d, "
            "CACHE_SIZE %d\n",
            KQuery, MAX_ITERATIONS, CACHE_SIZE);
    launcher<<<N, BLOCK_DIM_X>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    const float xi =
        (d_nn1_stats[1] * d_nn1_stats[1]) * c_tau_query * c_tau_query;

    const KeyT n = N_offset + (int)blockIdx.x;

    Cache cache(d_base, d_query, n, xi);

    __syncthreads();

    __shared__ KeyT s_knn[KS];
    if (threadIdx.x < S) s_knn[threadIdx.x] = threadIdx.x;
    __syncthreads();

    cache.fetch(s_knn, &d_translation[c_STs_offsets[c_L - 1]], S);
    __syncthreads();

    for (int layer = ((DIRECT_BTM) ? 0 : (c_L - 2)); layer >= 0; layer--) {
      __syncthreads();
      if (DIRECT_BTM)
        cache.transform(&d_translation[c_STs_offsets[c_L - 1]]);
      else
        cache.transform(&d_selection[c_STs_offsets[layer + 1]]);

      __syncthreads();

      for (int ite = 0; ite < MAX_ITERATIONS; ++ite) {
        __syncthreads();

        cache.xi = min(xi, cache.s_dists[0].dist * c_tau_query * c_tau_query);

        const KeyT anchor = cache.pop();
        if (anchor == Cache::EMPTY_KEY) {
          break;
        }
        __syncthreads();

        if (threadIdx.x < K) {
          s_knn[threadIdx.x] =
              d_graph[(static_cast<GAddrT>(c_Ns_offsets[layer]) + anchor) * K +
                      threadIdx.x];
        }

        __syncthreads();
        cache.fetch(s_knn, nullptr, K);

      }  // end iterations
    }

    __syncthreads();
    cache.write_best(d_query_results, n, KQuery);

    if (WRITE_DISTS) {
      if (threadIdx.x < KQuery) {
        d_query_results_dists[n * KQuery + threadIdx.x] =
            cache.s_dists[threadIdx.x].dist;
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
  const KeyT* d_selection;    // [Sall]

  const KeyT* d_graph;            // [Nall,K]
  KeyT* d_query_results;          // [Nq,KQuery]
  ValueT* d_query_results_dists;  // [Nq,KQuery]

  const float* d_nn1_stats;  // [sum,max]

  const float* d_nn1_buffer;  // [N0]
  int* d_dist_stats;          // [Nq]

  int N;         // number of points to query for -> Nq
  int N_offset;  // gpu offset in N
};

#endif  // CUDA_KNN_QUERY_LAYER_CUH_
