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

#ifndef CUDA_KNN_SYM_QUERY_LAYER_CUH_
#define CUDA_KNN_SYM_QUERY_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/cache/cuda_knn_cache_sym.cuh"
#include "ggnn/cache/cuda_knn_multi_worked_dists_cache.cuh"
#include "ggnn/config.hpp"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename ValueT, typename KeyT, int D, int K, int KF, int BLOCK_DIM_X,
          typename BaseT = ValueT, typename BAddrT = int32_t,
          typename GAddrT = int32_t>
struct SymQueryKernel {
  static constexpr int KL = K - KF;

  static constexpr int MAX_PER_PATH_ITERATIONS = 20;

  static constexpr int CASHE_SIZE = 256;
  static constexpr int PRIOQ_SIZE = 192;
  static constexpr int BEST_SIZE = K;

  typedef SortedBufferSymCache<ValueT, KeyT, KL, D, BLOCK_DIM_X, CASHE_SIZE,
                               PRIOQ_SIZE, BEST_SIZE, BaseT, BAddrT>
      Cache;

  void launch() {
    lprintf(1, "SymQueryKernel -- Layer: %d | N: %d [%d %d] \n", layer, N,
            N_offset, N_offset + N);
    launcher<<<N, BLOCK_DIM_X>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    const float xi =
        (d_nn1_stats[0] * d_nn1_stats[0]) * c_tau_build * c_tau_build;

    const KeyT n = N_offset + (int)blockIdx.x;
    const KeyT m = (!layer) ? n : d_translation[n];

    Cache cache(d_base, n, m, xi);

    int counter = 0;

    __shared__ KeyT s_knn[K];
    __shared__ KeyT s_sym_ids[KL];

    if (threadIdx.x < KL) {
      const KeyT sym_n = d_graph[static_cast<GAddrT>(n) * K + threadIdx.x];
      s_sym_ids[threadIdx.x] = sym_n;
    }
    for (int k = 0; k < KL; k++) {
      __syncthreads();

      cache.init_start_point(s_sym_ids[k], (layer) ? d_translation : nullptr);
      __syncthreads();

      bool result = true;

      for (int ite = 0; ite < MAX_PER_PATH_ITERATIONS; ++ite) {
        __syncthreads();
        const KeyT anchor = cache.pop();

        if (anchor == Cache::EMPTY_KEY) {
          break;
        }

        if (threadIdx.x < K) {
          const KeyT other_id =
              (threadIdx.x < KL)
                  ? d_graph[static_cast<GAddrT>(anchor) * K + threadIdx.x]
                  : d_sym_buffer[static_cast<GAddrT>(anchor) * KF +
                                 threadIdx.x - KL];

          s_knn[threadIdx.x] = other_id;
        }
        __syncthreads();

        result = cache.fetch(s_knn, (layer) ? d_translation : nullptr, K);

        if (result) {
          break;
        }
      }  // end per k iteration
      if (!result) {
        if (!threadIdx.x) {
          for (int i = 0; i < BEST_SIZE && !result; i++) {
            const KeyT other_n = cache.s_cache[i];
            if (other_n == Cache::EMPTY_KEY) break;
            constexpr KeyT inc = 1;
            const int pos = atomicAdd(&d_sym_atomic[other_n], inc);
            if (pos < KF) {
              d_sym_buffer[static_cast<GAddrT>(other_n) * KF + pos] = n;
              cache.set_connected(other_n);
              result = true;
            }
          }
          if (!result) {
            counter++;
          }
        }
      }
    }  // end k neighbors

    if (!threadIdx.x) {
      d_stats[n] = counter;
    }
  }

  const BaseT* d_base;        // [N0,D]
  const KeyT* d_graph;        // [N,K]
  const KeyT* d_translation;  // [N]

  KeyT* d_sym_atomic;  // [N]
  KeyT* d_sym_buffer;  // [N,KF]

  const float* d_nn1_stats;
  int* d_stats;

  int layer;

  int N;  // number of points to work on
  int N_offset;
};

#endif  // CUDA_KNN_SYM_QUERY_LAYER_CUH_
