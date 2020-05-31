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

#ifndef CUDA_KNN_MERGE_LAYER_CUH_
#define CUDA_KNN_MERGE_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/cache/cuda_knn_multi_worked_dists_cache.cuh"
#include "ggnn/cache/cuda_knn_sorted_buffer_cache.cuh"
#include "ggnn/config.hpp"
#include "ggnn/cuda_knn_config.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename ValueT, typename KeyT, int D, int K, int KF, int S,
          int BLOCK_DIM_X, typename BaseT = ValueT,
          typename BAddrT = int32_t, typename GAddrT = int32_t>
struct MergeKernel {
  static constexpr int KL = K - KF;
  static constexpr int MAX_ITERATIONS = 200;

  // static constexpr int CACHE_SIZE = 256;
  // static constexpr int PRIOQ_SIZE = 192;
  // static constexpr int BEST_SIZE = 32;

  static constexpr int CACHE_SIZE = 512;
  static constexpr int PRIOQ_SIZE = 256;
  static constexpr int BEST_SIZE = 32;

  static constexpr int KQuery = KL;

  // typedef LinearCache<ValueT, KeyT, K, KQuery, BLOCK_DIM_X, HASH_MAP_SIZE, D,
  //                     BaseT, BAddrT>
  //     Cache;

  // typedef SortedLinearCache<ValueT, KeyT, K, KQuery, BLOCK_DIM_X,
  // HASH_MAP_SIZE,
  //                           D, BaseT, BAddrT>
  //     Cache;

  typedef SortedBufferCache<ValueT, KeyT, KQuery, D, BLOCK_DIM_X, CACHE_SIZE,
                            PRIOQ_SIZE, BEST_SIZE, BaseT, BAddrT>
      Cache;

  void launch() {
    lprintf(1, "MergeKernel -- Layer: %d -> %d |  N: %d [%d %d] \n", layer_top,
            layer_btm, N, N_offset, N_offset + N);
    launcher<<<N, BLOCK_DIM_X>>>((*this));
  }

  __device__ __forceinline__ int get_seg_offset(const KeyT n) const {
    int s_seg_btm = -1;
    if (!layer_btm) {
      const int S0_plus_offset = c_S0_offset * (c_S0 + 1);
      const bool is_offset = n < S0_plus_offset;
      const int S0_actual = (is_offset) ? c_S0 + 1 : c_S0;
      s_seg_btm = (is_offset)
                      ? (n / S0_actual)
                      : (c_S0_offset + (n - S0_plus_offset) / S0_actual);
    } else {
      s_seg_btm = n / S;
    }

    return ((int)(s_seg_btm / pow(c_G, layer_top - layer_btm))) * S;
  }

  __device__ __forceinline__ void operator()() const {
    const float xi =
        (d_nn1_stats[0] * d_nn1_stats[0]) * c_tau_build * c_tau_build;

    const KeyT n = N_offset + (int)blockIdx.x;

    const KeyT m =
        (!layer_btm) ? n : d_translation[c_STs_offsets[layer_btm] + n];

    Cache cache(d_base, m, xi);

    const int s_offset = get_seg_offset(n);

    __shared__ KeyT s_knn[(K > S) ? K : S];

    for (int s = threadIdx.x; s < S; s += BLOCK_DIM_X) {
      s_knn[s] = s_offset + s;
    }
    cache.fetch(s_knn, &d_translation[c_STs_offsets[layer_top]], S);

    for (int layer = layer_top - 1; layer >= layer_btm; layer--) {
      __syncthreads();

      cache.transform(&d_selection[c_STs_offsets[layer + 1]]);
      __syncthreads();

      if (layer == layer_btm) {
        if (!threadIdx.x) s_knn[0] = n;
        __syncthreads();
        cache.fetch(
            s_knn, (layer > 0) ? &d_translation[c_STs_offsets[layer]] : nullptr,
            1);
      }
      __syncthreads();

      for (int ite = 0; ite < MAX_ITERATIONS; ++ite) {
        __syncthreads();
        const KeyT anchor = cache.pop();
        if (anchor == Cache::EMPTY_KEY) {
          break;
        }
        if (threadIdx.x < K) {
          s_knn[threadIdx.x] =
              d_graph[(static_cast<GAddrT>(c_Ns_offsets[layer]) + anchor) * K +
                      threadIdx.x];
        }
        __syncthreads();

        cache.fetch(s_knn,
                    (!layer) ? nullptr : &d_translation[c_STs_offsets[layer]],
                    K);
      }
    }

    __syncthreads();

    if (threadIdx.x < K) {
      const KeyT idx = cache.s_cache[threadIdx.x + 1];
      d_graph_buffer[static_cast<GAddrT>(n) * K + threadIdx.x] =
          (idx != Cache::EMPTY_KEY) ? idx : n;
    }

    if (!layer_btm && !threadIdx.x) {
      d_nn1_dist_buffer[n] = cache.get_nn1_dist();
    }
  }

  const BaseT* d_base;        // [Nall,D]
  const KeyT* d_translation;  // [Nall]
  const KeyT* d_selection;    // [Sall]

  const KeyT* d_graph;   // [N,K]
  KeyT* d_graph_buffer;  // [N,K]

  const float* d_nn1_stats;  // [sum,max]
  float* d_nn1_dist_buffer;  // [N0]

  int N;         // number of points to work on
  int N_offset;  // gpu offset in N

  int layer_btm;  // layer to merge
  int layer_top;  // layer to start
};

#endif  // CUDA_KNN_MERGE_LAYER_CUH_
