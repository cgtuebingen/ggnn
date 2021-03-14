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

#ifndef INCLUDE_GGNN_MERGE_CUDA_KNN_MERGE_LAYER_CUH_
#define INCLUDE_GGNN_MERGE_CUDA_KNN_MERGE_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/cache/cuda_simple_knn_cache.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename T>
__global__ void
merge(const T kernel) {
  kernel();
}

template <DistanceMeasure measure,
          typename ValueT, typename KeyT, int D, int K, int KF, int S,
          int BLOCK_DIM_X, typename BaseT = ValueT, typename BAddrT = int32_t,
          typename GAddrT = int32_t>
struct MergeKernel {
  static constexpr int KL = K - KF;
  static constexpr int MAX_ITERATIONS = 200;

  // this allows for loop unrolling
  static constexpr int ITERATIONS_FOR_K = (K+BLOCK_DIM_X-1)/BLOCK_DIM_X;
  static constexpr int ITERATIONS_FOR_S = (S+BLOCK_DIM_X-1)/BLOCK_DIM_X;

  static constexpr int KQuery = K;
  static constexpr int SORTED_SIZE = 128;
  // keep the visited list just long enough to keep track of all visited points
  static constexpr int CACHE_SIZE = ((SORTED_SIZE+MAX_ITERATIONS+BLOCK_DIM_X-1)
                                     /BLOCK_DIM_X)*BLOCK_DIM_X;

  static constexpr int BEST_SIZE = KQuery;
  static constexpr int VISITED_SIZE = CACHE_SIZE - SORTED_SIZE;
  static constexpr int PRIOQ_SIZE = SORTED_SIZE - BEST_SIZE;

  static constexpr bool DIST_STATS = false;
  static constexpr bool OVERFLOW_STATS = false;

  typedef SimpleKNNCache<measure, ValueT, KeyT, KQuery, D, BLOCK_DIM_X, VISITED_SIZE,
                          PRIOQ_SIZE, BEST_SIZE, BaseT, BAddrT, DIST_STATS,
                          OVERFLOW_STATS>
    Cache;

  void launch(const cudaStream_t stream = 0) {
    CHECK_GT(layer_top, layer_btm);
    VLOG(1) << "MergeKernel -- Layer: " << layer_top << " -> " << layer_btm
            << " |  N: " << N << " [" << N_offset << " " << N_offset+N << "] \n";
    merge<<<N, BLOCK_DIM_X, 0, stream>>>((*this));
  }

  // determine the start of the top-layer segment (always 0 for layer_top = L-1)
  __device__ __forceinline__ int get_top_seg_offset(const KeyT n) const {
    int seg_btm;
    if (!layer_btm) {
      seg_btm = n / (c_S0 + 1);
      if (seg_btm >= c_S0_offset)
        seg_btm = c_S0_offset + (n - (c_S0_offset * (c_S0 + 1))) / c_S0;
    } else {
      seg_btm = n / S;
    }

    int powG = c_G; //assuming layer_top > layer_btm (which should always be the case)
    for (int i=1; i<layer_top-layer_btm; ++i)
      powG *= c_G;

    return (seg_btm / powG) * S;
  }

  __device__ __forceinline__ void operator()() const {
    const float xi = (measure == Euclidean) ?
        (d_nn1_stats[0] * d_nn1_stats[0]) * c_tau_build * c_tau_build
        : d_nn1_stats[0] * c_tau_build;

    const KeyT n = N_offset + static_cast<int>(blockIdx.x);

    const KeyT m =
        (!layer_btm) ? n : d_translation[c_STs_offsets[layer_btm] + n];

    Cache cache(d_base, m, xi);

    const int s_offset = get_top_seg_offset(n);

    __shared__ KeyT s_knn[(K > S) ? K : S];

    for (int i=0; i < ITERATIONS_FOR_S; ++i) {
      const int s = i*BLOCK_DIM_X+threadIdx.x;
      if (s < S) {
        s_knn[s] = s_offset + s;
      }
    }
    __syncthreads();
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
        for (int i=0; i < ITERATIONS_FOR_K; ++i) {
          const int k = i*BLOCK_DIM_X+threadIdx.x;
          if (k < K) {
            s_knn[k] =
                d_graph[(static_cast<GAddrT>(c_Ns_offsets[layer]) + anchor)
                        * K + k];
          }
        }
        __syncthreads();

        cache.fetch(s_knn,
                    (!layer) ? nullptr : &d_translation[c_STs_offsets[layer]],
                    K);

      }
    }

    __syncthreads();

    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        const KeyT idx = cache.s_cache[k + 1];
        d_graph_buffer[static_cast<GAddrT>(n) * K + k] =
            (idx != Cache::EMPTY_KEY) ? idx : n;
      }
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

#endif  // INCLUDE_GGNN_MERGE_CUDA_KNN_MERGE_LAYER_CUH_
