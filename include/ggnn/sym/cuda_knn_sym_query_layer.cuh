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

#ifndef INCLUDE_GGNN_SYM_CUDA_KNN_SYM_QUERY_LAYER_CUH_
#define INCLUDE_GGNN_SYM_CUDA_KNN_SYM_QUERY_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/cache/cuda_simple_knn_sym_cache.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename T>
__global__ void
sym(const T kernel) {
  kernel();
}

template <DistanceMeasure measure,
          typename ValueT, typename KeyT, int D, int K, int KF, int BLOCK_DIM_X,
          typename BaseT = ValueT, typename BAddrT = int32_t,
          typename GAddrT = int32_t>
struct SymQueryKernel {
  static constexpr int KL = K - KF;

  // this allows for loop unrolling
  static constexpr int ITERATIONS_FOR_K = (K+BLOCK_DIM_X-1)/BLOCK_DIM_X;
  static constexpr int ITERATIONS_FOR_KL = (KL+BLOCK_DIM_X-1)/BLOCK_DIM_X;

  static constexpr int MAX_PER_PATH_ITERATIONS = 20;

  static constexpr int KQuery = KL;
  static constexpr int CACHE_SIZE = 256;
  static constexpr int SORTED_SIZE = 128;

  static constexpr int BEST_SIZE = KQuery;
  static constexpr int VISITED_SIZE = CACHE_SIZE - SORTED_SIZE;
  static constexpr int PRIOQ_SIZE = SORTED_SIZE - BEST_SIZE;

  static constexpr bool DIST_STATS = false;
  static constexpr bool OVERFLOW_STATS = false;

  typedef SimpleKNNSymCache<measure, ValueT, KeyT, KL, D, BLOCK_DIM_X, VISITED_SIZE,
                            PRIOQ_SIZE, BEST_SIZE, BaseT, BAddrT, DIST_STATS,
                            OVERFLOW_STATS>
      Cache;

  void launch(const cudaStream_t stream = 0) {
    VLOG(1) << "SymQueryKernel -- Layer: " << layer << " | N: " << N << " ["
               << N_offset << " " << N_offset + N << "]";
    sym<<<N, BLOCK_DIM_X, 0, stream>>>((*this));

  }

  __device__ __forceinline__ void operator()() const {
    const float xi =
        (measure == Euclidean)
            ? (d_nn1_stats[0] * d_nn1_stats[0]) * c_tau_build * c_tau_build
            : d_nn1_stats[0] * c_tau_build;

    const KeyT n = N_offset + static_cast<int>(blockIdx.x);

    Cache cache(d_base, (layer) ? d_translation[n] : n, xi);

    int counter = 0;

    __shared__ KeyT s_knn[K];
    __shared__ KeyT s_sym_ids[KL];
    __shared__ bool s_connected;

    // fetch neighbors in local neighbor list
    for (int i=0; i < ITERATIONS_FOR_KL; ++i) {
      const int kl = i*BLOCK_DIM_X+threadIdx.x;
      if (kl < KL) {
        const KeyT sym_n = d_graph[static_cast<GAddrT>(n) * K + kl];
        s_sym_ids[kl] = sym_n;
      }
    }
    for (int k = 0; k < KL; k++) {
      __syncthreads();
      if (!threadIdx.x) s_connected = false;

      // search for k-th local neighbor
      cache.init_start_point(s_sym_ids[k], (layer) ? d_translation : nullptr);

      bool result = false;

      for (int ite = 0; ite < MAX_PER_PATH_ITERATIONS && !result; ++ite) {
        __syncthreads();

        const KeyT anchor = cache.pop();

        if (anchor == Cache::EMPTY_KEY) {
          break;
        }

        // fetch neighbors at anchor point + points in sym buffer
        for (int i=0; i < ITERATIONS_FOR_K; ++i) {
          const int k = i*BLOCK_DIM_X+threadIdx.x;
          if (k < K) {
            const KeyT other_id =
                (k < KL)
                    ? d_graph[static_cast<GAddrT>(anchor) * K + k]
                    : d_sym_buffer[static_cast<GAddrT>(anchor) * KF +
                                   k - KL];
            if (other_id == n) {
              s_connected = true;
            }
            s_knn[k] = other_id;
          }
        }
        __syncthreads();


        // stop if the original index n has been found as a neighbor
        if(s_connected){
          result = true;
        }
        else
        {
          cache.fetch(s_knn, (layer) ? d_translation : nullptr, K);
        }

      }  // end per k iteration

      if (!result) {
        // we need to add a symmetric link to the original index n
        if (!threadIdx.x) {
          for (int i = 0; i < BEST_SIZE && !result; i++) {
            // try to enter the symmetric link at the i-th nearest neighbor
            // found on the path
            const KeyT other_n = cache.s_cache[i];
            if (other_n == Cache::EMPTY_KEY) break;
            const int pos = atomicAdd(&d_sym_atomic[other_n], 1);
            if (pos < KF) {
              d_sym_buffer[static_cast<GAddrT>(other_n) * KF + pos] = n;
              // cache.set_connected(other_n);
              result = true;
            }
          }
          // could not add a link, increment the counter
          if (!result) {
            counter++;
          }
        }
      }
    }  // end k neighbors

    if (OVERFLOW_STATS && !threadIdx.x) {
      d_stats[n] = counter;
    }
  }

  const BaseT* d_base;        // [N0,D]
  const KeyT* d_graph;        // [N,K]
  const KeyT* d_translation;  // [N] or nullptr if on base layer

  int* d_sym_atomic;  // [N]
  KeyT* d_sym_buffer;  // [N,KF]

  const float* d_nn1_stats;
  int* d_stats;  // number of links which could not be established

  // although this provides no additional useful information to the kernel,
  // it compiles to a faster version than checking for d_translation == nullptr
  int layer;

  int N;  // number of points to work on
  int N_offset;
};

#endif  // INCLUDE_GGNN_SYM_CUDA_KNN_SYM_QUERY_LAYER_CUH_
