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

#include <ggnn/construction/sym_query_layer.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/lib.h>

#include <ggnn/cuda_utils/simple_knn_sym_cache.cuh>

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

namespace ggnn {

template <typename T>
__global__ void __launch_bounds__(T::BLOCK_DIM_X) sym(const T kernel)
{
  kernel();
}

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
__device__ __forceinline__ void
SymQueryKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>::operator()() const
{
  static constexpr uint32_t K_BLOCK = 32;
  static_assert(K_BLOCK <= BLOCK_DIM_X);

  static constexpr bool DIST_STATS = false;

  const uint32_t KF{KBuild / 2};
  const uint32_t KL = KBuild - KF;

  using Cache =
      SimpleKNNSymCache<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD, DIST_STATS>;

  const float xi = (measure == DistanceMeasure::Euclidean)
                       ? (d_nn1_stats[0] * d_nn1_stats[0]) * tau_build * tau_build
                       : d_nn1_stats[0] * tau_build;

  const KeyT n = static_cast<KeyT>(blockIdx.x);

  Cache cache(D, measure, KF, sorted_size, CACHE_SIZE, d_base, d_translation ? d_translation[n] : n,
              xi);

  __shared__ bool s_connected;

  // fetch neighbors in local neighbor list
  for (uint32_t i = 0; i < KL; i += K_BLOCK) {
    __shared__ KeyT s_sym_ids[K_BLOCK];
    {
      const uint32_t kl = i + threadIdx.x;
      if (threadIdx.x < K_BLOCK && kl < KL) {
        const KeyT sym_n = d_graph[static_cast<size_t>(n) * KBuild + kl];
        s_sym_ids[threadIdx.x] = sym_n;
      }
    }

    for (uint32_t k = 0; i + k < KL && k < K_BLOCK; k++) {
      __syncthreads();
      if (!threadIdx.x)
        s_connected = false;

      // search for k-th local neighbor
      cache.init_start_point(s_sym_ids[k], d_translation);

      bool found_connection = false;

      for (uint32_t ite = 0; ite < MAX_PER_PATH_ITERATIONS && !found_connection; ++ite) {
        __syncthreads();

        const KeyT anchor = cache.pop();

        if (anchor == Cache::EMPTY_KEY) {
          break;
        }

        // fetch neighbors at anchor point + points in sym buffer
        __shared__ KeyT s_knn[K_BLOCK];
        for (uint32_t i = 0; i < KBuild; i += K_BLOCK) {
          if (threadIdx.x < K_BLOCK) {
            const uint32_t k = i + threadIdx.x;
            if (k < KBuild) {
              const KeyT other_id = (k < KL)
                                        ? d_graph[static_cast<size_t>(anchor) * KBuild + k]
                                        : d_sym_buffer[static_cast<size_t>(anchor) * KF + k - KL];
              if (other_id == n) {
                s_connected = true;
              }
              s_knn[threadIdx.x] = other_id;
            }
            else {
              s_knn[threadIdx.x] = Cache::EMPTY_KEY;
            }
          }
          __syncthreads();
          if (s_connected) {
            // stop if the original index n has been found as a neighbor
            found_connection = true;
            break;
          }
          cache.fetch(s_knn, d_translation, K_BLOCK);
        }
      }  // end per k iteration

      if (!found_connection) {
        // we need to add a symmetric link to the original index n
        if (!threadIdx.x) {
          for (uint32_t i = 0; i < KF; i++) {
            // try to enter the symmetric link at the i-th nearest neighbor
            // found on the path
            const KeyT other_n = cache.s_cache[i];
            if (other_n == Cache::EMPTY_KEY)
              break;
            const uint32_t pos = atomicAdd(&d_sym_atomic[other_n], 1U);
            if (pos < KF) {
              d_sym_buffer[static_cast<size_t>(other_n) * KF + pos] = n;
              break;
            }
          }
          // could not add a link
        }
      }
    }  // end k neighbors
    __syncthreads();
  }
}

#define GGNN_SYM(KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD)       \
  template __global__ void                                                     \
  sym<SymQueryKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>>( \
      const SymQueryKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>);

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_SYMS, GGNN_SYM);

};  // namespace ggnn
