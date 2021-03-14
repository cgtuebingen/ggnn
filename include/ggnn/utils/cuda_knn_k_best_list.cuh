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
// Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch

#ifndef INCLUDE_GGNN_UTILS_CUDA_KNN_K_BEST_LIST_CUH_
#define INCLUDE_GGNN_UTILS_CUDA_KNN_K_BEST_LIST_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

/**
 * KBestList stores the K best elements in parallel.
 */
template <typename ValueT, typename KeyT, int K, int BLOCK_DIM_X>
struct KBestList {
  // this allows for loop unrolling
  static constexpr int ITERATIONS_FOR_K = (K+BLOCK_DIM_X-1)/BLOCK_DIM_X;

  ValueT* dists;
  KeyT* ids;

  static constexpr KeyT EMPTY_KEY = -1;

  __device__ __forceinline__ void initSharedStorage() {
    __shared__ ValueT s_dists[K];
    __shared__ KeyT s_ids[K];
    dists = reinterpret_cast<ValueT*>(s_dists);
    ids = reinterpret_cast<KeyT*>(s_ids);
  }

  __device__ __forceinline__ void init() {
    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        dists[k] = std::numeric_limits<ValueT>::infinity();
        ids[k] = EMPTY_KEY;
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ KBestList() {
    initSharedStorage();
    init();
  }

  __device__ __forceinline__ ValueT worst() { return dists[K - 1]; }

  /**
   * Enters element with dist and id to list. [parallel call]:
   * Only enters the object if the id is not already in the list.
   * On same distances the entry is placed to the right.
   *
   * `list.add(dist, id)`
   *
   * Note: __syncthreads() need before next 'list' call.
   *
   */
  __device__ __forceinline__ void add(ValueT dist, KeyT id) {
    __shared__ bool s_enter;
    if (!threadIdx.x) s_enter = true;
    __syncthreads();
    ValueT r_dist[ITERATIONS_FOR_K];
    KeyT r_id[ITERATIONS_FOR_K];
    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        r_dist[i] = dists[threadIdx.x];
        r_id[i] = ids[threadIdx.x];
        if (r_id[i] == id) s_enter = false;
      }
    }
    __syncthreads();
    if (!s_enter)
      return;
    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        if (r_dist[i] > dist) {
          if (k < (K - 1)) {
            dists[k + 1] = r_dist[i];
            ids[k + 1] = r_id[i];
          }

          if (!k || dists[k - 1] <= dist) {
            dists[k] = dist;
            ids[k] = id;
          }
        }
      }
    }
  }

  /**
   * Enters element with dist and id to list. [parallel call]:
   * Only enters the object if the id is not already in the list.
   * On same distances the entry is placed to the left.
   *
   * `list.add_priority(dist, id)`
   *
   * Note: __syncthreads() need before next 'list' call.
   *
   */
  __device__ __forceinline__ void add_priority(ValueT dist, KeyT id) {
    __shared__ bool s_enter;
    if (!threadIdx.x) s_enter = true;
    __syncthreads();
    ValueT r_dist[ITERATIONS_FOR_K];
    KeyT r_id[ITERATIONS_FOR_K];
    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        r_dist[i] = dists[threadIdx.x];
        r_id[i] = ids[threadIdx.x];
        if (r_id[i] == id) s_enter = false;
      }
    }
    __syncthreads();
    if (!s_enter)
      return;

    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        if (r_dist[i] >= dist) {
          if (k < (K - 1)) {
            dists[k + 1] = r_dist[i];
            ids[k + 1] = r_id[i];
          }

          if (!k || dists[k - 1] < dist) {
            dists[k] = dist;
            ids[k] = id;
          }
        }
      }
    }
  }

  /**
   * Enters a (assumed to be) unique element with dist and id to list. [parallel
   * call]:
   * There is no check if the element is already in the list.
   *
   * `list.add_unique(dist, unique_id)`
   *
   * Note: __syncthreads() need before next 'list' call.
   *
   */
  __device__ __forceinline__ void add_unique(ValueT dist, KeyT unique_id) {
    ValueT r_dist[ITERATIONS_FOR_K];
    KeyT r_id[ITERATIONS_FOR_K];
    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        r_dist[i] = dists[k];
        r_id[i] = ids[k];
      }
    }
    __syncthreads();
    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        if (r_dist[i] > dist) {
          if (k < (K - 1)) {
            dists[k + 1] = r_dist[i];
            ids[k + 1] = r_id[i];
          }

          if (!k || dists[k - 1] <= dist) {
            dists[k] = dist;
            ids[k] = unique_id;
          }
        }
      }
    }
  }

  /**
   * Transforms all ids w.r.t. a transformation list. [parallel call]:
   *
   * `list.transform(transform_list)`
   *
   * Note: __syncthreads() need before next 'list' call.
   *
   */
  __device__ __forceinline__ void transform(const KeyT* transform) {
    for (int i=0; i < ITERATIONS_FOR_K; ++i) {
      const int k = i*BLOCK_DIM_X+threadIdx.x;
      if (k < K) {
        const KeyT id = ids[k];
        if (id >= 0) ids[k] = transform[id];
      }
    }
  }

  __device__ __forceinline__ void print(int len = -1) {
    __syncthreads();
    if (!threadIdx.x) {
      printf("KBestList: \n");
      for (int i = 0; i < K && (len < 0 || i < len); i++) {
        printf("(%d -> %f [%d]) ", i, dists[i], ids[i]);
      }
      printf("\n");
    }
  }
};

#endif  // INCLUDE_GGNN_UTILS_CUDA_KNN_K_BEST_LIST_CUH_
