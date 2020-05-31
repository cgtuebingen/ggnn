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
#ifndef CUDA_KNN_K_BEST_LIST_CUH_
#define CUDA_KNN_K_BEST_LIST_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/utils/cuda_knn_core_utils.cuh"

/**
 * KBestList stores the K best elements in parallel.
 */
template <typename ValueT, typename KeyT, int K>
struct KBestList {
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
    if (threadIdx.x < K) {
      dists[threadIdx.x] = std::numeric_limits<ValueT>::infinity();
      ids[threadIdx.x] = EMPTY_KEY;
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
    ValueT r_dist;
    KeyT r_id;
    if (threadIdx.x < K) {
      r_dist = dists[threadIdx.x];
      r_id = ids[threadIdx.x];
      if (r_id == id) s_enter = false;
    }
    __syncthreads();
    if (threadIdx.x < K && s_enter) {
      if (r_dist > dist) {
        if (threadIdx.x < (K - 1)) {
          dists[threadIdx.x + 1] = r_dist;
          ids[threadIdx.x + 1] = r_id;
        }

        if (!threadIdx.x || dists[threadIdx.x - 1] <= dist) {
          dists[threadIdx.x] = dist;
          ids[threadIdx.x] = id;
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
    ValueT r_dist;
    KeyT r_id;
    if (threadIdx.x < K) {
      r_dist = dists[threadIdx.x];
      r_id = ids[threadIdx.x];
      if (r_id == id) s_enter = false;
    }
    __syncthreads();
    if (threadIdx.x < K && s_enter) {
      if (r_dist >= dist) {
        if (threadIdx.x < (K - 1)) {
          dists[threadIdx.x + 1] = r_dist;
          ids[threadIdx.x + 1] = r_id;
        }

        if (!threadIdx.x || dists[threadIdx.x - 1] < dist) {
          //          printf("enter: %f %d -> %d \n", dist, id, threadIdx.x);
          dists[threadIdx.x] = dist;
          ids[threadIdx.x] = id;
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
    ValueT r_dist;
    KeyT r_id;
    if (threadIdx.x < K) {
      r_dist = dists[threadIdx.x];
      r_id = ids[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x < K) {
      if (r_dist > dist) {
        if (threadIdx.x < (K - 1)) {
          dists[threadIdx.x + 1] = r_dist;
          ids[threadIdx.x + 1] = r_id;
        }

        if (!threadIdx.x || dists[threadIdx.x - 1] <= dist) {
          dists[threadIdx.x] = dist;
          ids[threadIdx.x] = unique_id;
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
    if (threadIdx.x < K) {
      const KeyT id = ids[threadIdx.x];
      if (id >= 0) ids[threadIdx.x] = transform[id];
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

#endif  // CUDA_KNN_K_BEST_LIST_CUH_
