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
// Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch
// converted to GGNN library by: Lukas Ruppert, Deborah Kornwolf

#ifndef INCLUDE_GGNN_K_BEST_LIST_CUH
#define INCLUDE_GGNN_K_BEST_LIST_CUH

#include <cstdint>
#include <limits>

namespace ggnn {

/**
 * KBestList stores the K best elements in parallel.
 */
template <typename KeyT, typename ValueT, uint32_t BLOCK_DIM_X>
struct KBestList {
  const uint32_t BEST_SIZE;

  ValueT* s_dists;
  KeyT* s_ids;

  static constexpr KeyT EMPTY_KEY = -1;

  __device__ __forceinline__ void initSharedStorage(uint32_t BEST_SIZE)
  {
    extern __shared__ ValueT shared_kBestList[];
    s_dists = shared_kBestList;
    s_ids = reinterpret_cast<KeyT*>(&s_dists[BEST_SIZE]);
  }

  __device__ __forceinline__ void init()
  {
    for (uint32_t i = 0; i < BEST_SIZE; i += BLOCK_DIM_X) {
      const uint32_t k = i + threadIdx.x;
      if (k < BEST_SIZE) {
        s_dists[k] = std::numeric_limits<ValueT>::infinity();
        s_ids[k] = EMPTY_KEY;
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ KBestList(uint32_t BEST_SIZE) : BEST_SIZE(BEST_SIZE)
  {
    initSharedStorage(BEST_SIZE);
    init();
  }

  __device__ __forceinline__ ValueT worst()
  {
    return s_dists[BEST_SIZE - 1];
  }

  /**
   * Enters element with dist and id to list. [parallel call]:
   * On same distances the entry is placed to the left.
   *
   * `list.add_unique(dist, id)`
   *
   * Note: __syncthreads() need before next 'list' call.
   *
   */
  __device__ __forceinline__ void add_unique(ValueT dist, KeyT id)
  {
    // process blocks from right to left (we shift to the right)
    for (uint32_t i = ((BEST_SIZE - 1) / BLOCK_DIM_X) * BLOCK_DIM_X;; i -= BLOCK_DIM_X) {
      const uint32_t k = i + threadIdx.x;
      ValueT r_dist;
      KeyT r_id;
      // read current value
      if (k < BEST_SIZE) {
        r_dist = s_dists[k];
        r_id = s_ids[k];
      }
      __syncthreads();
      if (k < BEST_SIZE) {
        // shift and enter new point if new distance is smalller
        if (dist < r_dist) {
          // shift current value to next position
          if (k < (BEST_SIZE - 1)) {
            s_dists[k + 1] = r_dist;
            s_ids[k + 1] = r_id;
          }

          // enter new point if left index is smaller and will not shift into my position
          if (!k || s_dists[k - 1] <= dist) {
            s_dists[k] = dist;
            s_ids[k] = id;
          }
        }
      }
      if (!i)
        break;
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
  __device__ __forceinline__ void transform(const KeyT* transform)
  {
    for (int i = 0; i < BEST_SIZE; i += BLOCK_DIM_X) {
      const uint32_t k = i + threadIdx.x;
      if (k < BEST_SIZE) {
        const KeyT id = s_ids[k];
        if (id != EMPTY_KEY)
          s_ids[k] = transform[id];
      }
    }
  }

  __device__ __forceinline__ void print(int len = -1)
  {
    __syncthreads();
    if (!threadIdx.x) {
      printf("KBestList: \n");
      for (int i = 0; i < BEST_SIZE && (len < 0 || i < len); i++) {
        printf("(%d -> %f [%d]) ", i, s_dists[i], s_ids[i]);
      }
      printf("\n");
    }
  }
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_K_BEST_LIST_CUH
