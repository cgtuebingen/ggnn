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

#ifndef INCLUDE_GGNN_SIMPLE_KNN_CACHE_CUH
#define INCLUDE_GGNN_SIMPLE_KNN_CACHE_CUH

#include <ggnn/base/def.h>
#include <ggnn/cuda_utils/distance.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ggnn {

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_DIM_X,
          uint32_t DIST_ITEMS_PER_THREAD, bool DIST_STATS = false>
struct SimpleKNNCache {
  static constexpr KeyT EMPTY_KEY = static_cast<KeyT>(-1);
  static constexpr ValueT EMPTY_DIST = std::numeric_limits<ValueT>::infinity();

 private:
  using Distance = ggnn::Distance<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;

 public:
  const uint32_t BEST_SIZE;
  const uint32_t SORTED_SIZE;
  const uint32_t CACHE_SIZE;

  KeyT* s_cache;
  ValueT* s_dists;
  uint32_t r_prioQ_head;
  uint32_t r0_visited_head;

  bool& s_sync;
  Distance rs_dist_calc;

  ValueT r_xi;

  // # threadIdx.x == 0 stats registers only
  uint32_t dist_calc_counter;

  __device__ __forceinline__ void initSharedStorage()
  {
    extern __shared__ KeyT shared_cache[];

    s_cache = shared_cache;
    s_dists = reinterpret_cast<ValueT*>(
        &s_cache[CACHE_SIZE]);  // cacheSize = numIds in the cache, after that the dists start
  }

  __device__ __forceinline__ bool& SyncPrivateTmpStorage()
  {
    __shared__ bool s_sync_tmp;
    return s_sync_tmp;
  }

  __device__ __forceinline__ void init()
  {
    for (uint32_t i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      s_cache[i] = EMPTY_KEY;
      if (i < SORTED_SIZE)
        s_dists[i] = EMPTY_DIST;
    }
    r_prioQ_head = BEST_SIZE;
    if (!threadIdx.x) {
      if constexpr (DIST_STATS)
        dist_calc_counter = 0;
      r0_visited_head = SORTED_SIZE;
    }
    __syncthreads();
  }

  __device__ __forceinline__ SimpleKNNCache(const uint32_t D, const DistanceMeasure measure,
                                            const uint32_t BEST_SIZE, const uint32_t SORTED_SIZE,
                                            const uint32_t CACHE_SIZE, const BaseT* d_base,
                                            const KeyT n, const ValueT xi_criteria)
      : BEST_SIZE{BEST_SIZE},
        SORTED_SIZE{SORTED_SIZE},
        CACHE_SIZE(CACHE_SIZE),
        s_sync(SyncPrivateTmpStorage()),
        rs_dist_calc(D, measure, d_base, n),
        r_xi(xi_criteria)
  {
    initSharedStorage();
    init();
  }

  // dieses hier:
  __device__ __forceinline__ SimpleKNNCache(const uint32_t D, const DistanceMeasure measure,
                                            const uint32_t BEST_SIZE, const uint32_t SORTED_SIZE,
                                            const uint32_t CACHE_SIZE, const BaseT* d_base,
                                            const BaseT* d_query, const KeyT n,
                                            const ValueT xi_criteria)
      : BEST_SIZE{BEST_SIZE},
        SORTED_SIZE{SORTED_SIZE},
        CACHE_SIZE(CACHE_SIZE),
        s_sync(SyncPrivateTmpStorage()),
        rs_dist_calc(D, measure, d_base, d_query, n),
        r_xi(xi_criteria)
  {
    initSharedStorage();
    init();
  }

  __device__ __forceinline__ ValueT criteria() const
  {
    return s_dists[BEST_SIZE - 1] + r_xi;
  }

  __device__ __forceinline__ void push(const KeyT key, const ValueT dist)
  {
    __syncthreads();
    // Register for insertion in best and prioq

    // check for duplicates
    {
      if (!threadIdx.x)
        s_sync = false;

      __syncthreads();

      for (uint32_t idx = threadIdx.x; idx < SORTED_SIZE && !s_sync; idx += BLOCK_DIM_X) {
        if (s_cache[idx] == key)
          s_sync = true;
      }

      __syncthreads();
      if (s_sync)
        return;
    }

    const uint32_t head_idx_prioQ = r_prioQ_head;
    const uint32_t head_idx_in_prioQ = head_idx_prioQ - BEST_SIZE;

    // process blocks from right to left (we shift to the right)
    {
      KeyT r_cache;
      ValueT r_dists;

      uint32_t idx;
      bool active = false;

      // start with the last block
      uint32_t block_start = ((SORTED_SIZE + BLOCK_DIM_X - 1) / BLOCK_DIM_X) * BLOCK_DIM_X;

      while (true) {
        // shift
        if (active) {
          // Don't move if no entry or end of best or prioq.
          if (r_cache != EMPTY_KEY) {
            const uint32_t idx_next = (idx + 1 == SORTED_SIZE) ? BEST_SIZE : idx + 1;
            const bool has_next = idx_next != BEST_SIZE && idx_next != head_idx_prioQ;
            if (has_next) {
              s_cache[idx_next] = r_cache;
              s_dists[idx_next] = r_dists;
            }
          }

          // Find insert points.
          const bool has_prev = idx != 0 && idx != head_idx_prioQ;
          const uint32_t idx_prev = idx != BEST_SIZE ? idx - 1 : SORTED_SIZE - 1;
          if (!has_prev || s_dists[idx_prev] < dist) {
            // insert into best list and priority queue
            s_cache[idx] = key;
            s_dists[idx] = dist;
          }
        }

        if (!block_start)
          break;

        // update index
        block_start -= BLOCK_DIM_X;
        idx = block_start + threadIdx.x;
        active = idx < SORTED_SIZE;

        // read
        if (active) {
          // handle ringbuffer addresses
          // TODO: reorder between threads to fix bank conflicts
          if (idx >= BEST_SIZE) {
            idx = (idx + head_idx_in_prioQ < SORTED_SIZE)
                      ? idx + head_idx_in_prioQ
                      : idx + head_idx_in_prioQ - SORTED_SIZE + BEST_SIZE;
          }

          r_cache = s_cache[idx];
          r_dists = s_dists[idx];

          // shift all elements with larger/equal distance to the right
          active &= r_dists >= dist;
        }

        __syncthreads();
      }
    }
  }

  __device__ __forceinline__ KeyT pop()
  {
    __syncthreads();
    const uint32_t head_idx_prioQ = r_prioQ_head;
    const KeyT key = s_cache[head_idx_prioQ];
    // Pop on empty prioQ.
    const ValueT dist = s_dists[head_idx_prioQ];
    __syncthreads();
    if (key == EMPTY_KEY || dist >= criteria())
      return EMPTY_KEY;

    if (!threadIdx.x) {
      // update visited list
      const uint32_t head_idx_visited = r0_visited_head;
      s_cache[head_idx_visited] = key;
      r0_visited_head = (head_idx_visited + 1) >= CACHE_SIZE ? SORTED_SIZE : head_idx_visited + 1;
      // remove from prioQ
      s_cache[head_idx_prioQ] = EMPTY_KEY;
      s_dists[head_idx_prioQ] = EMPTY_DIST;
    }
    // Move ring-buffer head forward.
    r_prioQ_head = (head_idx_prioQ + 1) >= SORTED_SIZE ? BEST_SIZE : head_idx_prioQ + 1;
    __syncthreads();
    return key;
  }

  template <bool filter_known_keys = true>
  __device__ __forceinline__ void fetch(
      std::conditional_t<filter_known_keys, KeyT, const KeyT>* s_keys, const KeyT* d_translation,
      uint32_t len)
  {
    if constexpr (filter_known_keys) {
      __syncthreads();
      // filter known indices in the cache
      for (uint32_t i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
        const KeyT n = s_cache[i];
        if (n == EMPTY_KEY) {
          if (i >= SORTED_SIZE)
            break;
          continue;
        }
        for (uint32_t k = 0; k < len; ++k) {
          if (s_keys[k] == n)
            s_keys[k] = EMPTY_KEY;
        }
      }
    }

    __syncthreads();

    KeyT n_cache;
    uint32_t mask = 0;

    for (uint32_t k = 0; k < len || mask;) {
      if (!mask) {
        uint32_t idx = k + (threadIdx.x % 32);  // assuming block dim x is a multiple of 32
        n_cache = idx < len ? s_keys[idx] : EMPTY_KEY;
        mask = __ballot_sync(0xffffffff, n_cache != EMPTY_KEY);
        k += 32;
        continue;
      }

      const uint32_t first = __ffs(mask) - 1;
      mask ^= 1 << first;
      const KeyT other_n = __shfl_sync(0xffffffff, n_cache, first);

      const KeyT other_m = (d_translation) ? d_translation[other_n] : other_n;
      const ValueT dist = rs_dist_calc.distance_synced(other_m);

      if (dist < criteria())
        push(other_n, dist);
    }

    __syncthreads();
  }

  __device__ __forceinline__ void fetch_unfiltered(const KeyT* s_keys, const KeyT* d_translation,
                                                   const uint32_t len)
  {
    fetch<false>(s_keys, d_translation, len);
  }

  __device__ __forceinline__ void transform(const KeyT* transform)
  {
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      if (i < BEST_SIZE) {
        // transform best
        KeyT key = s_cache[i];
        if (key != EMPTY_KEY)
          key = transform[key];
        s_cache[i] = key;

        // copy best into prio queue
        if (i + BEST_SIZE < SORTED_SIZE) {
          s_cache[i + BEST_SIZE] = key;
          s_dists[i + BEST_SIZE] = s_dists[i];
        }
      }
      else if (i < 2 * BEST_SIZE && i < SORTED_SIZE) {
        // do nothing (handled by previous threads)
      }
      else {
        // reset remainder of the prio queue and visited cache
        s_cache[i] = EMPTY_KEY;
        if (i < SORTED_SIZE)
          s_dists[i] = EMPTY_DIST;
      }
    }

    // reset heads.
    r_prioQ_head = BEST_SIZE;
    if (!threadIdx.x) {
      r0_visited_head = SORTED_SIZE;
    }

    __syncthreads();
  }

  __device__ __forceinline__ void write_best(KeyT* d_buffer, const KeyT n, uint32_t stride)
  {
#pragma unroll
    for (uint32_t i = threadIdx.x; i < BEST_SIZE; i += BLOCK_DIM_X) {
      const KeyT idx = s_cache[i];
      d_buffer[static_cast<size_t>(n) * stride + i] = idx;
    }
  }

  __device__ __forceinline__ void write_best(KeyT* d_buffer, const KeyT n, uint32_t stride,
                                             uint32_t idx_offset)
  {
#pragma unroll
    for (uint32_t i = threadIdx.x; i < BEST_SIZE; i += BLOCK_DIM_X) {
      const KeyT idx = s_cache[i];
      d_buffer[static_cast<size_t>(n) * stride + i] = idx + idx_offset;
    }
  }

  __device__ __forceinline__ uint32_t get_dist_stats()
  {
    return dist_calc_counter;
  }

  /**
   * Prints first 'len' elements in the Cache. [parallel call]:
   * cash.print(8);
   *
   */
  __device__ __forceinline__ void print(uint32_t len = -1U)
  {
    if (len == -1U)
      len = CACHE_SIZE;
    __syncthreads();
    if (!threadIdx.x)
      printf("print \n");
    if (!threadIdx.x) {
      printf("Cache: ring: %d BEST_SIZE: %f (+xi -> %f) \n", r_prioQ_head, s_dists[BEST_SIZE - 1],
             s_dists[BEST_SIZE - 1] + r_xi);
      for (uint32_t i = 0; i < len; ++i) {
        if (i < BEST_SIZE) {
          printf("%d -> %d %f \n", i, s_cache[i], s_dists[i]);
        }
        else {
          if (i < SORTED_SIZE) {
            printf("%d -> %d %f | ", i, s_cache[i], s_dists[i]);
            if (i == r_prioQ_head)
              printf("X");
            printf("\n");
          }
          else {
            printf("%d -> %d | ", i, s_cache[i]);
            if (i == r0_visited_head)
              printf("X");
            printf("\n");
          }
        }
      }
    }
    __syncthreads();
  }
};

};  // namespace ggnn

// for checking for warnings with clangd - will be instantiated implicitly on demand
// template struct SimpleKNNCache<int32_t, float, float, Euclidean, 128, 10, 256, 32, false>;

#endif  // INCLUDE_GGNN_SIMPLE_KNN_CACHE_CUH
