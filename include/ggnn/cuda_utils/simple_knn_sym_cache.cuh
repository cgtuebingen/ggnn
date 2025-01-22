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

#ifndef INCLUDE_GGNN_SIMPLE_KNN_SYM_CACHE_CUH
#define INCLUDE_GGNN_SIMPLE_KNN_SYM_CACHE_CUH

#include <ggnn/base/def.h>
#include <ggnn/cuda_utils/distance.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/cub.cuh>

namespace ggnn {

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_DIM_X,
          uint32_t DIST_ITEMS_PER_THREAD, bool DIST_STATS = false>
struct SimpleKNNSymCache {
  using AddrT = size_t;

  static constexpr KeyT EMPTY_KEY = static_cast<KeyT>(-1);
  static constexpr ValueT EMPTY_DIST = std::numeric_limits<ValueT>::infinity();
  static constexpr float EPS = 0.1f;

 private:
  const uint32_t D;
  const DistanceMeasure measure;

  struct DistQueryAndHalf {
    ValueT dist_query;
    ValueT dist_half;

    struct Sum {
      __host__ __device__ __forceinline__ DistQueryAndHalf
      operator()(const DistQueryAndHalf& a, const DistQueryAndHalf& b) const
      {
        return {a.dist_query + b.dist_query, a.dist_half + b.dist_half};
      }
    };
  };

  typedef cub::BlockReduce<ValueT, BLOCK_DIM_X> DistReduce;
  typedef cub::BlockReduce<DistQueryAndHalf, BLOCK_DIM_X> DistQueryAndHalfReduce;

  struct CacheTempStorage {
    typename DistReduce::TempStorage dist_reduce;
    typename DistQueryAndHalfReduce::TempStorage dist_query_half_reduce;
  };

 public:
  const uint32_t BEST_SIZE;
  const uint32_t SORTED_SIZE;
  const uint32_t CACHE_SIZE;

  KeyT* s_cache;
  ValueT* s_dists;
  uint32_t r_prioQ_head;
  uint32_t r0_visited_head;

  CacheTempStorage& s_storage;
  bool& s_sync;
  DistQueryAndHalf& s_dist;

  ValueT r_criteria_half;
  ValueT r_xi;

  const BaseT* d_base;
  BaseT r_query[DIST_ITEMS_PER_THREAD];
  ValueT r_half[DIST_ITEMS_PER_THREAD];

  // only valid in thread 0
  ValueT r0_query_norm;
  ValueT r0_half_norm;

  // # threadIdx.x == 0 stats registers only
  uint32_t dist_calc_counter;

  __device__ __forceinline__ void initSharedStorage()
  {
    extern __shared__ KeyT shared_cache[];

    s_cache = shared_cache;
    s_dists = reinterpret_cast<ValueT*>(&s_cache[CACHE_SIZE]);
  }

  __device__ __forceinline__ CacheTempStorage& CachePrivateTmpStorage()
  {
    __shared__ CacheTempStorage cache_tmp_storage;
    return cache_tmp_storage;
  }

  __device__ __forceinline__ bool& SyncPrivateTmpStorage()
  {
    __shared__ bool s_sync_tmp;
    return s_sync_tmp;
  }

  __device__ __forceinline__ DistQueryAndHalf& DistTmpStorage()
  {
    __shared__ DistQueryAndHalf s_dist;
    return s_dist;
  }

  __device__ __forceinline__ SimpleKNNSymCache(const uint32_t D, const DistanceMeasure measure,
                                               const uint32_t BEST_SIZE, const uint32_t SORTED_SIZE,
                                               const uint32_t CACHE_SIZE, const BaseT* d_base,
                                               const KeyT n, const ValueT xi_criteria)
      : D(D),
        measure(measure),
        BEST_SIZE{BEST_SIZE},
        SORTED_SIZE{SORTED_SIZE},
        CACHE_SIZE{CACHE_SIZE},
        s_storage(CachePrivateTmpStorage()),
        d_base(d_base),
        r_xi(xi_criteria),
        s_sync(SyncPrivateTmpStorage()),
        s_dist(DistTmpStorage())
  {
    initSharedStorage();
    // init(); // will be initialized later with init_start_point
    if constexpr (DIST_STATS)
      if (!threadIdx.x)
        dist_calc_counter = 0;
    loadQueryPos(d_base + static_cast<AddrT>(n) * D);
  }

  __device__ __forceinline__ void loadQueryPos(const BaseT* d_query)
  {
    ValueT query_norm = 0.0f;
    for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
      r_query[item] = (read_dim < D) ? d_query[read_dim] : 0;
      if (measure == DistanceMeasure::Cosine)
        query_norm += static_cast<ValueT>(r_query[item]) * static_cast<ValueT>(r_query[item]);
    }
    if (measure == DistanceMeasure::Cosine) {
      // only needed by thread 0
      r0_query_norm = DistReduce(s_storage.dist_reduce).Sum(query_norm);
      __syncthreads();
    }
  }

  __device__ __forceinline__ void init_start_point(const KeyT other_n, const KeyT* d_translation)
  {
    const KeyT other_m = (d_translation == nullptr) ? other_n : d_translation[other_n];
    DistQueryAndHalf norms{0.0f, 0.0f};
    for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
      r_half[item] = (read_dim < D) ? d_base[static_cast<AddrT>(other_m) * D + read_dim] : 0;
    }
    for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        r_half[item] = static_cast<ValueT>(r_query[item]) +
                       (0.5f - EPS) * (static_cast<ValueT>(r_half[item]) - r_query[item]);
        if (measure == DistanceMeasure::Cosine) {
          norms.dist_query += static_cast<ValueT>(r_query[item]) * r_query[item];
          norms.dist_half += r_half[item] * r_half[item];
        }
      }
    }
    __syncthreads();
    if (measure == DistanceMeasure::Cosine) {
      DistQueryAndHalf norms_sum = DistQueryAndHalfReduce(s_storage.dist_query_half_reduce)
                                       .Reduce(norms, DistQueryAndHalf::Sum());
      if (!threadIdx.x) {
        r0_query_norm = norms_sum.dist_query;
        r0_half_norm = norms_sum.dist_half;
      }
      __syncthreads();
    }
    const DistQueryAndHalf dists = distance_synced(other_m);
    r_criteria_half = dists.dist_half + r_xi;

    // clear cache and add start point to best list and prioQ
    for (uint32_t i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      s_cache[i] = (i == 0 || i == BEST_SIZE) ? other_n : EMPTY_KEY;
      if (i < SORTED_SIZE)
        s_dists[i] = (i == 0 || i == BEST_SIZE) ? dists.dist_query : EMPTY_DIST;
    }
    r_prioQ_head = BEST_SIZE;
    if (!threadIdx.x)
      r0_visited_head = SORTED_SIZE;
    __syncthreads();
  }

  /**
   * Calculates synced distance of base vector to other_id vector.
   *
   * [parallel call]:
   * ValueT dist = cache.distance(other_id)
   *
   * Return:
   *   ValueT distance
   *
   * Note: distance valid in all threads.
   */
  __device__ __forceinline__ DistQueryAndHalf distance_synced(const KeyT other_id)
  {
    BaseT r_other[DIST_ITEMS_PER_THREAD];

    for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
      r_other[item] = (read_dim < D) ? d_base[static_cast<AddrT>(other_id) * D + read_dim] : 0;
    }

    DistQueryAndHalf dist{0.f, 0.f};
    ValueT norm_other = 0.0f;
    if (measure == DistanceMeasure::Euclidean) {
      for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
        const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
        if (read_dim < D) {
          const ValueT dist_query =
              static_cast<ValueT>(r_query[item]) - static_cast<ValueT>(r_other[item]);
          dist.dist_query += dist_query * dist_query;
          const ValueT dist_half =
              static_cast<ValueT>(r_half[item]) - static_cast<ValueT>(r_other[item]);
          dist.dist_half += dist_half * dist_half;
        }
      }
    }
    else if (measure == DistanceMeasure::Cosine) {
      for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
        const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
        if (read_dim < D) {
          const ValueT dist_query =
              static_cast<ValueT>(r_query[item]) * static_cast<ValueT>(r_other[item]);
          dist.dist_query += dist_query;
          const ValueT dist_half =
              static_cast<ValueT>(r_half[item]) * static_cast<ValueT>(r_other[item]);
          dist.dist_half += dist_half;
          norm_other += static_cast<ValueT>(r_other[item]) * static_cast<ValueT>(r_other[item]);
        }
      }
    }

    dist = DistQueryAndHalfReduce(s_storage.dist_query_half_reduce)
               .Reduce(dist, DistQueryAndHalf::Sum());
    if (measure == DistanceMeasure::Cosine) {
      __syncthreads();
      // need to normalize by the vectors' lengths (in high dimensions, no
      // vector has length 1.0f)
      norm_other = DistReduce(s_storage.dist_reduce).Sum(norm_other);
      if (!threadIdx.x) {
        const ValueT query_norm_sqr = norm_other * r0_query_norm;
        const ValueT half_norm_sqr = norm_other * r0_half_norm;
        // use negative dot product, as larger values are closer to each other
        // otherwise, we would need to adjust each and every distance comparison
        // in the code
        dist.dist_query =
            (query_norm_sqr > 0.0f) ? fabs(1.0f - dist.dist_query / sqrtf(query_norm_sqr)) : 1.0f;
        // while this could be computed in parallel to the query distance,
        // the necessary shuffling and synchronization costs more.
        dist.dist_half =
            (half_norm_sqr > 0.0f) ? fabs(1.0f - dist.dist_half / sqrtf(half_norm_sqr)) : 1.0f;
      }
    }

    if (!threadIdx.x) {
      if constexpr (DIST_STATS)
        dist_calc_counter++;
      s_dist = dist;
    }
    __syncthreads();

    return s_dist;
  }

  __device__ __forceinline__ ValueT criteria_sym()
  {
    return s_dists[0] + r_xi;
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
    if (key == EMPTY_KEY || dist >= criteria_sym())
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
  __device__ __forceinline__ void fetch(KeyT* s_keys, const KeyT* d_translation, uint32_t len)
  {
    if constexpr (filter_known_keys) {
      __syncthreads();
      for (uint32_t i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
        const KeyT n = s_cache[i];
        if (n != EMPTY_KEY) {
          for (uint32_t k = 0; k < len; ++k) {
            if (s_keys[k] == n)
              s_keys[k] = EMPTY_KEY;
          }
        }
      }
    }

    __syncthreads();

    for (uint32_t k = 0; k < len; ++k) {
      const KeyT other_n = s_keys[k];
      if (other_n == EMPTY_KEY)
        continue;

      const KeyT other_m = (d_translation == nullptr) ? other_n : d_translation[other_n];
      const DistQueryAndHalf dist = distance_synced(other_m);

      if (dist.dist_query < criteria_sym() && dist.dist_half < r_criteria_half)
        push(other_n, dist.dist_query);
    }

    __syncthreads();
  }

  __device__ __forceinline__ void write_best(KeyT* d_buffer, const KeyT n, uint32_t stride)
  {
    for (uint32_t i = threadIdx.x; i < BEST_SIZE; i += BLOCK_DIM_X) {
      const KeyT idx = s_cache[i];
      d_buffer[n * stride + i] = idx;
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

#endif  // INCLUDE_GGNN_SIMPLE_KNN_SYM_CACHE_CUH
