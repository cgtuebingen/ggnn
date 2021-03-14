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

#ifndef INCLUDE_GGNN_CACHE_CUDA_SIMPLE_KNN_SYM_CACHE_CUH_
#define INCLUDE_GGNN_CACHE_CUDA_SIMPLE_KNN_SYM_CACHE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/utils/cuda_knn_utils.cuh"

template <DistanceMeasure measure,
          typename ValueT, typename KeyT, int KQuery, int D, int BLOCK_DIM_X,
          int VISITED_SIZE = 256, int PRIOQ_SIZE = 128, int BEST_SIZE = 32,
          typename BaseT = ValueT, typename BAddrT = KeyT,
          bool DIST_STATS = false, bool OVERFLOW_STATS = false>
struct SimpleKNNSymCache {
  static constexpr KeyT EMPTY_KEY = (KeyT)-1;
  static constexpr ValueT EMPTY_DIST = std::numeric_limits<ValueT>::infinity();

  // TODO(fabi): change to constant?
  static constexpr float EPS = 0.1f;

 private:
  static constexpr int CACHE_SIZE = BEST_SIZE + PRIOQ_SIZE + VISITED_SIZE;
  static constexpr int SORTED_SIZE = BEST_SIZE + PRIOQ_SIZE;

  static constexpr int DIST_ITEMS_PER_THREAD = (D - 1) / BLOCK_DIM_X + 1;
  static constexpr int BEST_ITEMS_PER_THREAD =
      (BEST_SIZE - 1) / BLOCK_DIM_X + 1;
  static constexpr int PRIOQ_ITEMS_PER_THREAD =
      (PRIOQ_SIZE - 1) / BLOCK_DIM_X + 1;

  static constexpr int CACHE_ITEMS_PER_THREAD =
      (CACHE_SIZE - 1) / BLOCK_DIM_X + 1;
  static constexpr int SORTED_ITEMS_PER_THREAD =
      (SORTED_SIZE - 1) / BLOCK_DIM_X + 1;

  static constexpr int BEST_END = BEST_SIZE - 1;

  struct DistQueryAndHalf {
    ValueT dist_query;
    ValueT dist_half;

    __device__ __forceinline__ DistQueryAndHalf(const ValueT dist_query,
                                                const ValueT dist_half)
        : dist_query(dist_query), dist_half(dist_half) {}

    __device__ __forceinline__ DistQueryAndHalf() {}
  };

  struct DistanceAndNorm {
    ValueT r_dist;
    ValueT r_norm;

    __device__ __forceinline__ DistanceAndNorm(const ValueT dist,
                                               const ValueT norm)
        : r_dist(dist), r_norm(norm) {}

    __device__ __forceinline__ DistanceAndNorm() {}

    struct Sum {
      __host__ __device__ __forceinline__ DistanceAndNorm
      operator()(const DistanceAndNorm& a, const DistanceAndNorm& b) const {
        return DistanceAndNorm(a.r_dist + b.r_dist, a.r_norm + b.r_norm);
      }
    };
  };

  typedef cub::BlockReduce<ValueT, BLOCK_DIM_X> DistReduce;
  typedef cub::BlockReduce<DistQueryAndHalf, BLOCK_DIM_X>
      DistQueryAndHalfReduce;

  union CacheTempStorage {
    struct {
      typename DistReduce::TempStorage dist_reduce;
      typename DistQueryAndHalfReduce::TempStorage dist_query_half_reduce;
    };
  };

  union SyncTempStorage {
    KeyT cache;
    DistQueryAndHalf dist;
    bool flag;

    __device__ __forceinline__ SyncTempStorage() {}
  };

 public:
  KeyT* s_cache;
  ValueT* s_dists;
  int& s_prioQ_head;
  int& s_visited_head;
  int& s_overflow_counter;

  CacheTempStorage& s_storage;
  SyncTempStorage& s_sync;

  ValueT criteria_dist;
  ValueT xi;

  const BaseT* d_base;
  BaseT r_query[DIST_ITEMS_PER_THREAD];
  ValueT r_half[DIST_ITEMS_PER_THREAD];

  // only valid in thread 0
  ValueT query_norm;
  ValueT half_norm;

  //# threadIdx.x == 0 stats registers only
  int dist_calc_counter;

  __device__ __forceinline__ void initSharedStorage() {
    __shared__ KeyT s_cache_tmp[CACHE_SIZE];
    __shared__ ValueT s_dists_tmp[SORTED_SIZE];

    s_cache = reinterpret_cast<KeyT*>(s_cache_tmp);
    s_dists = reinterpret_cast<ValueT*>(s_dists_tmp);
  }

  __device__ __forceinline__ CacheTempStorage& CachePrivateTmpStorage() {
    __shared__ CacheTempStorage cache_tmp_storage;
    return cache_tmp_storage;
  }

  __device__ __forceinline__ SyncTempStorage& SyncPrivateTmpStorage() {
    __shared__ SyncTempStorage s_sync_tmp;
    return s_sync_tmp;
  }

  __device__ __forceinline__ int& PrioQRingPrivateTmpStorage() {
    __shared__ int s_prioQ_head_tmp;
    return s_prioQ_head_tmp;
  }

  __device__ __forceinline__ int& CacheRingPrivateTmpStorage() {
    __shared__ int s_visited_head_tmp;
    return s_visited_head_tmp;
  }

  __device__ __forceinline__ int& OverflowPrivateTmpStorage() {
    __shared__ int s_overflow_tmp;
    return s_overflow_tmp;
  }

  __device__ __forceinline__ void init() {
    for (int i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      s_cache[i] = EMPTY_KEY;
    }
    for (int i = threadIdx.x; i < SORTED_SIZE; i += BLOCK_DIM_X) {
      s_dists[i] = EMPTY_DIST;
    }
    if (DIST_STATS && !threadIdx.x) dist_calc_counter = 0;
    if (OVERFLOW_STATS && !threadIdx.x) s_overflow_counter = 0;
    if (!threadIdx.x) {
      s_prioQ_head = 0;
      s_visited_head = 0;
    }
    __syncthreads();
  }

  __device__ __forceinline__ SimpleKNNSymCache(const BaseT* d_base,
                                               const KeyT n,
                                               const ValueT xi_criteria)
      : s_storage(CachePrivateTmpStorage()),
        d_base(d_base),
        xi(xi_criteria),
        s_prioQ_head(PrioQRingPrivateTmpStorage()),
        s_visited_head(CacheRingPrivateTmpStorage()),
        s_overflow_counter(OverflowPrivateTmpStorage()),
        s_sync(SyncPrivateTmpStorage()) {
    initSharedStorage();
    init();
    loadQueryPos(d_base + static_cast<BAddrT>(n) * D);
  }

  __device__ __forceinline__ void loadQueryPos(const BaseT* d_query) {
    ValueT r_query_norm = 0.0f;
    for (int item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        r_query[item] = *(d_query + read_dim);
        if (measure == Cosine) r_query_norm += r_query[item] * r_query[item];
      }
    }
    if (measure == Cosine) {
      // only needed by thread 0
      query_norm = DistReduce(s_storage.dist_reduce).Sum(r_query_norm);
    }
  }

  __device__ __forceinline__ void init_start_point(const KeyT other_n,
                                                   const KeyT* d_translation) {
    init();
    const KeyT s =
        (d_translation == nullptr) ? other_n : d_translation[other_n];
    DistQueryAndHalf r_norms(0.0f, 0.0f);
    for (int item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        r_half[item] =
            r_query[item] +
            (0.5f - EPS) * ((d_base[static_cast<BAddrT>(s) * D + read_dim] -
                             r_query[item]));
        if (measure == Cosine) {
          r_norms.dist_query += r_query[item] * r_query[item];
          r_norms.dist_half += r_half[item] * r_half[item];
        }
      }
    }
    __syncthreads();
    if (measure == Cosine) {
      DistQueryAndHalf norms =
          DistQueryAndHalfReduce(s_storage.dist_query_half_reduce)
              .Reduce(r_norms, DistSum());
      if (!threadIdx.x) {
        query_norm = norms.dist_query;
        half_norm = norms.dist_half;
      }
    }
    const DistQueryAndHalf dists = distance_synced(other_n);
    criteria_dist = dists.dist_half + xi;
    if (!threadIdx.x) {
      // Add start point to best list...
      s_cache[0] = other_n;
      s_dists[0] = dists.dist_query;
      // ... and and prioQ.
      s_cache[BEST_SIZE] = other_n;
      s_dists[BEST_SIZE] = dists.dist_query;
    }
  }

  struct DistSum {
    __host__ __device__ __forceinline__ DistQueryAndHalf
    operator()(const DistQueryAndHalf& a, const DistQueryAndHalf& b) const {
      return DistQueryAndHalf(a.dist_query + b.dist_query,
                              a.dist_half + b.dist_half);
    }
  };

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
  __device__ __forceinline__ DistQueryAndHalf
  distance_synced(const KeyT other_id) {
    DistQueryAndHalf r_diff(0.f, 0.f);
    ValueT r_norm_other = 0.0f;
    for (int item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        const ValueT p = d_base[static_cast<BAddrT>(other_id) * D + read_dim];
        if (measure == Euclidean) {
          const ValueT dist_query = r_query[item] - p;
          r_diff.dist_query += dist_query * dist_query;
          const ValueT dist_half = r_half[item] - p;
          r_diff.dist_half += dist_half * dist_half;
        } else if (measure == Cosine) {
          const ValueT dist_query = r_query[item] * p;
          r_diff.dist_query += dist_query;
          const ValueT dist_half = r_half[item] * p;
          r_diff.dist_half += dist_half;
          r_norm_other += p * p;
        }
      }
    }

    DistQueryAndHalf aggregate =
        DistQueryAndHalfReduce(s_storage.dist_query_half_reduce)
            .Reduce(r_diff, DistSum());
    if (measure == Cosine) {
      // need to normalize by the vectors' lengths (in high dimensions, no
      // vector has length 1.0f)
      const ValueT norm_other =
          DistReduce(s_storage.dist_reduce).Sum(r_norm_other);
      const ValueT query_norm_sqr = norm_other * query_norm;
      const ValueT half_norm_sqr = norm_other * half_norm;
      // use negative dot product, as larger values are closer to each other
      // otherwise, we would need to adjust each and every distance comparison
      // in the code
      if (!threadIdx.x) {
        if (query_norm_sqr > 0.0f)
          aggregate.dist_query =
              fabs(1.0f - aggregate.dist_query / sqrt(query_norm_sqr));
        else
          aggregate.dist_query = 1.0f;
        // while this could be computed in parallel to the query distance,
        // the necessary shuffling and synchronization costs more.
        if (half_norm_sqr > 0.0f)
          aggregate.dist_half =
              fabs(1.0f - aggregate.dist_half / sqrt(half_norm_sqr));
        else
          aggregate.dist_half = 1.0f;
      }
    }
    if (!threadIdx.x) {
      if (DIST_STATS) dist_calc_counter++;
      s_sync.dist = aggregate;
    }
    __syncthreads();

    return s_sync.dist;
  }

  __device__ __forceinline__ bool criteria(const ValueT dist) {
    return (dist < (s_dists[0] + xi));
  }

  __device__ __forceinline__ bool criteria(const DistQueryAndHalf& dist) {
    return ((dist.dist_query < (s_dists[0] + xi)) &&
            (dist.dist_half < criteria_dist));
  }

  __device__ __forceinline__ bool is_end(int tid) {
    const int prev_prioQ_ring =
        (s_prioQ_head - 1 < 0) ? PRIOQ_SIZE - 1 : s_prioQ_head - 1;
    return tid == BEST_END || tid == BEST_SIZE + prev_prioQ_ring;
  }

  __device__ __forceinline__ void push(const KeyT key, const ValueT dist) {
    __syncthreads();
    // Register for insertion in best and prioq
    KeyT r_cache[SORTED_ITEMS_PER_THREAD];
    ValueT r_dists[SORTED_ITEMS_PER_THREAD];

    int r_write_item_best = -1;
    int r_write_item_prioQ = -1;
    if (!threadIdx.x) s_sync.flag = true;
    __syncthreads();

    // Load items for insertion.
    for (int item = 0; item < SORTED_ITEMS_PER_THREAD && s_sync.flag; ++item) {
      const int idx = item * BLOCK_DIM_X + threadIdx.x;
      if (idx < SORTED_SIZE) {
        r_cache[item] = s_cache[idx];
        r_dists[item] = s_dists[idx];
        if (r_cache[item] == key) s_sync.flag = false;
      }
    }
    __syncthreads();
    // TODO(fabi) return on s_sync.flag = true?
    for (int item = 0; item < SORTED_ITEMS_PER_THREAD && s_sync.flag; ++item) {
      const int idx = item * BLOCK_DIM_X + threadIdx.x;
      if (idx < SORTED_SIZE) {
        if (r_dists[item] >= dist) {
          // Don't move if no entry or end of best or prioq.
          if ((r_cache[item] != EMPTY_KEY) && !is_end(idx)) {
            const int idx_next = (idx + 1 == SORTED_SIZE) ? BEST_SIZE : idx + 1;
            s_cache[idx_next] = r_cache[item];
            s_dists[idx_next] = r_dists[item];
          }

          // Find insert points.
          const int idx_prev = idx - 1;
          const ValueT dist_prev =
              ((idx_prev == -1) || (idx_prev == BEST_SIZE + s_prioQ_head - 1))
                  ? -1.f
                  : (idx_prev == BEST_END) ? s_dists[SORTED_SIZE - 1]
                                           : s_dists[idx_prev];
          if (dist_prev < dist) {
            if (idx < BEST_SIZE)
              r_write_item_best = item;
            else
              r_write_item_prioQ = item;
          }
        }
      }
    }
    __syncthreads();

    // Insert into best and prioq.
    if (r_write_item_best >= 0) {
      const int idx = r_write_item_best * BLOCK_DIM_X + threadIdx.x;
      s_cache[idx] = key;
      s_dists[idx] = dist;
    }
    if (r_write_item_prioQ >= 0) {
      const int idx = r_write_item_prioQ * BLOCK_DIM_X + threadIdx.x;
      s_cache[idx] = key;
      s_dists[idx] = dist;
    }
  }

  __device__ __forceinline__ KeyT pop() {
    __syncthreads();

    if (!threadIdx.x) {
      const int head_idx_prioQ = BEST_SIZE + s_prioQ_head;
      const ValueT dist = s_dists[head_idx_prioQ];
      if (dist == EMPTY_DIST) {
        // Pop on empty prioQ.
        s_sync.cache = EMPTY_KEY;
      } else {
        if (!criteria(dist)) {
          s_sync.cache = EMPTY_KEY;
        } else {
          const KeyT key = s_cache[head_idx_prioQ];
          s_sync.cache = key;
          const int head_idx_visited = SORTED_SIZE + s_visited_head;
          s_cache[head_idx_visited] = key;
          s_visited_head = (s_visited_head + 1) % VISITED_SIZE;
        }
        s_cache[head_idx_prioQ] = EMPTY_KEY;
        s_dists[head_idx_prioQ] = EMPTY_DIST;
        // Move ring-buffer head forward.
        s_prioQ_head = (s_prioQ_head + 1) % PRIOQ_SIZE;
      }
    }
    __syncthreads();
    return s_sync.cache;
  }

  __device__ __forceinline__ void fetch(KeyT* s_keys, const KeyT* d_translation,
                                        int len, bool debug = false) {
    __syncthreads();
    for (int item = 0; item < CACHE_ITEMS_PER_THREAD; ++item) {
      const int i = item * BLOCK_DIM_X + threadIdx.x;
      if (i < CACHE_SIZE) {
        const KeyT n = s_cache[i];
        for (int k = 0; n != EMPTY_KEY && k < len; k++) {
          if (n == s_keys[k]) {
            s_keys[k] = EMPTY_KEY;
          }
        }
      }
    }

    for (int k = 0; k < len; k++) {
      __syncthreads();
      const KeyT other_n = s_keys[k];
      if (other_n == EMPTY_KEY) continue;
      const KeyT other_m =
          (d_translation == nullptr) ? other_n : d_translation[other_n];
      const DistQueryAndHalf dist = distance_synced(other_m);
      if (criteria(dist)) {
        push(other_n, dist.dist_query);
        __syncthreads();
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ void write_best_graph(KeyT* d_buffer, const KeyT n,
                                                   int K, int offset = 1) {
    for (int i = threadIdx.x; i < K; i += BLOCK_DIM_X) {
      const KeyT idx = s_cache[i + offset];
      d_buffer[n * K + i] = (idx != EMPTY_KEY) ? idx : n;
    }
  }

  __device__ __forceinline__ void write_best(KeyT* d_buffer, const KeyT n,
                                             int stride) {
    for (int i = threadIdx.x; i < KQuery; i += BLOCK_DIM_X) {
      const KeyT idx = s_cache[i];
      d_buffer[n * stride + i] = idx;
    }
  }

  __device__ __forceinline__ float get_nn1_dist() {
    if (measure == Euclidean) {
      return sqrtf(s_dists[1]);
    } else if (measure == Cosine) {
      return s_dists[1];
    }
    // TODO(fabi): restructure or error.
    return 0;
  }

  __device__ __forceinline__ int get_dist_stats() { return dist_calc_counter; }
  __device__ __forceinline__ int get_overflow_stats() {
    return s_overflow_counter;
  }

  /**
   * Prints first 'len' elements in the Cache. [parallel call]:
   * cash.print(8);
   *
   */
  __device__ __forceinline__ void print(int len = CACHE_SIZE) {
    __syncthreads();
    if (!threadIdx.x) printf("print \n");
    if (!threadIdx.x) {
      printf("Cache: ring: %d KQuery: %f (+xi -> %f) \n", s_prioQ_head,
             s_dists[KQuery - 1], s_dists[KQuery - 1] + xi);
      for (int i = 0; i < len; ++i) {
        if (i < BEST_SIZE) {
          printf("%d -> %d %f \n", i, s_cache[i], s_dists[i]);
        } else {
          if (i < SORTED_SIZE) {
            printf("%d -> %d %f | ", i, s_cache[i], s_dists[i]);
            if (i - BEST_SIZE == s_prioQ_head) printf("X");
            printf("\n");
          } else {
            printf("%d -> %d | ", i, s_cache[i]);
            if (i - SORTED_SIZE == s_visited_head) printf("X");
            printf("\n");
          }
        }
      }
    }
    __syncthreads();
  }
};

#endif  // INCLUDE_GGNN_CACHE_CUDA_SIMPLE_KNN_SYM_CACHE_CUH_
