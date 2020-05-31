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
#ifndef CUDA_KNN_SORTED_BUFFER_CACHE_CUH_
#define CUDA_KNN_SORTED_BUFFER_CACHE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename ValueT, typename KeyT, int KQuery, int D, int BLOCK_DIM_X,
          int CACHE_SIZE = 256, int PRIOQ_SIZE = 128, int BEST_SIZE = 32,
          typename BaseT = ValueT, typename BAddrT = KeyT,
          bool DIST_STATS = false, bool OVERFLOW_STATS = false>
struct SortedBufferCache {
  static constexpr KeyT EMPTY_KEY = (KeyT)-1;
  static constexpr ValueT EMPTY_DIST = std::numeric_limits<ValueT>::infinity();

 private:
  static constexpr int CACHE_BUFFER_SIZE = CACHE_SIZE - PRIOQ_SIZE;
  static constexpr int PRIOQ_BUFFER_SIZE = PRIOQ_SIZE - BEST_SIZE;

  static constexpr int CACHE_ITEMS_PER_THREAD =
      (CACHE_SIZE - 1) / BLOCK_DIM_X + 1;
  static constexpr int DIST_ITEMS_PER_THREAD = (D - 1) / BLOCK_DIM_X + 1;
  static constexpr int BEST_ITEMS_PER_THREAD =
      (BEST_SIZE - 1) / BLOCK_DIM_X + 1;

  static constexpr int PRIOQ_ITEMS_PER_THREAD =
      (PRIOQ_SIZE - 1) / BLOCK_DIM_X + 1;

  static constexpr int PRIOQ_BUFFER_ITEMS_PER_THREAD =
      (PRIOQ_BUFFER_SIZE - 1) / BLOCK_DIM_X + 1;

  typedef typename cub::KeyValuePair<KeyT, ValueT> DistPair;

  typedef cub::BlockReduce<ValueT, BLOCK_DIM_X> DistReduce;
  typedef cub::BlockReduce<DistPair, BLOCK_DIM_X> DistPairReduce;
  typedef cub::BlockRadixSort<ValueT, BLOCK_DIM_X, CACHE_ITEMS_PER_THREAD, KeyT>
      BEST_SIZESort;

  union CacheTempStorage {
    typename DistReduce::TempStorage dist_reduce;
    typename DistPairReduce::TempStorage distpair_reduce;
  };

  union SyncTempStorage {
    KeyT cache;
    ValueT dist;
    bool flag;
  };

  union DistBits {
    ValueT dist;
    uint32_t bits;
  };

  static __device__ __forceinline__ void set_worked(DistBits& distB) {
    static constexpr const uint32_t mask = 1;
    distB.bits |= mask;
  }

  static __device__ __forceinline__ void clear_worked(DistBits& distB) {
    static constexpr uint32_t mask = static_cast<uint32_t>(-2);
    distB.bits &= mask;
  }

  static __device__ __forceinline__ bool is_worked(ValueT& dist) {
    return DistBits{dist}.bits & 1;
  }

  struct ArgWorkedDistMin {
    __device__ __forceinline__ DistPair operator()(const DistPair& a,
                                                   const DistPair& b) const {
      if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
      return a;
    }
  };

 public:
  KeyT* s_cache;
  DistBits* s_dists;
  int& s_prioQ_ring;
  int& s_cache_ring;
  int& s_overflow_counter;

  CacheTempStorage& s_storage;
  SyncTempStorage& s_sync;

  ValueT xi;

  const BaseT* d_base;
  BaseT r_query[DIST_ITEMS_PER_THREAD];

  //# threadIdx.x == 0 stats registers only
  int dist_calc_counter;

  __device__ __forceinline__ void initSharedStorage() {
    __shared__ KeyT s_cache_tmp[CACHE_SIZE];
    __shared__ DistBits s_dists_tmp[PRIOQ_SIZE];

    s_cache = reinterpret_cast<KeyT*>(s_cache_tmp);
    s_dists = reinterpret_cast<DistBits*>(s_dists_tmp);
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
    __shared__ int s_prioQ_ring_tmp;
    return s_prioQ_ring_tmp;
  }

  __device__ __forceinline__ int& CacheRingPrivateTmpStorage() {
    __shared__ int s_cache_ring_tmp;
    return s_cache_ring_tmp;
  }

  __device__ __forceinline__ int& OverflowPrivateTmpStorage() {
    __shared__ int s_overflow_tmp;
    return s_overflow_tmp;
  }

  __device__ __forceinline__ void init() {
    for (int i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      s_cache[i] = EMPTY_KEY;
    }
    for (int i = threadIdx.x; i < PRIOQ_SIZE; i += BLOCK_DIM_X) {
      s_dists[i].dist = EMPTY_DIST;
    }
    if (DIST_STATS && !threadIdx.x) dist_calc_counter = 0;
    if (OVERFLOW_STATS && !threadIdx.x) s_overflow_counter = 0;
    if (!threadIdx.x) {
      s_prioQ_ring = 0;
      s_cache_ring = 0;
    }
    __syncthreads();
  }

  __device__ __forceinline__ SortedBufferCache(const BaseT* d_base,
                                               const KeyT n,
                                               const ValueT xi_criteria)
      : s_storage(CachePrivateTmpStorage()),
        d_base(d_base),
        xi(xi_criteria),
        s_prioQ_ring(PrioQRingPrivateTmpStorage()),
        s_cache_ring(CacheRingPrivateTmpStorage()),
        s_overflow_counter(OverflowPrivateTmpStorage()),
        s_sync(SyncPrivateTmpStorage()) {
    initSharedStorage();
    init();
    for (int item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D)
        r_query[item] = d_base[static_cast<BAddrT>(n) * D + read_dim];
    }
  }

  __device__ __forceinline__ SortedBufferCache(const BaseT* d_base,
                                               const BaseT* d_query,
                                               const KeyT n,
                                               const ValueT xi_criteria)
      : s_storage(CachePrivateTmpStorage()),
        d_base(d_base),
        xi(xi_criteria),
        s_prioQ_ring(PrioQRingPrivateTmpStorage()),
        s_cache_ring(CacheRingPrivateTmpStorage()),
        s_overflow_counter(OverflowPrivateTmpStorage()),
        s_sync(SyncPrivateTmpStorage()) {
    initSharedStorage();
    init();
    for (int item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D)
        r_query[item] = d_query[static_cast<BAddrT>(n) * D + read_dim];
    }
  }

  __device__ __forceinline__ SortedBufferCache(const BaseT* d_base,
                                               const ValueT xi_criteria)
      : s_storage(CachePrivateTmpStorage()),
        d_base(d_base),
        xi(xi_criteria),
        s_prioQ_ring(PrioQRingPrivateTmpStorage()),
        s_cache_ring(CacheRingPrivateTmpStorage()),
        s_overflow_counter(OverflowPrivateTmpStorage()),
        s_sync(SyncPrivateTmpStorage()) {
    initSharedStorage();
    init();
  }

  /**
   * Calculates synced distance of base vector to other_id vector.
   *
   * [parallel call]:
   * ValueT dist = cache.distance(other_id)
   *
   * Return:
   * 	ValueT distance
   *
   * Note: distance valid in all threads.
   */
  __device__ __forceinline__ ValueT distance_synced(const KeyT other_id) {
    ValueT r_diff = 0;
    for (int item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        ValueT pos_other = r_query[item] -
                           d_base[static_cast<BAddrT>(other_id) * D + read_dim];
        r_diff += pos_other * pos_other;
      }
    }

    ValueT aggregate = DistReduce(s_storage.dist_reduce).Sum(r_diff);
    if (!threadIdx.x) {
      if (DIST_STATS) dist_calc_counter++;
      s_sync.dist = aggregate;
    }
    __syncthreads();

    return s_sync.dist;
  }

  __device__ __forceinline__ bool criteria(ValueT dist) {
    if (dist < s_dists[KQuery - 1].dist + xi) return true;
    return false;
  }

  __device__ __forceinline__ void push_cache(KeyT key) {
    if (key == EMPTY_KEY) return;
    s_cache[PRIOQ_SIZE + s_cache_ring] = key;
    s_cache_ring = (s_cache_ring + 1);
    if (s_cache_ring >= CACHE_BUFFER_SIZE)
      s_cache_ring = s_cache_ring % CACHE_BUFFER_SIZE;
  }

  __device__ __forceinline__ void insert_front_prioQ(KeyT key, ValueT dist) {
    s_prioQ_ring =
        (s_prioQ_ring > 0) ? s_prioQ_ring - 1 : PRIOQ_BUFFER_SIZE - 1;
    const int pos = BEST_SIZE + s_prioQ_ring;

    if (s_cache[pos] != EMPTY_KEY) {
      if (s_dists[pos].dist == EMPTY_DIST)
        push_cache(s_cache[pos]);
      else {
        if (criteria(s_dists[pos].dist)) {
          s_overflow_counter++;
        }
      }
    }
    s_cache[pos] = key;
    s_dists[pos].dist = dist;
  }

  __device__ __forceinline__ void push2(const KeyT key, const ValueT dist) {
    __syncthreads();
    KeyT r_cache[PRIOQ_ITEMS_PER_THREAD];
    ValueT r_dists[PRIOQ_ITEMS_PER_THREAD];
    if (!threadIdx.x) s_sync.flag = true;
    __syncthreads();
    if (dist < s_dists[BEST_SIZE - 1].dist) {
      //# only move front part
      for (int item = 0; item < BEST_ITEMS_PER_THREAD; ++item) {
        const int i = item * BLOCK_DIM_X + threadIdx.x;
        if (i < BEST_SIZE) {
          r_cache[item] = s_cache[i];
          r_dists[item] = s_dists[i].dist;
          if (r_cache[item] == key) s_sync.flag = false;
        }
      }
      __syncthreads();
      for (int item = 0; item < BEST_ITEMS_PER_THREAD && s_sync.flag; ++item) {
        const int i = item * BLOCK_DIM_X + threadIdx.x;
        if (i < BEST_SIZE) {
          if (r_dists[item] >= dist) {
            if (r_cache[item] != EMPTY_KEY) {
              if (i == BEST_SIZE - 1) {
                if (is_worked(r_dists[item])) {
                  //# push element directly to cache
                  push_cache(r_cache[item]);
                } else {
                  //# set ringbuffer one back and insert
                  //# old element to cache
                  insert_front_prioQ(r_cache[item], r_dists[item]);
                }
              } else {
                s_cache[i + 1] = r_cache[item];
                s_dists[i + 1].dist = r_dists[item];
              }
            }

            if (i == 0 || s_dists[i - 1].dist < dist) {
              s_cache[i] = key;
              s_dists[i].dist = dist;
              clear_worked(s_dists[i]);
            }
          }
        }
      }
    } else {
      //# move ringbuffer around
      for (int item = 0; item < PRIOQ_BUFFER_ITEMS_PER_THREAD; ++item) {
        const int i = item * BLOCK_DIM_X + threadIdx.x;
        if (i < PRIOQ_BUFFER_SIZE) {
          r_cache[item] = s_cache[BEST_SIZE + i];
          r_dists[item] = s_dists[BEST_SIZE + i].dist;
          if (r_cache[item] == key) s_sync.flag = false;
        }
      }
      __syncthreads();
      for (int item = 0; item < PRIOQ_BUFFER_ITEMS_PER_THREAD; ++item) {
        const int i = item * BLOCK_DIM_X + threadIdx.x;
        if (i < PRIOQ_BUFFER_SIZE) {
          const int idx_next = (i == PRIOQ_BUFFER_SIZE - 1) ? 0 : i + 1;
          const int idx_prev = (i == 0) ? PRIOQ_BUFFER_SIZE - 1 : i - 1;
          //# do work if own dist greater or equal
          if (r_dists[item] >= dist) {
            if (r_cache[item] != EMPTY_KEY) {
              if (i == ((s_prioQ_ring == 0) ? PRIOQ_BUFFER_SIZE - 1
                                            : s_prioQ_ring - 1)) {
                if (r_dists[item] != std::numeric_limits<ValueT>::infinity()) {
                } else {
                  push_cache(r_cache[item]);
                }
              } else {
                s_cache[BEST_SIZE + idx_next] = r_cache[item];
                s_dists[BEST_SIZE + idx_next].dist = r_dists[item];
              }
            }

            if (i == s_prioQ_ring ||
                s_dists[BEST_SIZE + idx_prev].dist < dist) {
              s_cache[BEST_SIZE + i] = key;
              s_dists[BEST_SIZE + i].dist = dist;
              clear_worked(s_dists[BEST_SIZE + i]);
            }
          }
        }
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ void fetch(KeyT* s_keys, const KeyT* d_translation,
                                        int len) {
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
    __syncthreads();

    for (int k = 0; k < len; k++) {
      __syncthreads();
      const KeyT other_n = s_keys[k];
      if (other_n == EMPTY_KEY) continue;
      const KeyT other_m =
          (d_translation == nullptr) ? other_n : d_translation[other_n];
      const ValueT dist = distance_synced(other_m);

      if (criteria(dist)) {
        push2(other_n, dist);
        __syncthreads();
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ KeyT
  fetchReturnNearestKey(KeyT* s_keys, const KeyT* d_translation, int len) {
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
    __syncthreads();

    KeyT nearest = EMPTY_KEY;
    ValueT nearestDist;

    for (int k = 0; k < len; k++) {
      __syncthreads();
      const KeyT other_n = s_keys[k];
      if (other_n == EMPTY_KEY) continue;
      const KeyT other_m =
          (d_translation == nullptr) ? other_n : d_translation[other_n];
      const ValueT dist = distance_synced(other_m);

      if (k == 0 || dist < nearestDist) {
        nearest = other_n;
        nearestDist = dist;
      }

      if (criteria(dist)) {
        push2(other_n, dist);
        __syncthreads();
      } else {
      }
    }
    __syncthreads();

    return nearest;
  }

  __device__ __forceinline__ KeyT pop() {
    __syncthreads();
    DistPair pairs[PRIOQ_ITEMS_PER_THREAD];
    for (int item = 0; item < PRIOQ_ITEMS_PER_THREAD; ++item) {
      const int i = item * BLOCK_DIM_X + threadIdx.x;
      if (i < PRIOQ_SIZE && !is_worked(s_dists[i].dist)) {
        pairs[item].value = s_dists[i].dist;
      } else {
        pairs[item].value = std::numeric_limits<ValueT>::infinity();
      }
      pairs[item].key = i;
    }
    DistPair aggregate = DistPairReduce(s_storage.distpair_reduce)
                             .Reduce(pairs, ArgWorkedDistMin());

    if (!threadIdx.x) {
      if (aggregate.value == std::numeric_limits<ValueT>::infinity() ||
          !criteria(aggregate.value))
        s_sync.cache = EMPTY_KEY;
      else {
        s_sync.cache = s_cache[aggregate.key];
        if (aggregate.key >= BEST_SIZE) {
          s_prioQ_ring = (s_prioQ_ring + 1) % PRIOQ_BUFFER_SIZE;
          s_dists[aggregate.key].dist = std::numeric_limits<ValueT>::infinity();
        } else
          set_worked(s_dists[aggregate.key]);
      }
    }
    __syncthreads();
    return s_sync.cache;
  }

  __device__ void clear_worked() {
    for (int i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      if (i < BEST_SIZE) {
        clear_worked(s_dists[i]);
      } else {
        if (i < PRIOQ_SIZE) {
          s_dists[i].dist = EMPTY_DIST;
        }
        s_cache[i] = EMPTY_KEY;
      }
    }
    if (!threadIdx.x) {
      s_prioQ_ring = 0;
      s_cache_ring = 0;
    }
  }

  __device__ __forceinline__ void transform(const KeyT* transform,
                                            const int len = BEST_SIZE) {
    clear_worked();
    for (int i = threadIdx.x; i < len; i += BLOCK_DIM_X) {
      if (s_cache[i] != EMPTY_KEY) s_cache[i] = transform[s_cache[i]];
    }
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
    return sqrtf(s_dists[1].dist);
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
      printf("Cache: ring: %d KQuery: %f (+xi -> %f) \n", s_prioQ_ring,
             s_dists[KQuery - 1].dist, s_dists[KQuery - 1].dist + xi);
      for (int i = 0; i < len; ++i) {
        if (i < BEST_SIZE) {
          printf("%d -> %d %f (%d) \n", i, s_cache[i], s_dists[i].dist,
                 is_worked(s_dists[i].dist));
        } else {
          if (i < PRIOQ_SIZE) {
            printf("%d -> %d %f (%d) | ", i, s_cache[i], s_dists[i].dist,
                   is_worked(s_dists[i].dist));
            if (i - BEST_SIZE == s_prioQ_ring) printf("X");
            printf("\n");
          } else {
            printf("%d -> %d | ", i, s_cache[i]);
            if (i - PRIOQ_SIZE == s_cache_ring) printf("X");
            printf("\n");
          }
        }
      }
    }
    __syncthreads();
  }
};

#endif  // CUDA_KNN_SORTED_BUFFER_CACHE_CUH_
