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
#ifndef CUDA_KNN_MULTI_WORKED_DISTS_CACHE_CUH_
#define CUDA_KNN_MULTI_WORKED_DISTS_CACHE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename KeyT, typename ValueT, int BLOCK_DIM_X, int HASHMAP_BIT,
          int MAX_PROBE = 2>
struct MultiWorkedDistsCache {
  //% assumptions:
  //% MAX_PROBE <= 32

  union DistBits {
    ValueT dist;
    uint32_t bits;
  };

  static constexpr int HASHMAP_SIZE = get_power<2, HASHMAP_BIT>::value;
  static constexpr ValueT EMPTY = -23.f;
  static constexpr KeyT EMPTY_KEY = (KeyT)-1;
  static constexpr KeyT BUSY_KEY = (KeyT)-3;
  static constexpr DistBits EMPTY_DISTB = {EMPTY};

  static constexpr int SEGEMENTS_PER_WARP = 32 / MAX_PROBE;
  static constexpr int NUM_WARPS = BLOCK_DIM_X / 32;
  static constexpr int KFLAG = -1;

  static constexpr unsigned int FULL_MASK = 0xffffffffU;

  KeyT* hash;
  DistBits* values;

  __device__ __forceinline__ void initSharedStorage() {
    __shared__ KeyT s_hash[HASHMAP_SIZE];
    __shared__ DistBits s_values[HASHMAP_SIZE];
    hash = reinterpret_cast<KeyT*>(s_hash);
    values = reinterpret_cast<DistBits*>(s_values);
  }

  __device__ __forceinline__ void init() {
    for (int i = threadIdx.x; i < HASHMAP_SIZE; i += BLOCK_DIM_X) {
      hash[i] = EMPTY_KEY;
    }
    __syncthreads();
  }

  __device__ __forceinline__ MultiWorkedDistsCache() {
    initSharedStorage();
    init();
  }

  __device__ __forceinline__ int hash_fun(const KeyT key, const int p) {
    const uint32_t knuth = 2654435769;
    const uint32_t x = key;

    uint32_t hash =
        ((1 + p) * (x + 1) * knuth + 42 + ((1 << p) << (32 - HASHMAP_BIT))) >>
        (32 - HASHMAP_BIT);

    return hash;
  }

  struct WorkedDist {
    const ValueT dist;
    const bool worked;

    __device__ __forceinline__ WorkedDist() : dist(EMPTY), worked(false){};

    __device__ __forceinline__ WorkedDist(const DistBits distB)
        : dist(distB.dist), worked(distB.bits & 1){};
  };

  static __device__ __forceinline__ void set_worked(DistBits& distB) {
    static constexpr const uint32_t mask = 1;
    distB.bits |= mask;
  }

  static __device__ __forceinline__ void clear_worked(DistBits& distB) {
    static constexpr uint32_t mask = static_cast<uint32_t>(-2);
    distB.bits &= mask;
  }

  static __device__ __forceinline__ bool is_worked(const ValueT& dist) {
    return DistBits{dist}.bits & 1;
  }

  typedef typename cub::KeyValuePair<KeyT, DistBits> Pair;
  struct ArgWorkedDistMax {
    __device__ __forceinline__ Pair operator()(const Pair& a,
                                               const Pair& b) const {
      if (a.value.dist == EMPTY) return a;
      if (b.value.dist == EMPTY) return b;

      const bool workedA = a.value.bits & 1;
      const bool workedB = b.value.bits & 1;
      if (workedA == workedB) {
        if (b.value.dist > a.value.dist) return b;
        return a;
      }
      if (workedA) return b;
      return a;
    }
  };

  /**
   * Checks if key is already cashed and returns value.
   *
   * [parallel call]:
   * if(threadIdx.x < N)
   * 	cache.check(id)
   *
   * Note: __syncthreads() need before next 'cache' call.
   *
   * Return:
   * 	Returns value if key is already in cache, otherwise -1
   *
   */
  __device__ __forceinline__ WorkedDist check(const KeyT key) {
    for (int p = 0; p < MAX_PROBE; ++p) {
      const int hash_idx = hash_fun(key, p);
      const KeyT hash_key = hash[hash_idx];

      // hash still free
      if (hash_key == EMPTY_KEY) return WorkedDist();

      // id already in hash
      if (hash_key == key) {
        return WorkedDist(values[hash_idx]);
      }
    }
    return WorkedDist();
  }

  __device__ __forceinline__ void check(const KeyT* keys, ValueT* dists,
                                        const int K) {
    static constexpr bool DBG_MODE = 0;

    int num_iterations = ((K - 1) / (SEGEMENTS_PER_WARP * NUM_WARPS)) + 1;
    if (DBG_MODE)
      if (!threadIdx.x) printf("num_iterations: %d | %d \n", num_iterations, K);

    for (int ite = 0; ite < num_iterations; ite++) {
      const int warpId = ite * NUM_WARPS + threadIdx.x / 32;

      //# remove all warps that are not needed anymore
      if (warpId * SEGEMENTS_PER_WARP >= K) return;

      const int laneId = cub::LaneId();
      const int k_warp = warpId * SEGEMENTS_PER_WARP + laneId / MAX_PROBE;
      const int k = (laneId < SEGEMENTS_PER_WARP * MAX_PROBE && k_warp < K)
                        ? k_warp
                        : KFLAG;
      const int p = (laneId % MAX_PROBE);

      const KeyT key = (k != KFLAG) ? keys[k] : EMPTY_KEY;
      const int hash_idx = (k != KFLAG) ? hash_fun(key, p) : -1;
      const KeyT hash_key = (k != KFLAG) ? hash[hash_idx] : EMPTY_KEY;

      const bool flag = (k != KFLAG) ? key == hash_key : false;

      __syncwarp();
      const uint32_t mask = __ballot_sync(FULL_MASK, flag);
      const uint32_t mask_k = ((1 << MAX_PROBE) - 1)
                              << (laneId / MAX_PROBE) * MAX_PROBE;
      const uint32_t leader = __ffs(mask & mask_k);
      const uint32_t self = threadIdx.x + 1 - (threadIdx.x / 32) * 32;

      if (k != KFLAG) {
        if (leader == self) {
          dists[k] = values[hash_idx].dist;
          set_worked(values[hash_idx]);
        } else if (!leader && !p) {
          dists[k] = EMPTY;
        }
      }

      if (DBG_MODE && 0) {
        printf(
            "[%d] k: %d | p: %d | key: %d | (hash) idx: %d key: %d -> "
            "dist: %f dist_worked: %u worked: %u => found and updated? (%d)\n",
            threadIdx.x, k, p, (k != KFLAG) ? keys[k] : -1, hash_idx, hash_key,
            dists[k], is_worked(dists[k]), values[hash_idx].bits & 1,
            leader == self);
      }

      if (DBG_MODE) {
        if (leader == self && k != KFLAG)
          printf(
              "<cached> [%d] k: %d | p: %d | key: %d | (hash) idx: %d key: %d "
              "-> "
              "dist: %f dist_worked: %u worked: %u => found and updated? "
              "(%d)\n",
              threadIdx.x, k, p, (k != KFLAG) ? keys[k] : -1, hash_idx,
              hash_key, dists[k], is_worked(dists[k]),
              values[hash_idx].bits & 1, leader == self);
      }
    }
  }

  // A stateful callback functor that maintains a running prefix to be applied
  // during consecutive scan operations.
  struct BlockPrefixCallbackOp {
    // Running prefix
    int running_total;
    // Constructor
    __device__ BlockPrefixCallbackOp(int running_total)
        : running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the
    // block. Thread-0 is responsible for returning a value for seeding the
    // block-wide scan.
    __device__ int operator()(int block_aggregate) {
      int old_prefix = running_total;
      running_total += block_aggregate;
      return old_prefix;
    }
  };

  __device__ __forceinline__ int check_reduce(KeyT* keys, ValueT* dists,
                                              KeyT* ids, const int K) {
    static constexpr bool DBG_MODE = 0;

    typedef cub::BlockScan<int, BLOCK_DIM_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockPrefixCallbackOp prefix_op(0);

    int num_iterations = ((K - 1) / (SEGEMENTS_PER_WARP * NUM_WARPS)) + 1;
    if (DBG_MODE)
      if (!threadIdx.x) printf("num_iterations: %d \n", num_iterations);

    for (int ite = 0; ite < num_iterations; ite++) {
      const int warpId = ite * NUM_WARPS + threadIdx.x / 32;
      const int laneId = cub::LaneId();
      const int k_warp = warpId * SEGEMENTS_PER_WARP + laneId / MAX_PROBE;
      const int k = (laneId < SEGEMENTS_PER_WARP * MAX_PROBE && k_warp < K)
                        ? k_warp
                        : KFLAG;
      const int p = (laneId % MAX_PROBE);

      const KeyT key = (k != KFLAG) ? keys[k] : -1;
      const KeyT id = (k != KFLAG) ? ids[k] : -1;
      const int hash_idx = (k != KFLAG) ? hash_fun(key, p) : -1;
      const KeyT hash_key = (k != KFLAG) ? hash[hash_idx] : -1;

      const bool flag = key == hash_key;

      __syncwarp();
      const uint32_t mask = __ballot_sync(FULL_MASK, flag);
      const uint32_t mask_k = ((1 << MAX_PROBE) - 1)
                              << (laneId / MAX_PROBE) * MAX_PROBE;
      const uint32_t leader = __ffs(mask & mask_k);
      const uint32_t self = threadIdx.x + 1 - (threadIdx.x / 32) * 32;

      const WorkedDist hash_wdist =
          (leader == self) ? WorkedDist(values[hash_idx]) : WorkedDist();

      if (leader == self) set_worked(values[hash_idx]);

      if (DBG_MODE && 0)
        printf(
            "[%d] k: %d | p: %d | key: %d | (hash) idx: %d key: %d -> "
            "dist: %f worked : % d\n ",
            threadIdx.x, k, p, (k != KFLAG) ? keys[k] : -1, hash_idx, hash_key,
            hash_wdist.dist, hash_wdist.worked);

      int head_flag = (leader == self && !hash_wdist.worked) ||
                      (!leader && !p && k != KFLAG);
      int thread_data = head_flag;

      if (DBG_MODE) {
        printf("%d -> %d %d | %d %d (%d) -> %d  \n", threadIdx.x, mask, mask_k,
               leader, self, head_flag, thread_data);
      }

      BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, prefix_op);
      // cub::CTA_SYNC();

      if (DBG_MODE) {
        __syncthreads();

        if (head_flag)
          printf(
              "<leader!> [%d | %d ] k: %d | p: %d | key: %d | (hash) idx: %d "
              "key: %d "
              "-> dits: %f "
              " worked: %d || mask %d %d => %d | ffs: %d self: %d ===> [%d]\n",
              threadIdx.x, ite, k, p, (k != KFLAG) ? keys[k] : -1, hash_idx,
              hash_key, hash_wdist.dist, hash_wdist.worked, mask, mask_k,
              __popc(mask & mask_k), leader, self, thread_data);
      }

      __syncthreads();
      if (head_flag) {
        keys[thread_data] = key;
        dists[thread_data] = hash_wdist.dist;
        ids[thread_data] = id;
      }
    }
    __syncthreads();
    __shared__ int Kp;
    if (!threadIdx.x) Kp = prefix_op.running_total;
    __syncthreads();
    return Kp;
  }

  //! TODO Doc
  // /**
  //  * Updates the cache with the key value pair
  //  *
  //  * [parallel call]:

  //  *
  //  * Note: __syncthreads() need before next 'cache' call.
  //  *
  //  * Return:
  //  * 	returns if entry was succesfull or not
  //  *
  //  */

  static constexpr int BUSY_LANE_KEY = -4;
  typedef unsigned long long int ull_t;
  __device__ __forceinline__ KeyT atomicCAS_helper(uint32_t* hash_ptr,
                                                   uint32_t head_key) {
    return atomicCAS(hash_ptr, head_key, BUSY_KEY);
  }

  __device__ __forceinline__ KeyT atomicCAS_helper(uint64_t* hash_ptr,
                                                   uint64_t head_key) {
    return atomicCAS((unsigned long long int*)(hash_ptr),
                     (unsigned long long int)(head_key), BUSY_KEY);
  }

  __device__ __forceinline__ KeyT atomicCAS_helper(int* hash_ptr,
                                                   int head_key) {
    return atomicCAS(hash_ptr, head_key, BUSY_KEY);
  }

  template <bool DBG_MODE = false>
  __device__ __forceinline__ void push(const KeyT* keys, const ValueT* dists,
                                       const int Kp) {
    typedef cub::WarpReduce<Pair> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    int num_iterations = ((Kp - 1) / (SEGEMENTS_PER_WARP * NUM_WARPS)) + 1;
    if (DBG_MODE)
      if (!threadIdx.x)
        printf("num_iterations: %d | Kp: %d \n", num_iterations, Kp);

    __shared__ int s_busy_lane[SEGEMENTS_PER_WARP * NUM_WARPS * MAX_PROBE];

    for (int ite = 0; ite < num_iterations; ite++) {
      const int warpId = ite * NUM_WARPS + threadIdx.x / 32;

      if (threadIdx.x < SEGEMENTS_PER_WARP * NUM_WARPS * MAX_PROBE)
        s_busy_lane[threadIdx.x] = -1;

      __syncthreads();

      //# remove all warps that are not needed anymore
      if (warpId * SEGEMENTS_PER_WARP >= Kp) return;

      const int laneId = cub::LaneId();
      const int k_warp = warpId * SEGEMENTS_PER_WARP + laneId / MAX_PROBE;
      const int k = (laneId < SEGEMENTS_PER_WARP * MAX_PROBE && k_warp < Kp)
                        ? k_warp
                        : KFLAG;
      const int p = (laneId % MAX_PROBE);
      const int head_flag = (p == 0) || (k < 0);

      const int hash_idx = (k != KFLAG) ? hash_fun(keys[k], p) : -1;
      const KeyT hash_key = (k != KFLAG) ? hash[hash_idx] : EMPTY_KEY;

      const int busy_lane_id =
          ((threadIdx.x / 32) * SEGEMENTS_PER_WARP + (laneId / MAX_PROBE)) *
              MAX_PROBE +
          p;

      if (k != KFLAG &&
          (busy_lane_id < 0 ||
           busy_lane_id >= SEGEMENTS_PER_WARP * NUM_WARPS * MAX_PROBE))
        printf("BIGO PROBLEMO 123 !!! \n");

      //# in case we try to put in a key that is already in the hash.
      //# Value gets updated with new dist!
      //# that case should be avoided!
      if ((k != KFLAG) && keys[k] == hash_key) {
        if (DBG_MODE && (hash_idx < 0 || hash_idx >= HASHMAP_SIZE))
          printf("PROBLEMO hash_idx: %d \n", hash_idx);
        if (DBG_MODE) printf("enter on idx %d -> %f \n", hash_idx, dists[k]);

        s_busy_lane[busy_lane_id] = BUSY_LANE_KEY - 1;
        values[hash_idx].dist = dists[k];
        set_worked(values[hash_idx]);
      }

      Pair data;
      data.key = hash_idx;
      data.value = (k != KFLAG) ? values[hash_idx] : EMPTY_DISTB;

      for (int inner = 0; inner < MAX_PROBE / 2 + 1; inner++) {
        Pair aggregate =
            WarpReduce(temp_storage)
                .HeadSegmentedReduce(data, head_flag, ArgWorkedDistMax());

        bool entry = true;

        if (head_flag && k != KFLAG &&
            s_busy_lane[busy_lane_id] > BUSY_LANE_KEY) {
          if (DBG_MODE)
            printf("<head> [%d | %d] k: %d key: %d => %d %f ||| %d \n",
                   threadIdx.x, inner, k, keys[k], aggregate.key,
                   aggregate.value.dist, __activemask());

          const int head_idx = aggregate.key;
          const KeyT head_key = hash[head_idx];

          if ((!std::is_unsigned<KeyT>::value && head_idx < 0) ||
              head_idx >= HASHMAP_SIZE)
            printf("PROBLEMO head_idx: %d \n", head_idx);

          if ((head_key != BUSY_KEY) &&
              atomicCAS_helper(&hash[head_idx], head_key) == head_key) {
            if (DBG_MODE)
              printf("[%d] entered %d %f on idx %d \n", k, keys[k], dists[k],
                     head_idx);

            values[head_idx].dist = dists[k];
            set_worked(values[head_idx]);

            hash[head_idx] = keys[k];
            s_busy_lane[busy_lane_id] = BUSY_LANE_KEY;
          } else {
            if (DBG_MODE)
              printf("[%d | %d] hash was busy!! \n", head_idx, keys[k]);
            if (DBG_MODE && (busy_lane_id < 0 ||
                             busy_lane_id >= SEGEMENTS_PER_WARP * NUM_WARPS))
              printf("Problem 441 \n");
            s_busy_lane[busy_lane_id] = head_idx;
            entry = false;
          }
        }

        if (__all_sync(FULL_MASK, entry) > 0) break;

        if (DBG_MODE && k != KFLAG &&
            (busy_lane_id < 0 ||
             busy_lane_id >= SEGEMENTS_PER_WARP * NUM_WARPS))
          printf("Problem 449: %d \n", busy_lane_id);

        if (k != KFLAG && s_busy_lane[busy_lane_id] == data.key) {
          if (DBG_MODE) printf("clean out [%d] %d\n", k, data.key);
          data.value.dist = 0.f;
          set_worked(data.value);
        }
      }
    }
  }

  __device__ __forceinline__ void clear_worked() {
    for (int i = threadIdx.x; i < HASHMAP_SIZE; i += BLOCK_DIM_X) {
      clear_worked(values[i]);
    }
  }

  // /**
  //  * Clear all but the first 'begin' entries. [parallel call]:
  //  *
  //  * cash.clear(1);
  //  *
  //  * Note: __syncthreads() need before next 'cache' call.
  //  *
  //  */
  // __device__ __forceinline__ void clear(const int begin = 0) {
  //   for (int i = threadIdx.x + begin; i < HASHMAP_SIZE::value;
  //        i += BLOCK_DIM_X) {
  //     hash[i] = -1;
  //   }
  // }

  /**
   * Prints first 'len' elements in the Cache. [parallel call]:
   * cash.print(8);
   *
   */
  __device__ __forceinline__ void print(int len = HASHMAP_SIZE) {
    __syncthreads();
    if (!threadIdx.x) {
      printf("Cache: \n");
      for (int i = 0; i < len; ++i) {
        printf("%d -> %llu [%2.2f] (%d) \n", i, hash[i], values[i].dist,
               (values[i].bits & 1));
      }
      printf("\n");
    }
  }
};

#endif  // CUDA_KNN_MULTI_WORKED_DISTS_CACHE_CUH_
