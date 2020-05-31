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
#ifndef CUDA_KNN_DISTANCE_CUH_
#define CUDA_KNN_DISTANCE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/utils/cuda_knn_core_utils.cuh"

/**
 * Distance calculates the distance/difference between the base vector and
 * other_id vector.
 */
template <typename ValueT, typename KeyT, int D, int BLOCK_DIM_X,
          typename BaseT = ValueT, typename AddrT = KeyT>
struct Distance {
  enum { ITEMS_PER_THREAD = (D - 1) / BLOCK_DIM_X + 1 };

  typedef cub::BlockReduce<ValueT, BLOCK_DIM_X> BlockReduce;
  typedef typename BlockReduce::TempStorage TempStorage;

  TempStorage* storage;
  const BaseT* d_base;
  BaseT r_query[ITEMS_PER_THREAD];

  /**
   *
   * Distance uses cub and TempStorage:
   *
   * typedef Distance<ValueT, KeyT, D, BLOCK_DIM_X> Distance;
   * __shared__ union { typename Distance::TempStorage dist; } temp_storage;
   *
   * Distance dist_calc(&temp_storage.dist, d_base, d_query, blockIdx.x);
   *
   */
  __device__ __forceinline__ Distance(TempStorage* storage, const BaseT* d_base,
                                      const BaseT* d_query, const KeyT n)
      : storage(storage), d_base(d_base) {
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D)
        r_query[item] = d_query[static_cast<AddrT>(n) * D + read_dim];
    }
  }

  /**
   *
   * Distance uses cub and TempStorage:
   *
   * typedef Distance<ValueT, KeyT, D, BLOCK_DIM_X> Distance;
   * __shared__ union { typename Distance::TempStorage dist; } temp_storage;
   *
   * Distance dist_calc(&temp_storage.dist, d_base, blockIdx.x);
   *
   */
  __device__ __forceinline__ Distance(TempStorage* storage, const BaseT* d_base,
                                      const KeyT n)
      : storage(storage), d_base(d_base) {
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D)
        r_query[item] = d_base[static_cast<AddrT>(n) * D + read_dim];
    }
  }

  /**
   *
   * Distance uses cub and TempStorage:
   *
   * typedef Distance<ValueT, KeyT, D, BLOCK_DIM_X> Distance;
   * __shared__ union { typename Distance::TempStorage dist; } temp_storage;
   *
   * Distance dist_calc(&temp_storage.dist, d_base);
   *
   * WARNING: r_query remains uninitialized
   * make sure to initialize it before querying distances
   */
  __device__ __forceinline__ Distance(TempStorage* storage, const BaseT* d_base)
      : storage(storage), d_base(d_base) {}

  /**
   * Calculates distance of base vector to other_id vector.
   *
   * [parallel call]:
   * ValueT dist = distCalc.distance(other_id)
   *
   * Return:
   * 	ValueT distance
   *
   * Note: distance only valid in first thread.
   */
  __device__ __forceinline__ ValueT distance(const KeyT other_id) {
    ValueT r_diff = 0;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        ValueT pos_other =
            r_query[item] - d_base[static_cast<AddrT>(other_id) * D + read_dim];
        r_diff += pos_other * pos_other;
      }
    }

    return BlockReduce(*storage).Sum(r_diff);
  }

  /**
   * Calculates the distance between vectors q and p.
   *
   * [parallel call]:
   * ValueT dist = distCalc.distance(q_id,p_id)
   *
   * Return:
   * 	ValueT distance
   *
   * Note: distance only valid in first thread.
   */
  __device__ __forceinline__ ValueT distance(const KeyT q_id, const KeyT p_id) {
    ValueT r_diff = 0;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        ValueT pos_other = d_base[static_cast<AddrT>(q_id) * D + read_dim] -
                           d_base[static_cast<AddrT>(p_id) * D + read_dim];
        r_diff += pos_other * pos_other;
      }
    }

    return BlockReduce(*storage).Sum(r_diff);
  }

  /**
   * Calculates synced distance of base vector to other_id vector.
   *
   * [parallel call]:
   * ValueT dist = distCalc.distance(other_id)
   *
   * Return:
   * 	ValueT distance
   *
   * Note: distance valid in all threads.
   */
  __device__ __forceinline__ ValueT distance_synced(const KeyT other_id) {
    ValueT r_diff = 0;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        ValueT pos_other =
            r_query[item] - d_base[static_cast<AddrT>(other_id) * D + read_dim];
        r_diff += pos_other * pos_other;
      }
    }

    __shared__ ValueT s_diff;

    ValueT aggregate = BlockReduce(*storage).Sum(r_diff);
    if (!threadIdx.x) s_diff = aggregate;
    __syncthreads();

    return s_diff;
  }

  __device__ __forceinline__ ValueT distance_synced_dbg(const KeyT other_id) {
    ValueT r_diff = 0;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        ValueT pos_other =
            r_query[item] - d_base[static_cast<AddrT>(other_id) * D + read_dim];
        printf("<gpu> %d -> %f (%f)| %u - %u \n", read_dim,
               pos_other * pos_other, pos_other, r_query[item],
               d_base[other_id * D + read_dim]);
        r_diff += pos_other * pos_other;
      }
    }

    __shared__ ValueT s_diff;

    ValueT aggregate = BlockReduce(*storage).Sum(r_diff);
    if (!threadIdx.x) printf("res: %f \n", threadIdx.x);
    if (!threadIdx.x) s_diff = aggregate;
    __syncthreads();

    return s_diff;
  }
};

#endif  // CUDA_KNN_DISTANCE_CUH_
