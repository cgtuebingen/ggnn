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

#ifndef INCLUDE_GGNN_DISTANCE_CUH
#define INCLUDE_GGNN_DISTANCE_CUH

#include <ggnn/base/def.h>

#include <cstddef>
#include <cstdint>

#include <cub/cub.cuh>

namespace ggnn {

/**
 * Distance calculates the distance/difference between the base vector and
 * other_id vector.
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_DIM_X,
          uint32_t DIST_ITEMS_PER_THREAD>
struct Distance {
  const uint32_t D;
  const DistanceMeasure measure;

  // only valid in thread 0, only needed if measure == Cosine
  ValueT r_query_norm;

  using AddrT = size_t;

  struct DistanceAndNorm {
    ValueT dist{0.0f};
    ValueT norm{0.0f};

    struct Sum {
      __host__ __device__ __forceinline__ DistanceAndNorm operator()(const DistanceAndNorm& a,
                                                                     const DistanceAndNorm& b) const
      {
        return {a.dist + b.dist, a.norm + b.norm};
      }
    };
  };

  using BlockReduceDist = cub::BlockReduce<ValueT, BLOCK_DIM_X>;
  using BlockReduceDistAndNorm = cub::BlockReduce<DistanceAndNorm, BLOCK_DIM_X>;

  union TempStorage {
    typename BlockReduceDist::TempStorage dist_temp_storage;
    typename BlockReduceDistAndNorm::TempStorage dist_and_norm_temp_storage;
  };

  const BaseT* d_base;
  BaseT r_query[DIST_ITEMS_PER_THREAD];

  TempStorage& s_temp_storage;
  __device__ __forceinline__ TempStorage& PrivateTmpStorage()
  {
    __shared__ TempStorage s_tmp;
    return s_tmp;
  }
  ValueT& s_dist;
  __device__ __forceinline__ ValueT& DistTmpStorage()
  {
    __shared__ ValueT s_dist;
    return s_dist;
  }

  __device__ __forceinline__ Distance(const uint32_t D, DistanceMeasure measure,
                                      const BaseT* d_base, const BaseT* d_query, const KeyT n)
      : D(D),
        measure(measure),
        d_base(d_base),
        s_temp_storage(PrivateTmpStorage()),
        s_dist(DistTmpStorage())
  {
    loadQueryPos(d_query + static_cast<AddrT>(n) * D);
  }

  __device__ __forceinline__ Distance(const uint32_t D, DistanceMeasure measure,
                                      const BaseT* d_base, const KeyT n)
      : D(D),
        measure(measure),
        d_base(d_base),
        s_temp_storage(PrivateTmpStorage()),
        s_dist(DistTmpStorage())
  {
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
      r_query_norm = BlockReduceDist(s_temp_storage.dist_temp_storage).Sum(query_norm);
    }
  }

  __device__ __forceinline__ ValueT distance_synced(const KeyT other_id)
  {
    BaseT r_other[DIST_ITEMS_PER_THREAD];

    for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
      const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
      r_other[item] = (read_dim < D) ? d_base[static_cast<AddrT>(other_id) * D + read_dim] : 0;
    }
    ValueT dist{0.0f};
    if (measure == DistanceMeasure::Euclidean) {
      for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
        const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
        const ValueT diff =
            (read_dim < D) ? static_cast<ValueT>(r_other[item]) - static_cast<ValueT>(r_query[item])
                           : 0;
        dist += diff * diff;
      }
      dist = BlockReduceDist(s_temp_storage.dist_temp_storage).Sum(dist);
      if (!threadIdx.x)
        s_dist = dist;
    }
    if (measure == DistanceMeasure::Cosine) {
      DistanceAndNorm dist_and_norm{0.0f, 0.0f};
      for (uint32_t item = 0; item < DIST_ITEMS_PER_THREAD; ++item) {
        const uint32_t read_dim = item * BLOCK_DIM_X + threadIdx.x;
        dist_and_norm.dist +=
            (read_dim < D) ? static_cast<ValueT>(r_other[item]) * static_cast<ValueT>(r_query[item])
                           : 0;
        dist_and_norm.norm +=
            (read_dim < D) ? static_cast<ValueT>(r_other[item]) * static_cast<ValueT>(r_other[item])
                           : 0;
      }
      dist_and_norm = BlockReduceDistAndNorm(s_temp_storage.dist_and_norm_temp_storage)
                          .Reduce(dist_and_norm, DistanceAndNorm::Sum());
      if (!threadIdx.x) {
        // need to normalize by the vectors' lengths (in high dimensions, no vector has length 1.0f)
        const ValueT norm_sqr = r_query_norm * dist_and_norm.norm;
        // use negative dot product, as larger values are closer to each other
        s_dist = (norm_sqr > 0.0f) ? fabs(1.0f - dist_and_norm.dist / sqrtf(norm_sqr)) : 1.0f;
      }
    }
    __syncthreads();

    return s_dist;
  }
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_DISTANCE_CUH
