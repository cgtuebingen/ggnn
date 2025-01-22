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

#ifndef INCLUDE_GGNN_BF_QUERY_LAYER_CUH
#define INCLUDE_GGNN_BF_QUERY_LAYER_CUH

#include <ggnn/base/def.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>

namespace ggnn {

template <typename T>
__global__ void bf_query(const T kernel);

/**
 * query which loops through all points and can be used to create ground truth data
 */
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          bool WRITE_DISTS = false>
struct BruteForceQueryKernel {
  static constexpr uint32_t BLOCK_DIM_X = BLOCK_SIZE;
  static constexpr uint32_t DIST_ITEMS_PER_THREAD = 4;

  void launch(const uint32_t N, const cudaStream_t stream = 0)
  {
    VLOG(1) << "BruteForceQueryKernel -- KQuery: " << KQuery;

    CHECK_LE(D, BLOCK_DIM_X * DIST_ITEMS_PER_THREAD);

    const size_t sm_size = KQuery * (sizeof(KeyT) + sizeof(ValueT));

    bf_query<<<N, BLOCK_DIM_X, sm_size, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const;

  const uint32_t D;
  const DistanceMeasure measure;
  const uint32_t KQuery;

  KeyT N_base;  // number of base points

  const BaseT* d_base;   // [Nall,D]
  const BaseT* d_query;  // [Nq,D]

  KeyT* d_query_results;          // [Nq,KQuery]
  ValueT* d_query_results_dists;  // [Nq,KQuery]
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_BF_QUERY_LAYER_CUH
