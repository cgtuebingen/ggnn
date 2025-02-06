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

#include <ggnn/query/bf_query_layer.cuh>

#include <ggnn/base/lib.h>

#include <ggnn/cuda_utils/distance.cuh>
#include <ggnn/cuda_utils/k_best_list.cuh>

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

namespace ggnn {

template <typename T>
__global__ void __launch_bounds__(T::BLOCK_DIM_X) bf_query(const T kernel)
{
  kernel();
}

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE, bool WRITE_DISTS>
__device__ __forceinline__ void
BruteForceQueryKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, WRITE_DISTS>::operator()() const
{
  using Distance = Distance<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;
  using KBestList = KBestList<KeyT, ValueT, BLOCK_DIM_X>;

  const KeyT n = static_cast<int>(blockIdx.x);

  Distance distCalc(D, measure, d_base, d_query, n);
  KBestList best(KQuery);
  __syncthreads();

  for (KeyT i = 0; i < N_base; ++i) {
    // fetch the entire base, one by one
    ValueT dist = distCalc.distance_synced(i);
    if (dist < best.worst())  // should be faster than checking all elements
      best.add_unique(dist, i);
  }
  __syncthreads();

  for (uint32_t k = threadIdx.x; k < KQuery; k += BLOCK_DIM_X) {
    d_query_results[static_cast<size_t>(n) * KQuery + k] = best.s_ids[k];
    if constexpr (WRITE_DISTS)
      d_query_results_dists[static_cast<size_t>(n) * KQuery + k] = best.s_dists[k];
  }
}

#define GGNN_BF_QUERY(KeyT, ValueT, BaseT, BLOCK_DIM_X, WRITE_DISTS)              \
  template __global__ void                                                        \
  bf_query<BruteForceQueryKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, WRITE_DISTS>>( \
      const BruteForceQueryKernel<KeyT, ValueT, BaseT, BLOCK_DIM_X, WRITE_DISTS>);

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_QUERYS, GGNN_WRITE_DISTS, GGNN_BF_QUERY);

};  // namespace ggnn
