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

#include <ggnn/construction/top_merge_layer.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/lib.h>

#include <ggnn/cuda_utils/distance.cuh>
#include <ggnn/cuda_utils/k_best_list.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ggnn {

template <typename T>
__global__ void __launch_bounds__(T::BLOCK_DIM_X) top(const T kernel)
{
  kernel();
}

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
__device__ __forceinline__ void
TopMergeKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>::operator()() const
{
  using Distance = ggnn::Distance<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD>;
  using KBestList = ggnn::KBestList<KeyT, ValueT, BLOCK_DIM_X>;

  const uint32_t n = blockIdx.x;
  const KeyT m = (!layer) ? n : d_translation[n];

  Distance distCalc(D, measure, d_base, m);
  KBestList best(KBuild);

  const uint32_t S_plus_offset = S_offset * (S + 1);
  const uint32_t S_actual = (!layer && n < S_plus_offset) ? S + 1 : S;

  const KeyT start = (layer || n < S_plus_offset)
                         ? (n / S_actual) * S_actual
                         : S_plus_offset + ((n - S_plus_offset) / S_actual) * S_actual;
  const KeyT end = start + S_actual;

  for (KeyT other_n = start; other_n < end; other_n++) {
    __syncthreads();
    const KeyT other_m = (layer) ? d_translation[other_n] : other_n;

    if (m == other_m)
      continue;
    ValueT dist = distCalc.distance_synced(other_m);

    best.add_unique(dist, other_n);
  }

  for (uint32_t k = threadIdx.x; k < KBuild; k += BLOCK_DIM_X) {
    d_graph[static_cast<size_t>(n) * KBuild + k] = best.s_ids[k];
  }
  if (!threadIdx.x) {
    ValueT nn1_dist = best.s_dists[1];
    if (measure == DistanceMeasure::Euclidean)
      nn1_dist = sqrtf(nn1_dist);
    d_nn1_dist_buffer[n] = nn1_dist;
  }
}

#define GGNN_TOP_MERGE(KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD) \
  template __global__ void                                                     \
  top<TopMergeKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>>( \
      const TopMergeKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>);

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_TOPS, GGNN_TOP_MERGE);

};  // namespace ggnn
