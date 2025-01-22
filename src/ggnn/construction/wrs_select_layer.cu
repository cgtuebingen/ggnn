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

#include <ggnn/construction/wrs_select_layer.cuh>

#include <ggnn/base/lib.h>

#include <cstdint>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

namespace ggnn {

template <typename T>
__global__ void __launch_bounds__(T::BLOCK_DIM_X) select(const T kernel)
{
  kernel();
}

/*
 * Selection of K Points per B for Layers.
 */
template <typename KeyT, typename ValueT>
__device__ __forceinline__ void WRSSelectionKernel<KeyT, ValueT>::operator()() const
{
  using BlockRadixSort = cub::BlockRadixSort<ValueT, BLOCK_DIM_X, ITEMS_PER_THREAD, KeyT>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const uint32_t b = blockIdx.x;

  const uint32_t S_current = S + (b < S_offset);
  const uint32_t start = b * S + min(b, S_offset);

  ValueT keys[ITEMS_PER_THREAD];
  KeyT values[ITEMS_PER_THREAD];

  for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
    const uint32_t i = item * BLOCK_DIM_X + threadIdx.x;
    if (i < S_current) {
      const KeyT n = start + i;
      const float e = (-1 * logf(d_rng[n])) /
                      // the top merge kernel is configured to output the matching values for the
                      // current layer otherwise, we would need to translate n to the bottom layer
                      (d_nn1_dist_buffer[n] + std::numeric_limits<float>::epsilon());
      keys[item] = e;
      values[item] = n;
    }
    else {
      // NOTE: if this happens, the following sym query will fail
      // this is prevented by keeping the segment size (S) > the number of foreign edges (KF)
      keys[item] = -1.f;
      values[item] = -1;
    }
  }

  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, values);

  // block index / growth ==> index of the upper segment
  const uint32_t upper_segment = b / G;
  // n-th segment contributing to the upper segment
  const uint32_t nth_lower_segment = b - upper_segment * G;

  // number of points contributed by the current block
  // evenly distributed between blocks as SG=S/G + the first SG_offset many blocks contribute one
  // more
  const uint32_t num_selected_points = SG + (nth_lower_segment < SG_offset);

  // destination for selected points
  // start of upper segment + point contributed by previous blocks to this segment
  const uint32_t dest =
      upper_segment * Sglob + nth_lower_segment * SG + min(nth_lower_segment, SG_offset);

  __syncthreads();

  for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
    const uint32_t s = threadIdx.x + item * BLOCK_DIM_X;
    if (s < num_selected_points) {
      const KeyT n = values[item];

      d_selection[dest + s] = n;
      d_translation[dest + s] = (!layer) ? n : d_translation_layer[n];
    }
  }
}

#define GGNN_SELECT(KeyT, ValueT)                                    \
  template __global__ void select<WRSSelectionKernel<KeyT, ValueT>>( \
      const WRSSelectionKernel<KeyT, ValueT>);

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_SELECT);

};  // namespace ggnn
