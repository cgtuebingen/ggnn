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

#include <ggnn/construction/merge_layer.cuh>

#include <ggnn/base/def.h>
#include <ggnn/base/lib.h>

#include <ggnn/cuda_utils/simple_knn_cache.cuh>

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

namespace ggnn {

template <typename T>
__global__ void __launch_bounds__(T::BLOCK_DIM_X) merge(const T kernel)
{
  kernel();
}

// determine the start of the top-layer segment (always 0 for layer_top = L-1)
template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
__device__ __forceinline__ uint32_t
MergeKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>::get_top_seg_offset(
    const KeyT n) const
{
  // first, determine the bottom-level segment
  uint32_t seg_btm = n / S;
  if (!layer_btm) {
    const KeyT offset_points = S0_offset * (S0 + 1);
    seg_btm = (n < offset_points) ? n / (S0 + 1) : S0_offset + (n - offset_points) / S0;
  }

  // then divide by G once per layer to step up the tree
  // and finally multiply by S to get the start of the segment

  uint32_t powG = G;  // assuming layer_top > layer_btm (which should always be the case)
  for (uint32_t i = 1; i < layer_top - layer_btm; ++i)
    powG *= G;

  return (seg_btm / powG) * S;
}

template <typename KeyT, typename ValueT, typename BaseT, uint32_t BLOCK_SIZE,
          uint32_t DIST_ITEMS_PER_THREAD>
__device__ __forceinline__ void
MergeKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>::operator()() const
{
  static constexpr uint32_t K_BLOCK = 32;
  static_assert(K_BLOCK <= BLOCK_DIM_X);
  static constexpr bool DIST_STATS = false;

  using Cache = SimpleKNNCache<KeyT, ValueT, BaseT, BLOCK_DIM_X, DIST_ITEMS_PER_THREAD, DIST_STATS>;

  const float xi = (measure == DistanceMeasure::Euclidean)
                       ? (d_nn1_stats[0] * d_nn1_stats[0]) * tau_build * tau_build
                       : d_nn1_stats[0] * tau_build;

  const KeyT n = static_cast<KeyT>(blockIdx.x);

  const KeyT m = (!layer_btm) ? n : d_translation[STs_offsets[layer_btm] + n];

  Cache cache(D, measure, KBuild + 1, SORTED_SIZE, CACHE_SIZE, d_base, m, xi);

  __shared__ KeyT s_knn[K_BLOCK];

  {
    const uint32_t s_offset = get_top_seg_offset(n);

    // fetch starting points
    for (uint32_t i = 0; i < S; i += K_BLOCK) {
      if (threadIdx.x < K_BLOCK) {
        const uint32_t s = i + threadIdx.x;
        s_knn[threadIdx.x] = (s < S) ? static_cast<KeyT>(s_offset + s) : Cache::EMPTY_KEY;
      }
      cache.fetch_unfiltered(s_knn, &d_translation[STs_offsets[layer_top]], K_BLOCK);
    }
  }

  // hierarchic kNN search
  for (uint32_t layer = layer_top - 1; layer >= layer_btm && layer != -1U; layer--) {
    cache.transform(&d_selection[STs_offsets[layer + 1]]);

    if (layer == layer_btm)
      cache.fetch_unfiltered(&n, (!layer) ? nullptr : &d_translation[STs_offsets[layer]], 1);

    for (uint32_t ite = 0; ite < MAX_ITERATIONS; ++ite) {
      const KeyT anchor = cache.pop();
      if (anchor == Cache::EMPTY_KEY)
        break;

      for (uint32_t j = 0; j < KBuild; j += K_BLOCK) {
        if (threadIdx.x < K_BLOCK) {
          const uint32_t k = j + threadIdx.x;
          s_knn[threadIdx.x] =
              (k < KBuild) ? d_graph[(static_cast<size_t>(Ns_offsets[layer]) + anchor) * KBuild + k]
                           : Cache::EMPTY_KEY;
        }
        cache.fetch(s_knn, (!layer) ? nullptr : &d_translation[STs_offsets[layer]], K_BLOCK);
      }
    }
  }

  KeyT& s_own_idx{s_knn[0]};
  if (!threadIdx.x)
    s_own_idx = static_cast<KeyT>(-1);
  __syncthreads();

  // check if own index is part of cache and mark its index to skip it
  // we cannot rely on it being at index 0 (in case of duplicates) or in the cache at all
  for (uint32_t j = 0; j < KBuild; j += BLOCK_DIM_X) {
    const uint32_t k = j + threadIdx.x;
    if (k < KBuild) {
      if (cache.s_cache[k] == n)
        s_own_idx = static_cast<KeyT>(k);
    }
  }
  __syncthreads();
  for (uint32_t j = 0; j < KBuild; j += BLOCK_DIM_X) {
    const uint32_t k = j + threadIdx.x;
    if (k < KBuild) {
      // skip self-referential link (if any)
      const KeyT idx = cache.s_cache[k + (static_cast<KeyT>(k) >= s_own_idx)];
      d_graph_buffer[static_cast<size_t>(n) * KBuild + k] = (idx != Cache::EMPTY_KEY) ? idx : n;
    }
  }

  if (!layer_btm && !threadIdx.x) {
    uint32_t i = s_own_idx + 1;
    ValueT dist;
    do {
      dist = cache.s_dists[i];
      ++i;
    } while (dist == 0.0f && i < cache.BEST_SIZE);
    if (measure == DistanceMeasure::Euclidean)
      dist = sqrtf(dist);
    d_nn1_dist_buffer[n] = dist;
  }
}

#define GGNN_MERGE(KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD)    \
  template __global__ void                                                    \
  merge<MergeKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>>( \
      const MergeKernel<KeyT, ValueT, BaseT, BLOCK_SIZE, DIST_ITEMS_PER_THREAD>);

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_BASES, GGNN_MERGES, GGNN_MERGE);

};  // namespace ggnn
