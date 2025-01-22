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

#include <ggnn/construction/sym_buffer_merge_layer.cuh>

#include <ggnn/base/lib.h>

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

namespace ggnn {

template <typename T>
__global__ void __launch_bounds__(T::BLOCK_DIM_X) sym_buffer_merge(const T kernel, const uint32_t N)
{
  kernel(N);
}

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void SymBufferMergeKernel<KeyT, ValueT>::operator()(uint32_t N) const
{
  const uint32_t n = blockIdx.x * POINTS_PER_BLOCK + threadIdx.y;
  const uint32_t kf = threadIdx.x;

  if (n >= N)
    return;

  /// inverse links which need to be added to the graph
  extern __shared__ KeyT s_sym_buffer[];  // [POINTS_PER_BLOCK * KF];
  /// current contents of the graph's foreign/inverse link storage
  KeyT* s_graph_buffer{&s_sym_buffer[POINTS_PER_BLOCK * KF]};  // [POINTS_PER_BLOCK * KF];
  /// whether the foreign link in the graph exists in the list of inverse links to be added
  bool* s_found{
      reinterpret_cast<bool*>(&s_sym_buffer[2 * POINTS_PER_BLOCK * KF])};  // [POINTS_PER_BLOCK];

  // number of inverse links to be entered per point (only valid for threadIdx.x == 0)
  uint32_t r_num_links;
  if (!threadIdx.x) {
    r_num_links = d_sym_atomic[n];
  }

  const uint32_t tid = threadIdx.y * KF + threadIdx.x;
  // # load buffer
  s_sym_buffer[tid] = d_sym_buffer[static_cast<size_t>(n) * KF + kf];
  s_graph_buffer[tid] = d_graph[static_cast<size_t>(n) * KBuild + KL + kf];

  // add existing foreign links to the inverse link list if there is still room
  for (uint32_t i = 0; i < KF; i++) {
    if (!threadIdx.x) {
      // only search if there is a spot where we could add another link
      s_found[threadIdx.y] = r_num_links >= KF;
    }
    __syncthreads();

    KeyT r_graph;

    if (!s_found[threadIdx.y]) {
      // read all requested inverse links per point
      const KeyT r_sym_buffer = s_sym_buffer[tid];
      // read existing foreign link i per point from graph
      r_graph = s_graph_buffer[threadIdx.y * KF + i];
      // existing foreign link exists in requested inverse link list? ==> found
      if (r_graph == r_sym_buffer)
        s_found[threadIdx.y] = true;
    }
    __syncthreads();

    // if there is still room and the existing foreign link is not part of the requested inverse
    // links, add it
    if (!threadIdx.x && !s_found[threadIdx.y]) {
      s_sym_buffer[threadIdx.y * KF + r_num_links] = r_graph;
      ++r_num_links;
    }
  }

  __syncthreads();

  // store requested inverse links and added previous foreign links in the graph's foreign link
  // list. if there aren't enough links, store the points own index (to avoid entries with -1)
  const KeyT res = s_sym_buffer[tid];
  d_graph[static_cast<size_t>(n) * KBuild + KL + kf] = (res >= 0) ? res : n;
}

#define GGNN_SYM_BUFFER_MERGE(KeyT, ValueT)                                      \
  template __global__ void sym_buffer_merge<SymBufferMergeKernel<KeyT, ValueT>>( \
      const SymBufferMergeKernel<KeyT, ValueT>, const uint32_t);

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_SYM_BUFFER_MERGE);

};  // namespace ggnn
