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

#ifndef INCLUDE_GGNN_SYM_CUDA_KNN_SYM_BUFFER_MERGE_LAYER_CUH_
#define INCLUDE_GGNN_SYM_CUDA_KNN_SYM_BUFFER_MERGE_LAYER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename T>
__global__ void
sym_buffer_merge(const T kernel) {
  kernel();
}

template <typename ValueT, typename KeyT, int K, int KF, int BLOCK_DIM_X,
          typename GAddrT = int32_t>
struct SymBufferMergeKernel {
  static constexpr int POINTS_PER_BLOCK = BLOCK_DIM_X / KF;
  static constexpr int KL = K - KF;

  void launch(const cudaStream_t stream = 0) {
    VLOG(2) << "SymBufferMergeKernel -- N: " << N << " [" << N_offset << " " << N_offset+N << "]\n";
    dim3 block(KF, POINTS_PER_BLOCK);
    sym_buffer_merge<<<(N - 1) / POINTS_PER_BLOCK + 1, block, 0, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    const GAddrT n = N_offset + blockIdx.x * POINTS_PER_BLOCK + threadIdx.y;
    const int kf = threadIdx.x;

    if (n >= N) return;

    __shared__ KeyT s_sym_buffer[POINTS_PER_BLOCK * KF]; // inverse links which need to be added to the graph
    __shared__ KeyT s_graph_buffer[POINTS_PER_BLOCK * KF]; // current contents of the graph's foreign/inverse link storage
    __shared__ bool s_found[POINTS_PER_BLOCK]; // whether the foreign link in the graph exists in the list of inverse links to be added

    // number of inverse links to be entered per point (only valid for threadIdx.x == 0)
    int r_num_links;
    if (!threadIdx.x) {
      r_num_links = d_sym_atomic[N_offset + blockIdx.x * POINTS_PER_BLOCK + threadIdx.y];
    }

    const GAddrT addr_graph = n * K + KL + kf;
    const int tid = threadIdx.y * KF + threadIdx.x;
    //# load buffer
    s_sym_buffer[tid] = d_sym_buffer[n * KF + kf];
    s_graph_buffer[tid] = d_graph[addr_graph];

    // add existing foreign links to the inverse link list if there is still room
    for (int i = 0; i < KF; i++) {
      if (!threadIdx.x) {
        // only search if there is a spot where we could add another link
        s_found[threadIdx.y] = r_num_links >= KF;
      }
      __syncthreads();

      KeyT r_graph;

      if (!s_found[threadIdx.y])
      {
        // read all requested inverse links per point
        const KeyT r_sym_buffer = s_sym_buffer[tid];
        // read existing foreign link i per point from graph
        r_graph = s_graph_buffer[threadIdx.y * KF + i];
        // existing foreign link exists in requested inverse link list? ==> found
        if (r_graph == r_sym_buffer) s_found[threadIdx.y] = true;
      }
      __syncthreads();

      // if there is still room and the existing foreign link is not part of the requested inverse links, add it
      if (!threadIdx.x && !s_found[threadIdx.y]) {
        s_sym_buffer[threadIdx.y * KF + r_num_links] = r_graph;
        ++r_num_links;
      }
    }

    __syncthreads();

    // store requested inverse links and added previous foreign links in the graph's foreign link list.
    // if there aren't enough links, store the points own index (to avoid entries with -1)
    const KeyT res = s_sym_buffer[tid];
    d_graph[addr_graph] = (res >= 0) ? res : n;
  }

  const KeyT* d_sym_buffer;  // [N, KF]
  const int* d_sym_atomic;  // [N]
  KeyT* d_graph;             // [N, K]

  int N;  // number of points to work on
  int N_offset;
};

#endif  // INCLUDE_GGNN_SYM_CUDA_KNN_SYM_BUFFER_MERGE_LAYER_CUH_
