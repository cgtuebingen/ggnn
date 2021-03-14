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
//

#ifndef INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_DEVICE_CUH_
#define INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_DEVICE_CUH_

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <glog/logging.h>

#include "ggnn/utils/cuda_knn_utils.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"

/**
 * GGNN graph data (on the GPU)
 *
 * @param KeyT datatype of dataset indices
 * @param BaseT datatype of dataset values
 * @param ValueT distance value type
 */
template <typename KeyT, typename BaseT, typename ValueT>
struct GGNNGraphDevice {
  /// neighborhood vectors
  KeyT* d_graph;
  /// translation of upper layer points into lowest layer
  KeyT* d_translation;
  /// translation of upper layer points into one layer below
  KeyT* d_selection;

  /// average and maximum distance to nearest known neighbors
  ValueT* d_nn1_stats;

  /// base data pointer for the shard.
  BaseT* d_base;

  /// combined memory pool
  char* d_memory;
  size_t base_size {0};
  size_t total_graph_size {0};

  int current_part_id {-1};

  cudaStream_t stream;

  GGNNGraphDevice(const int N, const int D, const int K, const int N_all, const int ST_all) {
    // just to make sure that everything is sufficiently aligned
    auto align8 = [](size_t size) -> size_t {return ((size+7)/8)*8;};

    const size_t graph_size = align8(static_cast<size_t>(N_all) * K * sizeof(KeyT));
    const size_t selection_translation_size = align8(ST_all * sizeof(KeyT));
    const size_t nn1_stats_size = align8(2 * sizeof(ValueT));
    total_graph_size = graph_size + 2 * selection_translation_size + nn1_stats_size;
    base_size = align8(static_cast<size_t>(N) * D * sizeof(BaseT));

    const size_t total_size = base_size+total_graph_size;

    VLOG(2) << "GGNNGraphDevice(): allocating GPU memory... ("
            << total_graph_size/(1024.0f*1024.0f*1024.0f) << " GB graph + "
            << base_size/(1024.0f*1024.0f*1024.0f) << " GB base)\n";

    {
      size_t free, total;
      CHECK_CUDA(cudaMemGetInfo(&free, &total));
      CHECK_GE(free, total_size) << "out of memory.";
    }

    CHECK_CUDA(cudaMalloc(&d_memory, total_size));

    size_t pos = 0;
    d_base = reinterpret_cast<BaseT*>(d_memory+pos);
    pos += base_size;
    d_graph = reinterpret_cast<KeyT*>(d_memory+pos);
    pos += graph_size;
    d_translation = reinterpret_cast<KeyT*>(d_memory+pos);
    pos += selection_translation_size;
    d_selection = reinterpret_cast<KeyT*>(d_memory+pos);
    pos += selection_translation_size;
    d_nn1_stats = reinterpret_cast<ValueT*>(d_memory+pos);
    pos += nn1_stats_size;

    CHECK_EQ(pos, total_size);

    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());
  }

  GGNNGraphDevice(const GGNNGraphDevice& other) {
    // this exists to allow using vector::emplace_back
    // when it triggers a reallocation, this code will be called.
    // always make sure that enough memory is reserved ahead of time.
    LOG(FATAL) << "copying is not supported. reserve()!";
  }

  ~GGNNGraphDevice() {
    cudaFree(d_memory);

    CHECK_CUDA(cudaStreamDestroy(stream));
  }
};

#endif  // INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_DEVICE_CUH_
