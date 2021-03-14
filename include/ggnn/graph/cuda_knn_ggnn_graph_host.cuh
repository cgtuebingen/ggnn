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
// Authors: Fabian Groh, Lukas Rupert, Patrick Wieschollek, Hendrik P.A. Lensch
//

#ifndef INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_HOST_CUH_
#define INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_HOST_CUH_

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "ggnn/utils/cuda_knn_utils.cuh"
#include "ggnn/graph/cuda_knn_ggnn_graph_device.cuh"

/**
 * GGNN graph data (on the CPU)
 *
 * @param KeyT datatype of dataset indices
 * @param BaseT datatype of dataset values
 * @param ValueT distance value type
 */
template <typename KeyT, typename BaseT, typename ValueT>
struct GGNNGraphHost {
  typedef GGNNGraphDevice<KeyT, BaseT, ValueT> GGNNGraphDevice;

  /// neighborhood vectors
  KeyT* h_graph;
  /// translation of upper layer points into lowest layer
  KeyT* h_translation;
  /// translation of upper layer points into one layer below
  KeyT* h_selection;

  /// average and maximum distance to nearest known neighbors
  ValueT* h_nn1_stats;

  /// combined memory pool
  char* h_memory;

  size_t total_graph_size;

  int current_part_id {-1};

  std::thread disk_io_thread;

  GGNNGraphHost(const int N, const int K, const int N_all, const int ST_all) {
    // just to make sure that everything is sufficiently aligned
    auto align8 = [](size_t size) -> size_t {return ((size+7)/8)*8;};

    const size_t graph_size = align8(static_cast<size_t>(N_all) * K * sizeof(KeyT));
    const size_t selection_translation_size = align8(ST_all * sizeof(KeyT));
    // const size_t nn1_dist_buffer_size = N * sizeof(ValueT);
    const size_t nn1_stats_size = align8(2 * sizeof(ValueT));
    total_graph_size = graph_size + 2 * selection_translation_size + nn1_stats_size;

    VLOG(1) << "GGNNGraphHost(): N: " << N << ", K: " << K
            << ", N_all: " << N_all << ", ST_all: " << ST_all
            << " (" << total_graph_size/(1024.0f*1024.0f*1024.0f) <<" GB total)\n";

    CHECK_CUDA(cudaMallocHost(&h_memory, total_graph_size));

    size_t pos = 0;
    h_graph = reinterpret_cast<KeyT*>(h_memory+pos);
    pos += graph_size;
    h_translation = reinterpret_cast<KeyT*>(h_memory+pos);
    pos += selection_translation_size;
    h_selection = reinterpret_cast<KeyT*>(h_memory+pos);
    pos += selection_translation_size;
    h_nn1_stats = reinterpret_cast<ValueT*>(h_memory+pos);
    pos += nn1_stats_size;

    CHECK_EQ(pos, total_graph_size);

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());
  }

  GGNNGraphHost(const GGNNGraphHost& other) {
    // this exists to allow using vector::emplace_back
    // when it triggers a reallocation, this code will be called.
    // always make sure that enough memory is reserved ahead of time.
    LOG(FATAL) << "copying is not supported. reserve()!";
  }

  ~GGNNGraphHost() {
    cudaFreeHost(h_memory);
  }

  void downloadAsync(const GGNNGraphDevice& graph) {
    cudaMemcpyAsync(h_graph, graph.d_graph, total_graph_size, cudaMemcpyDeviceToHost, graph.stream);
  }

  void uploadAsync(GGNNGraphDevice& graph) {
    cudaMemcpyAsync(graph.d_graph, h_graph, total_graph_size, cudaMemcpyHostToDevice, graph.stream);
  }

  void store(const std::string& filename){
    std::ofstream outFile;

    outFile.open(filename, std::ofstream::out | std::ofstream::binary |
                               std::ofstream::trunc);

    CHECK(outFile.is_open()) << "Unable to open " << filename;

    outFile.write(h_memory, total_graph_size);

    outFile.close();
  }

  void load(const std::string& filename){
    std::ifstream inFile;

    inFile.open(filename, std::ifstream::in | std::ifstream::binary);

    CHECK(inFile.is_open()) << "Unable to open " << filename;

    inFile.seekg(0, std::ifstream::end);
    size_t filesize = inFile.tellg();
    inFile.seekg(0, std::ifstream::beg);

    CHECK_EQ(filesize, total_graph_size) << "Error on loading" << filename <<
       ". File size of GGNNGraph does not match the expected size.";

    inFile.read(h_memory, total_graph_size);

    inFile.close();
  }
};

#endif  // INCLUDE_GGNN_GRAPH_CUDA_KNN_GGNN_GRAPH_HOST_CUH_
