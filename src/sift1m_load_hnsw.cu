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
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#include "ggnn/cuda_knn_ggnn.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_dataset.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"
#include "ggnn/utils/hnswlib_loader.hpp"

DEFINE_string(base_filename, "", "path to file with base vectors");
DEFINE_string(query_filename, "", "path to file with perform_query vectors");
DEFINE_string(groundtruth_filename, "",
              "path to file with groundtruth vectors");
DEFINE_string(graph_filename, "",
              "path to file that contains the serialized HNSW index (Hnswlib)");
DEFINE_int32(gpu_id, 0, "GPU id");
DEFINE_bool(grid_search, false,
            "Perform queries for a wide range of parameters.");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  gflags::SetUsageMessage(
      "GGNN: Graph-based GPU Nearest Neighbor Search\n"
      "by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. "
      "Lensch\n"
      "(c) 2020 Computer Graphics University of Tuebingen");
  gflags::SetVersionString("1.0.0");
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Reading files";
  CHECK(file_exists(FLAGS_base_filename))
      << "File for base vectors has to exist";
  CHECK(file_exists(FLAGS_query_filename))
      << "File for perform_query vectors has to exist";

  // ####################################################################
  // compile-time configuration
  //
  // data types
  //
  /// data type for addressing points (needs to be able to represent N)
  using KeyT = int32_t;
  /// data type of the dataset (e.g., char, int, float)
  using BaseT = float;
  /// data type of computed distances
  using ValueT = float;
  /// data type for addressing base-vectors (needs to be able to represent N*D)
  using BAddrT = uint32_t;
  /// data type for addressing the graph (needs to be able to represent
  /// N*KBuild)
  using GAddrT = uint32_t;
  //
  // dataset configuration (here: SIFT1M)
  //
  /// dimension of the dataset
  const int D = 128;
  /// distance measure (Euclidean or Cosine)
  const DistanceMeasure measure = Euclidean;
  //
  // HNSW configuration
  const int M = 20;
  const int KBuild = M * 2;
  const int KF = KBuild / 2;
  // only one entry point, otherwise no hierarchy
  const int S = 1;
  const int L = 2;
  //
  // query configuration
  //
  /// number of neighbors to search for
  const int KQuery = 10;

  // static_assert(KBuild-KF < S, "there are not enough points to fill the local
  // neighbor list!");

  LOG(INFO) << "Using the following parameters " << KBuild << " (KBuild) " << KF
            << " (KF) " << S << " (S) " << L << " (L) " << D << " (D) ";

  const bool import_graph =
      !FLAGS_graph_filename.empty() && file_exists(FLAGS_graph_filename);

  CHECK(import_graph) << "A HNSW index must be provided.";

  // Set the requested GPU id, if possible.
  {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    CHECK_GE(FLAGS_gpu_id, 0) << "This GPU does not exist";
    CHECK_LT(FLAGS_gpu_id, numGpus) << "This GPU does not exist";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, FLAGS_gpu_id);
    LOG(INFO) << "device name: " << prop.name;
  }
  cudaSetDevice(FLAGS_gpu_id);

  typedef GGNN<measure, KeyT, ValueT, GAddrT, BaseT, BAddrT, D, KBuild, KF,
               KQuery, S>
      GGNN;
  GGNN m_ggnn{
      FLAGS_base_filename, FLAGS_query_filename,
      file_exists(FLAGS_groundtruth_filename) ? FLAGS_groundtruth_filename : "",
      L, 0.0f};

  typedef HNSWLoader<ValueT, D, M> HNSWLoader;
  {
    // load HNSW
    HNSWLoader m_hnsw_loader(FLAGS_graph_filename);

    const int N = m_ggnn.ggnn_gpu_instance.N_shard;
    auto& graph_host = m_ggnn.ggnn_gpu_instance.ggnn_cpu_buffers.at(0);
    auto& graph_device = m_ggnn.ggnn_gpu_instance.ggnn_shards.at(0);

    // transfer base-level neighborhood information
    for (size_t n = 0; n < N; ++n) {
      for (size_t k = 0; k < KBuild; ++k) {
        if (m_hnsw_loader.data_level0_memory_.at(n).link_count > k)
          graph_host.h_graph[n * KBuild + k] =
              m_hnsw_loader.data_level0_memory_.at(n).links[k];
        else
          graph_host.h_graph[n * KBuild + k] = -1;
      }
    }

    // FIXME: does HNSW have a useful neighborhood we should load on the top
    // layer?
    for (size_t k = 0; k < KBuild; ++k) {
      graph_host.h_graph[N * KBuild + k] = -1;
    }
    // set starting point
    graph_host.h_translation[0] = m_hnsw_loader.hnsw_header.enterpoint_node_;
    graph_host.h_selection[0] = m_hnsw_loader.hnsw_header.enterpoint_node_;

    // this could be done on the GPU
    float max_nn1_dist = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      max_nn1_dist =
          std::max(max_nn1_dist,
                   m_ggnn.dataset
                       .template compute_distance_base_to_base<measure, ValueT>(
                           n, graph_host.h_graph[n * KBuild]));
    }
    // don't need the mean for querying - just set it to max as well
    graph_host.h_nn1_stats[0] = max_nn1_dist;
    graph_host.h_nn1_stats[1] = max_nn1_dist;

    graph_host.uploadAsync(graph_device);

    CHECK_CUDA(cudaStreamSynchronize(graph_device.stream));
  }

  auto query_function = [&m_ggnn](const float tau_query) {
    cudaMemcpyToSymbol(c_tau_query, &tau_query, sizeof(float));
    LOG(INFO) << "--";
    LOG(INFO) << "Query with tau_query " << tau_query;
    // faster for C@1 = 99%
    LOG(INFO) << "fast query (good for C@1)";
    m_ggnn.queryLayer<32, 200, 256, 64>();
    // better for C@10 > 99%
    LOG(INFO) << "regular query (good for C@10)";
    m_ggnn.queryLayer<32, 400, 448, 64>();
    // expensive, can get to 99.99% C@10
    // m_ggnn.queryLayer<128, 2000, 2048, 256>();
  };

  if (FLAGS_grid_search) {
    LOG(INFO) << "--";
    LOG(INFO) << "grid-search:";
    for (int i = 0; i <= 140; ++i) query_function(i * 0.01f);
  } else {  // by default, just execute a few queries
    LOG(INFO) << "--";
    LOG(INFO) << "90, 95, 99% R@1, 99% C@10 (using -tau 0.5 "
                 "-refinement_iterations 2):";
    query_function(0.34f);
    query_function(0.41f);
    query_function(0.51f);
    query_function(0.64f);
  }

  printf("done! \n");
  gflags::ShutDownCommandLineFlags();
  return 0;
}
