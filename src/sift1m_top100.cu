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

DEFINE_string(base_filename, "", "path to file with base vectors");
DEFINE_string(query_filename, "", "path to file with perform_query vectors");
DEFINE_string(groundtruth_filename, "",
              "path to file with groundtruth vectors");
DEFINE_string(graph_filename, "",
              "path to file that contains the serialized graph");
DEFINE_double(tau, 0.5, "Parameter tau");
DEFINE_int32(refinement_iterations, 2, "Number of refinement iterations");
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

  CHECK_GE(FLAGS_tau, 0) << "Tau has to be bigger or equal 0.";
  CHECK_GE(FLAGS_refinement_iterations, 0)
      << "The number of refinement iterations has to be non-negative.";

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
  // search-graph configuration
  //
  /// number of neighbors per point in the graph
  const int KBuild = 40;
  /// maximum number of inverse/symmetric links (KBuild / 2 usually works best)
  const int KF = KBuild / 2;
  /// segment/batch size (needs to be > KBuild-KF)
  const int S = 32;
  /// graph height / number of layers (4 usually performs best)
  const int L = 4;
  //
  // query configuration
  //
  /// number of neighbors to search for
  const int KQuery = 100;

  static_assert(KBuild - KF < S,
                "there are not enough points to fill the local neighbor list!");

  LOG(INFO) << "Using the following parameters " << KBuild << " (KBuild) " << KF
            << " (KF) " << S << " (S) " << L << " (L) " << D << " (D) ";

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
  GGNN m_ggnn{FLAGS_base_filename, FLAGS_query_filename,
              FLAGS_groundtruth_filename, L, static_cast<float>(FLAGS_tau)};

  m_ggnn.ggnnMain(FLAGS_graph_filename, FLAGS_refinement_iterations);

  auto query_function = [&m_ggnn](const float tau_query) {
    cudaMemcpyToSymbol(c_tau_query, &tau_query, sizeof(float));
    LOG(INFO) << "--";
    LOG(INFO) << "Query with tau_query " << tau_query;
    // faster for C@1 = 99%
    // LOG(INFO) << "fast query (good for C@1)";
    // m_ggnn.queryLayer<32, 200, 256, 128>();
    // better for C@10 > 99%
    LOG(INFO) << "regular query (good for C@10)";
    m_ggnn.queryLayer<32, 400, 448, 128>();
    LOG(INFO) << "extended query (good for C@100)";
    m_ggnn.queryLayer<64, 1000, 1024, 128>();
    // expensive, can get to 99.99% C@10
    // LOG(INFO) << "expensive query";
    // m_ggnn.queryLayer<128, 2000, 2048, 256>();
  };

  if (FLAGS_grid_search) {
    LOG(INFO) << "--";
    LOG(INFO) << "grid-search:";
    for (int i = 0; i <= 100; ++i) query_function(i * 0.01f);
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
