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
  const int KBuild = 24;
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
  const int KQuery = 10;

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

  const float tau_query = 0.0f;
  cudaMemcpyToSymbol(c_tau_query, &tau_query, sizeof(float));

#define QUERY_BEST_SIZE(best_size)                              \
  {                                                             \
    LOG(INFO) << "Query with best size " << best_size;          \
    m_ggnn.noSlackQueryLayer<32, 400, 448 + best_size - 10,     \
                             64 + best_size - 10, best_size>(); \
  }
  QUERY_BEST_SIZE(10);
  QUERY_BEST_SIZE(20);
  QUERY_BEST_SIZE(30);
  QUERY_BEST_SIZE(40);
  QUERY_BEST_SIZE(50);
  QUERY_BEST_SIZE(60);
  QUERY_BEST_SIZE(70);
  QUERY_BEST_SIZE(80);
  QUERY_BEST_SIZE(90);
  QUERY_BEST_SIZE(100);
  QUERY_BEST_SIZE(110);
  QUERY_BEST_SIZE(120);
  QUERY_BEST_SIZE(130);
  QUERY_BEST_SIZE(140);
  QUERY_BEST_SIZE(150);
  QUERY_BEST_SIZE(160);
  QUERY_BEST_SIZE(170);
  QUERY_BEST_SIZE(180);
  QUERY_BEST_SIZE(190);
  QUERY_BEST_SIZE(200);
  QUERY_BEST_SIZE(220);
  QUERY_BEST_SIZE(240);
  QUERY_BEST_SIZE(260);
  QUERY_BEST_SIZE(280);
  QUERY_BEST_SIZE(300);
  QUERY_BEST_SIZE(320);
  QUERY_BEST_SIZE(340);
  QUERY_BEST_SIZE(360);
  QUERY_BEST_SIZE(380);
  QUERY_BEST_SIZE(400);
  QUERY_BEST_SIZE(450);
  QUERY_BEST_SIZE(500);
  QUERY_BEST_SIZE(550);
  QUERY_BEST_SIZE(600);
  QUERY_BEST_SIZE(700);
  QUERY_BEST_SIZE(800);

  printf("done! \n");
  gflags::ShutDownCommandLineFlags();
  return 0;
}
