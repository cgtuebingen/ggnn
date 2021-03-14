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
DEFINE_double(tau, 0.5, "Parameter tau");
DEFINE_int32(refinement_iterations, 2, "Number of refinement iterations");
DEFINE_int32(gpu_id, 0, "GPU id");

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
  using BaseT = uint8_t;
  /// data type of computed distances
  using ValueT = float;
  /// data type for addressing base-vectors (needs to be able to represent N*D)
  using BAddrT = uint64_t;
  /// data type for addressing the graph (needs to be able to represent
  /// N*KBuild)
  using GAddrT = uint64_t;
  //
  // dataset configuration (here: SIFT1B)
  //
  /// dimension of the dataset
  const int D = 128;
  /// distance measure (Euclidean or Cosine)
  const DistanceMeasure measure = Euclidean;
  //
  // search-graph configuration
  //
  /// number of neighbors per point in the graph
  const int KBuild = 20;
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
  GGNN m_ggnn{FLAGS_base_filename,
              FLAGS_query_filename,
              "",
              L,
              static_cast<float>(FLAGS_tau),
              100000000};

  for (KeyT n = 10000000; n <= m_ggnn.dataset.N_base; n += 10000000) {
    LOG(INFO) << "Constructing graph for " << n << " points.";
    m_ggnn.reinit_graph_for_subset(n);
    m_ggnn.generateGTUsingBF();

    {
      std::vector<float> construction_times;

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      LOG(INFO) << "Starting Graph construction... (tau=" << FLAGS_tau << ")";

      cudaEventRecord(start);
      m_ggnn.build();
      cudaEventRecord(stop);

      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaPeekAtLastError());

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      construction_times.push_back(milliseconds);

      for (int refinement_step = 0;
           refinement_step < FLAGS_refinement_iterations; ++refinement_step) {
        DLOG(INFO) << "Refinement step " << refinement_step;
        m_ggnn.refine();

        cudaEventRecord(stop);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaPeekAtLastError());
        cudaEventSynchronize(stop);

        float elapsed_milliseconds = 0;
        cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
        construction_times.push_back(elapsed_milliseconds);
      }
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      for (int refinement_step = 0; refinement_step < construction_times.size();
           refinement_step++) {
        const float elapsed_milliseconds = construction_times[refinement_step];
        const float elapsed_seconds = elapsed_milliseconds / 1000.0f;
        const int number_of_points = m_ggnn.ggnn_gpu_instance.N_shard;

        LOG(INFO) << "Graph construction + " << refinement_step
                  << " refinement step(s)";
        LOG(INFO) << "                   -- secs: " << elapsed_seconds;
        LOG(INFO) << "                   -- points: " << number_of_points;
        LOG(INFO) << "                   -- ms/point: "
                  << elapsed_milliseconds / number_of_points;
      }
    }

    {
      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaPeekAtLastError());

      auto query_function = [&m_ggnn](const float tau_query) {
        cudaMemcpyToSymbol(c_tau_query, &tau_query, sizeof(float));
        LOG(INFO) << "--";
        LOG(INFO) << "Query with tau_query " << tau_query;
        // faster for C@1 = 99%
        // LOG(INFO) << "fast query (good for C@1)";
        // m_ggnn.queryLayer<32, 200, 256, 64>();
        // better for C@10 > 99%
        LOG(INFO) << "regular query (good for C@10)";
        m_ggnn.queryLayer<32, 400, 448, 64>();
        // expensive, can get to 99.99% C@10
        // m_ggnn.queryLayer<128, 2000, 2048, 256>();
      };

      {  // by default, just execute a few queries
        LOG(INFO) << "--";
        LOG(INFO) << "90, 95, 99% R@1, 99% C@10 (using -tau 0.5 "
                     "-refinement_iterations 2):";
        query_function(0.34f);
        query_function(0.41f);
        query_function(0.51f);
        query_function(0.64f);
      }
    }
  }

  printf("done! \n");
  gflags::ShutDownCommandLineFlags();
  return 0;
}
