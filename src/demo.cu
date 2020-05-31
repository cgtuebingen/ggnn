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
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <stdio.h>

#include <vector>

#include "cub/cub.cuh"
#include "ggnn/config.hpp"
#include "ggnn/cuda_knn_config.cuh"
#include "ggnn/cuda_knn_ggnn.cuh"
#include "ggnn/graph/cuda_knn_ggnn_graph.cuh"
#include "ggnn/utils/cuda_knn_core_utils.cuh"
#include "ggnn/utils/cuda_knn_dataset.cuh"

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  printf("GGNN: Graph-based GPU Nearest Neighbor Search\n");
  printf(
      "by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. "
      "Lensch\n");
  printf("(c) 2020 Computer Graphics University of Tuebingen\n");
  printf("\n");

  // ####################################################################
  // compile-time configuration
  // ####################################################################
  // build configuration
  const int KBuild = 24;
  const int KF = KBuild / 2;
  const int S = 32;
  const int L = 4;
  const bool bubble_merge = true;

  // query configuration
  const int KQuery = 10;

  // dataset configuration (here: SIFT1M)
  const int D = 128;
  typedef int32_t KeyT;
  typedef float BaseT;
  typedef float ValueT;
  typedef uint32_t BAddrT;
  typedef uint32_t GAddrT;

  // print compile time configuration
  {
    printf("compile-time configuration:\n");
    printf("KBuild: %d, ", KBuild);
    printf("KF: %d, ", KF);
    printf("S: %d, ", S);
    printf("L: %d, ", L);
    printf("D: %d\n", D);
#define PRINT_TYPE(T)                                                        \
  {                                                                          \
    if (std::is_arithmetic<T>::value)                                        \
      printf("%s: %s %s (%zu)\n", #T,                                        \
             std::is_signed<T>::value ? "signed" : "unsigned",               \
             std::is_floating_point<T>::value ? "float" : "int", sizeof(T)); \
    else                                                                     \
      printf("%s: %s\n", #T, typeid(T).name());                              \
  }
    PRINT_TYPE(KeyT);
    PRINT_TYPE(BaseT);
    PRINT_TYPE(ValueT);
    PRINT_TYPE(BAddrT);
    PRINT_TYPE(GAddrT);
    printf("\n");
#undef PRINT_TYPE
  }

  // ####################################################################
  // parameter parsing
  // ####################################################################
  // check argument count
  {
    // TODO: proper argument parsing
    if (argc < 6) {
      printf("usage:\n%s", argv[0]);
      printf(" <base.xvecs> <query.xvecs> <gt.ivecs>");
      printf(" <tau_build> <refinement_iterations>");
      printf(" [<GPU id>] [<graph_cache.ggnn>]\n");

      return -1;
    }
  }

  const std::string basePath{argv[1]};
  const std::string queryPath{argv[2]};
  const std::string gtPath{argv[3]};
  // check files
  {
    bool filesExist = false;
    if (!exists(basePath))
      printf("base file %s does not exist.\n", basePath.c_str());
    else if (!exists(queryPath))
      printf("query file %s does not exist.\n", queryPath.c_str());
    else if (!exists(gtPath))
      printf("ground truth file %s does not exist.\n", gtPath.c_str());
    else
      filesExist = true;
    if (!filesExist) return -1;
  }
  const float tau_build = strtod(argv[4], nullptr);
  const int refinement_iterations = strtol(argv[5], nullptr, 10);

  int gpuId = (argc > 6 ? strtol(argv[6], nullptr, 10) : 0);

  const std::string graph_file{argc > 7 ? argv[7] : ""};
  const bool export_graph = !graph_file.empty() && !exists(graph_file);
  const bool import_graph = !graph_file.empty() && exists(graph_file);

  // print parsed parameters for verification
  printf("parsed parameters:\n");
  printf("base: %s\n", basePath.c_str());
  printf("queries: %s\n", queryPath.c_str());
  printf("ground truth: %s\n", gtPath.c_str());
  printf("tau_build: %g\n", tau_build);
  printf("refinement iterations: %d\n", refinement_iterations);
  printf("GPU id: %d\n", gpuId);
  printf("graph file: %s\n", graph_file.c_str());

  // set the requested GPU id, if possible
  {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    if (numGpus <= gpuId) {
      printf("GPU %d does not exist, using GPU 0 instead.\n", gpuId);
      gpuId = 0;
    }
  }
  cudaSetDevice(gpuId);

  printf("\n");

  const bool build = export_graph || !import_graph;
  const bool query = true;

  // print build/query and import/export configuration
  if (build) {
    printf("graph will be built.\n");
    if (export_graph)
      printf("graph will be exported to: %s\n", graph_file.c_str());
  } else
    printf("graph will NOT be built.\n");
  if (query) {
    if (import_graph)
      printf("graph will be imported from: %s\n", graph_file.c_str());
    printf("graph will be queried.\n");
  } else
    printf("graph will NOT be queried.\n");
  printf("\n");

  // ####################################################################
  // load dataset and allocate graph memory
  // ####################################################################

  typedef GGNN<KeyT, ValueT, GAddrT, BaseT, BAddrT, D, KBuild, KF, KQuery, S>
      GGNN;
  GGNN m_ggnn{basePath, queryPath, gtPath, L, tau_build};

  // ####################################################################
  // build graph
  // ####################################################################
  if (build) {
    std::vector<float> construction_times;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\nStarting Graph construction...\n");

    cudaEventRecord(start);

    if (bubble_merge)
      m_ggnn.build_bubble_merge();
    else
      m_ggnn.build_simple_merge();

    cudaEventRecord(stop);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    construction_times.push_back(milliseconds);

    for (int i = 0; i < refinement_iterations; i++) {
      lprintf(1, "\nRefinment step: %d \n", i);
      m_ggnn.refine();

      cudaEventRecord(stop);

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaPeekAtLastError());

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      construction_times.push_back(milliseconds);
    }

    printf("\nGraph construction -- secs: %f || %d points -> %f ms/point \n",
           construction_times[0] / 1000.f, m_ggnn.m_ggnn_graph.N,
           construction_times[0] / m_ggnn.m_ggnn_graph.N);

    for (int i = 1; i < construction_times.size() - 1; i++) {
      float milliseconds = construction_times[i];
      lprintf(1,
              "Graph and %d refinement steps included -- seconds: %f || %d "
              "points -> %f ms/point \n",
              i, milliseconds / 1000.f, m_ggnn.m_ggnn_graph.N,
              milliseconds / m_ggnn.m_ggnn_graph.N);
    }
    if (refinement_iterations) {
      printf(
          "Graph and %d refinement steps included -- seconds: %f || %d points "
          "-> %f ms/point \n",
          refinement_iterations,
          construction_times[construction_times.size() - 1] / 1000.f,
          m_ggnn.m_ggnn_graph.N,
          construction_times[construction_times.size() - 1] /
              m_ggnn.m_ggnn_graph.N);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (export_graph) {
      m_ggnn.write(graph_file, basePath);
    }
  }

  // ####################################################################
  // query graph
  // ####################################################################
  if (query) {
    if (import_graph) {
      m_ggnn.read(graph_file);
    }

    m_ggnn.prefetch(gpuId);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    auto query = [&m_ggnn](const float tau_query) {
      cudaMemcpyToSymbol(c_tau_query, &tau_query, sizeof(float));
      printf("\nQuery with tau_query: %.2f \n", tau_query);

      m_ggnn.queryLayer();
      m_ggnn.queryLayerFast();
    };

    query(0.35f);
    query(0.42f);
    query(0.60f);
  }

  printf("done! \n");

  return 0;
}
