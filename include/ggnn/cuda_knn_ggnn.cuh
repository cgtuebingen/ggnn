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

#ifndef GGNN_GRAPH_OPERATIONS_CUH
#define GGNN_GRAPH_OPERATIONS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <limits>

#include "cub/cub.cuh"
#include "ggnn/cuda_knn_config.cuh"
#include "ggnn/graph/cuda_knn_ggnn_graph.cuh"
#include "ggnn/merge/cuda_knn_merge_layer.cuh"
#include "ggnn/merge/cuda_knn_top_merge_layer.cuh"
#include "ggnn/query/cuda_knn_query_layer.cuh"
#include "ggnn/select/cuda_knn_wrs_select_layer.cuh"
#include "ggnn/sym/cuda_knn_sym_buffer_merge_layer.cuh"
#include "ggnn/sym/cuda_knn_sym_query_layer.cuh"
#include "ggnn/utils/cuda_knn_core_utils.cuh"
#include "ggnn/utils/cuda_knn_dataset.cuh"
#include "ggnn/utils/knn_graph_serialization.hpp"

/**
 * GGNN core operations
 *
 * @param KeyT datatype of dataset indices (needs to be able to represent
 * N_base, signed integer required)
 * @param ValueT distance value type
 * @param GAddrT address type used to access neighborhood vectors (needs to be
 * able to represent N_all*K)
 * @param BaseT datatype of dataset vector elements
 * @param BAddrT address type used to access dataset vectors (needs to be able
 * to represent N_base*D)
 * @param D dimension of dataset
 * @param KBuild neighbors per node in the GGNN graph
 * @param KF maximum number of inverse links per node in the GGNN graph
 * @param KQuery number of nearest neighbors to retrieve during query
 * @param S segment size
 */
template <typename KeyT, typename ValueT, typename GAddrT, typename BaseT,
          typename BAddrT, int D, int KBuild, int KF, int KQuery, int S>
struct GGNN {
  Dataset<KeyT, BaseT, BAddrT> m_dataset;
  GGNNGraph<KeyT, ValueT, GAddrT> m_ggnn_graph;

  /// prng for selection
  curandGenerator_t gen;

  GGNN(const std::string& basePath, const std::string& queryPath,
       const std::string& gtPath, const int L, const float tau_build)
      : m_dataset{basePath, queryPath, gtPath},
        m_ggnn_graph{m_dataset.N_base, L, S, KBuild, KF, tau_build} {
    if (m_dataset.D != D) {
      printf("DIM needs to be the same -> Data: %u != constant: %d \n",
             m_dataset.D, D);
      throw std::runtime_error("DIM needs to be the same");
    }

    /* Create pseudo-random number generator */
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  }

  bool read(const std::string& filename) {
    KNNGraphReader<KeyT, ValueT> graphReader;
    graphReader.open(filename);

    const bool valid = graphReader.verify(
        m_ggnn_graph.N, D, m_ggnn_graph.K, m_ggnn_graph.KF, m_ggnn_graph.L,
        m_ggnn_graph.S, m_ggnn_graph.tau_build);
    bool success = valid;
    if (valid)
      success =
          graphReader.read(m_ggnn_graph.m_graph, m_ggnn_graph.m_translation,
                           m_ggnn_graph.m_selection, m_ggnn_graph.m_nn1_stats,
                           m_ggnn_graph.m_nn1_dist_buffer);
    graphReader.close();

    return success;
  }

  bool write(const std::string& filename, const std::string& comment) const {
    KNNGraphWriter<KeyT, ValueT> graphWriter;
    graphWriter.configure(m_ggnn_graph.N, D, m_ggnn_graph.K, m_ggnn_graph.KF,
                          m_ggnn_graph.L, m_ggnn_graph.S, m_ggnn_graph.tau_build,
                          comment.c_str());
    graphWriter.open(filename);
    const bool success =
        graphWriter.write(m_ggnn_graph.m_graph, m_ggnn_graph.m_translation,
                          m_ggnn_graph.m_selection, m_ggnn_graph.m_nn1_stats,
                          m_ggnn_graph.m_nn1_dist_buffer);
    graphWriter.close();

    return success;
  }

  void evaluateQueryResults(const KeyT* query_results,
                            const int* dist_statistics) const {
    size_t calculated = 0;
    int c1 = 0;
    int cKQuery = 0;

    for (int n = 0; n < m_dataset.N_query; n++) {
      calculated += dist_statistics[n];
      for (int k0 = 0; k0 < KQuery; k0++) {
        KeyT q = query_results[n * KQuery + k0];
        bool found = false;
        for (int k1 = 0; k1 < KQuery && !found; k1++) {
          KeyT gt = m_dataset.m_gt[n * m_dataset.K_gt + k1];
          if (!k0 && !k1 && q == gt) c1++;
          if (q == gt) found = true;
        }
        if (found) cKQuery++;
      }
    }

    printf("c@1: %f \n", c1 / (float)m_dataset.N_query);
    printf("c@%d: %f \n", KQuery,
           cKQuery / (float)(m_dataset.N_query * KQuery));
    lprintf(2, "calculated dists: %f (%zu) \n",
            calculated / (float)m_dataset.N_query, calculated);
  }

  void queryLayer() const {
    typedef QueryKernel<ValueT, KeyT, D, KBuild, KF, KQuery, S, 64, BaseT,
                        BAddrT, GAddrT, true>
        QueryKernel;

    KeyT* m_query_results;
    cudaMallocManaged(&m_query_results,
                      m_dataset.N_query * KQuery * sizeof(KeyT));
    int* m_dist_statistics;
    cudaMallocManaged(&m_dist_statistics, m_dataset.N_query * sizeof(int));

    QueryKernel query_kernel;
    query_kernel.d_base = m_dataset.m_base;
    query_kernel.d_query = m_dataset.m_query;

    query_kernel.d_graph = m_ggnn_graph.m_graph;
    query_kernel.d_query_results = m_query_results;

    query_kernel.d_translation = m_ggnn_graph.m_translation;
    query_kernel.d_selection = m_ggnn_graph.m_selection;

    query_kernel.d_nn1_stats = m_ggnn_graph.m_nn1_stats;

    query_kernel.N = m_dataset.N_query;
    query_kernel.N_offset = 0;

    query_kernel.d_dist_stats = m_dist_statistics;
    query_kernel.d_nn1_buffer = m_ggnn_graph.m_nn1_dist_buffer;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    time_launcher(0, query_kernel, query_kernel.N);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    evaluateQueryResults(m_query_results, m_dist_statistics);

    cudaFree(m_query_results);
    cudaFree(m_dist_statistics);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  };

  void queryLayerFast() const {
    typedef QueryKernel<ValueT, KeyT, D, KBuild, KF, KQuery, S, 64, BaseT,
                        BAddrT, GAddrT, false, false, 150, 256, 192>
        QueryKernel;

    KeyT* m_query_results;
    cudaMallocManaged(&m_query_results,
                      m_dataset.N_query * KQuery * sizeof(KeyT));
    int* m_dist_statistics;
    cudaMallocManaged(&m_dist_statistics, m_dataset.N_query * sizeof(int));

    QueryKernel query_kernel;
    query_kernel.d_base = m_dataset.m_base;
    query_kernel.d_query = m_dataset.m_query;

    query_kernel.d_graph = m_ggnn_graph.m_graph;
    query_kernel.d_query_results = m_query_results;

    query_kernel.d_translation = m_ggnn_graph.m_translation;
    query_kernel.d_selection = m_ggnn_graph.m_selection;

    query_kernel.d_nn1_stats = m_ggnn_graph.m_nn1_stats;

    query_kernel.N = m_dataset.N_query;
    query_kernel.N_offset = 0;

    query_kernel.d_dist_stats = m_dist_statistics;
    query_kernel.d_nn1_buffer = m_ggnn_graph.m_nn1_dist_buffer;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    time_launcher(0, query_kernel, query_kernel.N);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    evaluateQueryResults(m_query_results, m_dist_statistics);

    cudaFree(m_query_results);
    cudaFree(m_dist_statistics);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  };

  void prefetch(const int gpuId) {
    m_dataset.prefetch(gpuId);
    m_ggnn_graph.prefetch(gpuId);
  }

  void select(const int layer) {
    typedef WRSSelectionKernel<ValueT, KeyT, 128, S> SelectionKernel;

    float* d_rng;
    /* Allocate n floats on device */
    cudaMalloc((void**)&d_rng, m_ggnn_graph.getNs(layer) * sizeof(float));

    SelectionKernel select_kernel;

    select_kernel.d_selection = m_ggnn_graph.getSelection(layer + 1);
    select_kernel.d_translation = m_ggnn_graph.getTranslation(layer + 1);

    if (layer)
      select_kernel.d_translation_layer = m_ggnn_graph.getTranslation(layer);

    select_kernel.layer = layer;

    select_kernel.S = m_ggnn_graph.getS(layer);
    select_kernel.S_offset = m_ggnn_graph.getS_offset(layer);

    const int SG = m_ggnn_graph.S / m_ggnn_graph.G;
    const int SG_offset = m_ggnn_graph.S - SG * m_ggnn_graph.G;

    select_kernel.SG = SG;
    select_kernel.SG_offset = SG_offset;

    select_kernel.B = pow(m_ggnn_graph.G, m_ggnn_graph.L - 1 - layer);
    select_kernel.B_offset = 0;

    select_kernel.d_rng = d_rng;
    select_kernel.d_nn1_dist_buffer = m_ggnn_graph.m_nn1_dist_buffer;

    /* Generate n floats on device */
    curandGenerateUniform(gen, d_rng, m_ggnn_graph.getNs(layer));

    time_launcher(2, select_kernel, m_ggnn_graph.getNs(layer));

    cudaFree(d_rng);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  };

  void top(const int layer) {
    typedef TopMergeKernel<ValueT, KeyT, D, KBuild, 128, BaseT, BAddrT, GAddrT>
        TopMergeKernel;

    TopMergeKernel top_kernel;
    top_kernel.d_base = m_dataset.m_base;
    if (layer) top_kernel.d_translation = m_ggnn_graph.getTranslation(layer);
    top_kernel.d_graph = m_ggnn_graph.getGraph(layer);
    top_kernel.d_nn1_dist_buffer = m_ggnn_graph.m_nn1_dist_buffer;

    top_kernel.layer = layer;

    top_kernel.N = m_ggnn_graph.getNs(layer);
    top_kernel.N_offset = 0;

    top_kernel.S = m_ggnn_graph.getS(layer);
    top_kernel.S_offset = m_ggnn_graph.getS_offset(layer);

    time_launcher(2, &top_kernel, m_ggnn_graph.getNs(layer));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  };

  void mergeLayer(const int layer_top, const int layer_btm) {
    typedef MergeKernel<ValueT, KeyT, D, KBuild, KF, S, 64, BaseT,
                        BAddrT, GAddrT>
        MergeKernel;

    size_t graph_buffer_size =
        static_cast<GAddrT>(m_ggnn_graph.getNs(layer_btm)) * KBuild *
        sizeof(KeyT);

    // buffer for the new neighborhoods created by the merge
    KeyT* d_graph_buffer;
    cudaMalloc(&d_graph_buffer, graph_buffer_size);

    MergeKernel merge_kernel;
    merge_kernel.d_base = m_dataset.m_base;

    merge_kernel.d_graph = m_ggnn_graph.m_graph;
    merge_kernel.d_graph_buffer = d_graph_buffer;

    merge_kernel.d_translation = m_ggnn_graph.m_translation;
    merge_kernel.d_selection = m_ggnn_graph.m_selection;

    merge_kernel.d_nn1_stats = m_ggnn_graph.m_nn1_stats;
    merge_kernel.d_nn1_dist_buffer = m_ggnn_graph.m_nn1_dist_buffer;

    merge_kernel.N = m_ggnn_graph.getNs(layer_btm);
    merge_kernel.N_offset = 0;

    merge_kernel.layer_top = layer_top;
    merge_kernel.layer_btm = layer_btm;

    time_launcher(2, merge_kernel, m_ggnn_graph.getNs(layer_btm));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaMemcpy((void*)m_ggnn_graph.getGraph(layer_btm), (void*)d_graph_buffer,
               graph_buffer_size, cudaMemcpyDeviceToDevice);

    cudaFree(d_graph_buffer);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  };

  void merge(const int layer_top, const int layer_btm) {
    lprintf(2, "merge: %d %d \n", layer_top, layer_btm);
    if (layer_top == layer_btm)
      top(layer_btm);
    else
      mergeLayer(layer_top, layer_btm);

    if (!layer_btm) m_ggnn_graph.computeNN1Stats();
  };

  void sym(const int layer) {
    typedef SymQueryKernel<ValueT, KeyT, D, KBuild, KF, 64, BaseT,
                           BAddrT, GAddrT>
        SymQueryKernel;

    KeyT* d_sym_buffer;
    cudaMalloc(&d_sym_buffer, static_cast<GAddrT>(m_ggnn_graph.getNs(layer)) *
                                  KF * sizeof(KeyT));
    cudaMemset(
        d_sym_buffer, -1,
        static_cast<GAddrT>(m_ggnn_graph.getNs(layer)) * KF * sizeof(KeyT));

    KeyT* m_sym_atomic;
    cudaMallocManaged(&m_sym_atomic, m_ggnn_graph.N * sizeof(KeyT));
    cudaMemset(m_sym_atomic, 0, m_ggnn_graph.getNs(layer) * sizeof(KeyT));

    int* m_statistics;
    cudaMallocManaged(&m_statistics, m_ggnn_graph.N * sizeof(int));

    SymQueryKernel sym_kernel;

    sym_kernel.d_base = m_dataset.m_base;
    sym_kernel.d_graph = m_ggnn_graph.getGraph(layer);
    if (layer) sym_kernel.d_translation = m_ggnn_graph.getTranslation(layer);

    sym_kernel.d_sym_atomic = m_sym_atomic;
    sym_kernel.d_sym_buffer = d_sym_buffer;

    sym_kernel.d_nn1_stats = m_ggnn_graph.m_nn1_stats;
    sym_kernel.d_stats = m_statistics;

    sym_kernel.layer = layer;

    sym_kernel.N = m_ggnn_graph.getNs(layer);

    sym_kernel.N_offset = 0;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    time_launcher(2, sym_kernel, m_ggnn_graph.getNs(layer));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    typedef SymBufferMergeKernel<ValueT, KeyT, KBuild, KF, 128, GAddrT>
        SymBufferMergeKernel;
    SymBufferMergeKernel sym_buffer_merge_kernel;

    sym_buffer_merge_kernel.d_sym_buffer = d_sym_buffer;
    sym_buffer_merge_kernel.d_sym_atomic = m_sym_atomic;
    sym_buffer_merge_kernel.d_graph = m_ggnn_graph.getGraph(layer);

    sym_buffer_merge_kernel.N = m_ggnn_graph.getNs(layer);
    sym_buffer_merge_kernel.N_offset = 0;

    time_launcher(3, sym_buffer_merge_kernel, m_ggnn_graph.getNs(layer));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    int c = 0;
    int m = 0;
    int unconnected = 0;
    for (int i = 0; i < m_ggnn_graph.getNs(layer); i++) {
      if (m_sym_atomic[i] > KF) c++;
      m += (m_sym_atomic[i] > KF) ? KF : m_sym_atomic[i];
      unconnected += m_statistics[i];
    }
    lprintf(2,
            "Layer %d [N: %d] | overflow: %d (%f) | added_links: %d (%f) || "
            "unconnected: %d "
            "(%f)\n",
            layer, m_ggnn_graph.getNs(layer), c,
            c / (float)m_ggnn_graph.getNs(layer), m,
            m / (float)m_ggnn_graph.getNs(layer), unconnected,
            unconnected / (float)m_ggnn_graph.getNs(layer));

    cudaFree(d_sym_buffer);
    cudaFree(m_sym_atomic);
    cudaFree(m_statistics);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  };

  void build_bubble_merge() {
    for (int layer_top = 0; layer_top < m_ggnn_graph.L; layer_top++) {
      for (int layer_btm = layer_top; layer_btm >= 0; layer_btm--) {
        lprintf(2, "layer_top: %d -> layer_btm: %d \n", layer_top, layer_btm);

        merge(layer_top, layer_btm);

        if (layer_top < (m_ggnn_graph.L - 1) && layer_top == layer_btm)
          select(layer_top);

        sym(layer_btm);
      }
    }
  }
  void build_simple_merge() {
    for (int layer = 0; layer < m_ggnn_graph.L - 1; layer++) {
      select(layer);
      merge(layer, layer);
      sym(layer);
    }
    for (int layer = m_ggnn_graph.L - 1; layer >= 0; layer--) {
      merge(m_ggnn_graph.L - 1, layer);
      sym(layer);
    }
  }

  void refine() {
    for (int layer = m_ggnn_graph.L - 2; layer >= 0; layer--) {
      merge(m_ggnn_graph.L - 1, layer);
      sym(layer);
    }
  }
};

#endif  // GGNN_GRAPH_OPERATIONS_CUH
