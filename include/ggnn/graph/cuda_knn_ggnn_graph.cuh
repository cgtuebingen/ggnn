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

#ifndef GGNN_GRAPH_CUH
#define GGNN_GRAPH_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <limits>

#include "cub/cub.cuh"
#include "ggnn/utils/cuda_knn_core_utils.cuh"

/**
 * GGNN graph data
 *
 * @param KeyT datatype of dataset indices
 * @param ValueT distance value type
 * @param GAddrT address type used to access neighborhood vectors (needs to be
 * able to represent N_all*K)
 */
template <typename KeyT, typename ValueT, typename GAddrT>
struct GGNNGraph {
  /// number of base points
  int N;
  /// number of layers
  int L;
  /// growth factor (number of sub-graphs merged together per layer)
  int G;
  /// segment size
  int S;
  /// segment size in base layer
  int S0;
  /// number of segments in base layer with one additional element
  int S0_off;
  /// number of neighbors
  int K;
  /// maximum number of inverse links
  int KF;
  /// slack factor for symmetric linking
  float tau_build;

  /// total number of neighborhoods in the graph
  int N_all;
  /// total number of selection/translation entries
  int ST_all;

  /// neighborhoods per layer
  std::vector<int> Ns;  //[L]
  /// start of neighborhoods per layer
  std::vector<int> Ns_offsets;  //[L]
  /// start of selection/translation per layer
  std::vector<int> STs_offsets;  //[L]

  /// neighborhood vectors
  KeyT* m_graph;
  /// translation of upper layer points into lowest layer
  KeyT* m_translation;
  /// translation of upper layer points into one layer below
  KeyT* m_selection;

  /// distance to nearest known neighbor per point
  ValueT* m_nn1_dist_buffer;
  /// average and maximum distance to nearest known neighbors
  ValueT* m_nn1_stats;

  GGNNGraph(const int N, const int L, const int S, const int K, const int KF,
            const float tau_build)
      : N{N},
        L{L},
        S{S},
        K{K},
        KF{KF},
        tau_build{tau_build},
        Ns(L),
        Ns_offsets(L),
        STs_offsets(L) {
    /// theoretical growth factor (number of sub-graphs merged together per
    /// layer)
    const float growth = powf((float)N / (float)S, 1.f / (L - 1));

    const int Gf = growth;
    const int Gc = growth + 1;

    const float S0f = N / (pow(Gf, (L - 1)));
    const float S0c = N / (pow(Gc, (L - 1)));

    const bool is_floor =
        (growth > 0) && ((S0c < K) || (fabs(S0f - S) < fabs(S0c - S)));

    G = (is_floor) ? Gf : Gc;
    S0 = (is_floor) ? S0f : S0c;
    S0_off = N - pow(G, L - 1) * S0;

    lprintf(1,
            "GGNNGraph(): N: %d, L: %d, G: %d, S: %d, S0: %d, S0_off: %d, K: "
            "%d, KF: %d\n",
            N, L, G, S, S0, S0_off, K, KF);

    N_all = 0;
    ST_all = 0;
    for (int l = 0; l < L; l++) {
      int N_current = (!l) ? N : S * powf(G, L - l - 1);
      Ns[l] = N_current;
      Ns_offsets[l] = N_all;
      STs_offsets[l] = ST_all;
      N_all += N_current;
      if (l) ST_all += N_current;
    }

    if (static_cast<size_t>(N_all) * static_cast<size_t>(K) >
        static_cast<size_t>(std::numeric_limits<GAddrT>::max()))
      throw std::runtime_error(
          "GGNNGraph(): address type is insufficient to address the requested "
          "graph. aborting.\n");

    lprintf(2, "GGNNGraph(): allocating memory...\n");
    cudaError_t result;
    result = cudaMallocManaged(&m_graph,
                               static_cast<GAddrT>(N_all) * K * sizeof(KeyT));
    if (result != cudaSuccess)
      throw std::runtime_error(
          "GGNNGraph(): failed to allocate memory for graph.\n");
    result = cudaMallocManaged(&m_translation, ST_all * sizeof(KeyT));
    if (result != cudaSuccess)
      throw std::runtime_error(
          "GGNNGraph(): failed to allocate memory for translation.\n");
    result = cudaMallocManaged(&m_selection, ST_all * sizeof(KeyT));
    if (result != cudaSuccess)
      throw std::runtime_error(
          "GGNNGraph(): failed to allocate memory for selection.\n");
    result = cudaMallocManaged(&m_nn1_dist_buffer, N * sizeof(ValueT));
    if (result != cudaSuccess)
      throw std::runtime_error(
          "GGNNGraph(): failed to allocate memory for first neighbor "
          "distances.\n");
    result = cudaMallocManaged(&m_nn1_stats, 2 * sizeof(ValueT));
    if (result != cudaSuccess)
      throw std::runtime_error(
          "GGNNGraph(): failed to allocate memory for first neighbor distance "
          "statistics.\n");

    copyConstantsToGPU();

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    lprintf(2, "GGNNGraph(): done.\n");
  }

  ~GGNNGraph() {
    cudaFree(m_graph);
    cudaFree(m_translation);
    cudaFree(m_selection);
    cudaFree(m_nn1_dist_buffer);
    cudaFree(m_nn1_stats);
  }

  void copyConstantsToGPU() const {
    lprintf(2, "GGNNGraph::copyConstantsToGPU().\n");

    cudaMemcpyToSymbol(c_Ns, Ns.data(), L * sizeof(int));
    cudaMemcpyToSymbol(c_Ns_offsets, Ns_offsets.data(), L * sizeof(int));

    cudaMemcpyToSymbol(c_G, &G, sizeof(int));
    cudaMemcpyToSymbol(c_L, &L, sizeof(int));
    cudaMemcpyToSymbol(c_S0, &S0, sizeof(int));
    cudaMemcpyToSymbol(c_S0_offset, &S0_off, sizeof(int));

    cudaMemcpyToSymbol(c_tau_build, &tau_build, sizeof(float));
    cudaMemcpyToSymbol(c_STs_offsets, STs_offsets.data(), L * sizeof(int));
  }

  KeyT* getGraph(const int layer) {
    return &m_graph[static_cast<GAddrT>(Ns_offsets[layer]) * K];
  }

  KeyT* getSelection(const int layer) {
    if (!layer) {
      printf("There is no selection for layer 0 \n");
      return nullptr;
    }
    return &m_selection[STs_offsets[layer]];
  }

  KeyT* getTranslation(const int layer) {
    if (!layer) {
      printf("There is no translation for layer 0 \n");
      return nullptr;
    }
    return &m_translation[STs_offsets[layer]];
  }

  int getNs(const int layer) const { return Ns[layer]; }

  int getS(const int layer) const { return layer ? S : S0; }

  int getS_offset(const int layer) const { return layer ? 0 : S0_off; }

  KeyT translate(const int layer, KeyT n) const {
    return (!layer) ? n : getTranslation(layer)[n];
  }

  void printGraph(int layer, int len = -1, int skip = 1) const {
    printf("Graph Layer %d: \n", layer);
    const size_t skip_len = len * skip;
    for (int n = 0; n < ((len > 0) ? skip_len : Ns[layer]); n += skip) {
      printf("%d (%d) -> ", n, translate(layer, n));
      for (int k = 0; k < K; k++) {
        KeyT other_n = getGraph(layer)[n * K + k];
        printf(" %d (%d) |", other_n, translate(layer, other_n));
      }
      printf("\n");
    }
  }

  bool checkGraph(int layer) const {
    printf("Check Graph Layer %d: \n", layer);
    for (int n = 0; n < Ns[layer]; n++) {
      for (int k = 0; k < K; k++) {
        KeyT other_n = getGraph(layer)[n * K + k];
        if (other_n < 0 || other_n >= Ns[layer]) {
          printf("problem on %d [%d] -> %d \n", n, k, other_n);
          return true;
        }
      }
      return false;
    }
  }

  bool checkDuplicates(int layer) const {
    printf("Check Duplicates Graph Layer %d: \n", layer);
    for (int n = 0; n < Ns[layer]; n++) {
      for (int k0 = 0; k0 < K; k0++) {
        KeyT other_k0 = getGraph(layer)[n * K + k0];
        for (int k1 = 0; k1 < K; k1++) {
          if (k0 != k1) {
            KeyT other_k1 = getGraph(layer)[n * K + k1];
            if (other_k0 == other_k1) {
              printf("duplicate problem on %d [%d] -> %d vs %d \n", n, k0,
                     other_k0, other_k1);
              return true;
            }
          }
        }
      }
      return false;
    }
  }

  void printSelection(int layer) const {
    printf("selection %d: \n", layer);
    if (!layer) printf("no selection for layer 0 \n");

    for (int n = 0; n < Ns[layer]; n++) {
      printf("%d -> %d \n", n, getSelection(layer)[n]);
    }
  }

  void printNN1DistBuffer() const {
    for (int n = 0; n < N; n++) {
      printf("%d (%f): ", n, m_nn1_dist_buffer[n]);
      for (int k = 0; k < K; k++) {
        printf("%d ", m_graph[n * K + k]);
      }
      printf("\n");
    }
  }

  void computeNN1Stats() {
    void* d_temp_storage = NULL;

    size_t temp_storage_bytes_sum = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes_sum,
                           m_nn1_dist_buffer, &m_nn1_stats[0], N);

    size_t temp_storage_bytes_max = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes_max,
                           m_nn1_dist_buffer, &m_nn1_stats[1], N);

    size_t temp_storage_bytes =
        std::max(temp_storage_bytes_sum, temp_storage_bytes_max);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           m_nn1_dist_buffer, &m_nn1_stats[0], N);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           m_nn1_dist_buffer, &m_nn1_stats[1], N);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    m_nn1_stats[0] = m_nn1_stats[0] / (float)N;
    lprintf(2, "mean: %f | max: %f \n", m_nn1_stats[0], m_nn1_stats[1]);

    cudaFree(d_temp_storage);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  }

  // TODO(fabi): remove?
  void prefetch(int gpuId) const {
    lprintf(1, "GGNNGraph::prefetch() to GPU %d.\n", gpuId);

    // push graph data to the gpu
    cudaMemAdvise(m_graph, static_cast<GAddrT>(N_all) * K * sizeof(KeyT),
                  cudaMemAdviseSetAccessedBy, gpuId);
    cudaMemAdvise(m_translation, ST_all * sizeof(KeyT),
                  cudaMemAdviseSetAccessedBy, gpuId);
    cudaMemAdvise(m_selection, ST_all * sizeof(KeyT),
                  cudaMemAdviseSetAccessedBy, gpuId);
    cudaMemAdvise(m_nn1_stats, 2 * sizeof(ValueT), cudaMemAdviseSetAccessedBy,
                  gpuId);
    cudaMemAdvise(m_nn1_dist_buffer, N * sizeof(ValueT),
                  cudaMemAdviseSetAccessedBy, gpuId);

    cudaMemAdvise(m_graph, static_cast<GAddrT>(N_all) * K * sizeof(KeyT),
                  cudaMemAdviseSetReadMostly, gpuId);
    cudaMemAdvise(m_translation, ST_all * sizeof(KeyT),
                  cudaMemAdviseSetReadMostly, gpuId);
    cudaMemAdvise(m_selection, ST_all * sizeof(KeyT),
                  cudaMemAdviseSetReadMostly, gpuId);
    cudaMemAdvise(m_nn1_stats, 2 * sizeof(ValueT), cudaMemAdviseSetReadMostly,
                  gpuId);
    cudaMemAdvise(m_nn1_dist_buffer, N * sizeof(ValueT),
                  cudaMemAdviseSetReadMostly, gpuId);

    cudaMemPrefetchAsync(m_graph, static_cast<GAddrT>(N_all) * K * sizeof(KeyT),
                         gpuId);
    cudaMemPrefetchAsync(m_translation, ST_all * sizeof(KeyT), gpuId);
    cudaMemPrefetchAsync(m_selection, ST_all * sizeof(KeyT), gpuId);
    cudaMemPrefetchAsync(m_nn1_stats, 2 * sizeof(ValueT), gpuId);
    cudaMemPrefetchAsync(m_nn1_dist_buffer, N * sizeof(ValueT), gpuId);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  }
};

#endif  // GGNN_GRAPH_CUH
