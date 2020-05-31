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

#ifndef DATASET_CUH
#define DATASET_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <limits>

#include "ggnn/config.hpp"
#include "io/loader_ann.hpp"
#include "io/storer_ann.hpp"

/**
 * KNN database data that will be shared with the GPU
 * and some utilities to load (and store) that data
 *
 * @param KeyT datatype of dataset indices
 * @param BaseT datatype of dataset vector elements
 * @param BAddrT address type used to access dataset vectors (needs to be able
 * to represent N_base*D)
 */
template <typename KeyT, typename BaseT, typename BAddrT>
struct Dataset {
  /// dataset vectors
  BaseT* m_base{nullptr};
  /// query vectors
  BaseT* m_query{nullptr};
  /// ground truth indices in the dataset for the given queries
  KeyT* m_gt{nullptr};

  /// number of dataset vectors
  int N_base{0};
  /// number of query vectors (and ground truth indices)
  int N_query{0};
  /// dimension of vectors in the dataset and query
  int D{0};
  /// number of nearest neighbors per ground truth entry
  int K_gt{0};

  Dataset(const std::string& basePath, const std::string& queryPath,
          const std::string& gtPath) {
    bool success = loadBase(basePath) && loadQuery(queryPath) && loadGT(gtPath);

    if (!success)
      throw std::runtime_error(
          "failed to load dataset (see previous log entries for details).\n");
  }

  ~Dataset() {
    cudaFree(m_base);
    cudaFree(m_query);
    cudaFree(m_gt);
  }

  void freeBase() {
    cudaFree(m_base);
    m_base = nullptr;
    N_base = 0;
    if (!m_query) D = 0;
  }

  void freeQuery() {
    cudaFree(m_query);
    m_query = nullptr;
    if (!m_gt) N_query = 0;
    if (!m_base) D = 0;
  }

  void freeGT() {
    cudaFree(m_gt);
    m_gt = nullptr;
    if (!m_query) N_query = 0;
    K_gt = 0;
  }

  /// load base vectors from file
  bool loadBase(const std::string& base_file, KeyT from = 0,
                KeyT num = std::numeric_limits<KeyT>::max()) {
    freeBase();
    XVecsLoader<BaseT> base_loader(base_file);
    lprintf(1,
            "Dataset::loadBase(): opened file containing %d %d-dimensional "
            "vectors.\n",
            base_loader.Num(), base_loader.Dim());
    num = std::min(num, base_loader.Num() - from);
    if (num <= 0) {
      fprintf(stderr,
              "Dataset::loadBase(): the requested range contains no vectors. "
              "aborting.\n");
      freeBase();
      return false;
    }
    N_base = num;
    if (D == 0)
      D = base_loader.Dim();
    else if (D != base_loader.Dim()) {
      fprintf(stderr,
              "Dataset::loadBase(): dimension mismatch (expected %d, got %d). "
              "aborting.\n",
              D, base_loader.Dim());
      freeBase();
      return false;
    }

    if (static_cast<size_t>(N_base) * static_cast<size_t>(D) >
        static_cast<size_t>(std::numeric_limits<BAddrT>::max())) {
      fprintf(stderr,
              "Dataset::loadBase(): address type is insufficient to address "
              "the requested dataset. aborting.\n");
      freeBase();
      return false;
    }

    cudaError_t result = cudaMallocManaged(
        &m_base, static_cast<BAddrT>(N_base) * D * sizeof(BaseT));
    if (result != cudaSuccess) {
      fprintf(stderr,
              "Dataset::loadBase(): failed to allocate memory. aborting.\n");
      freeBase();
      return false;
    }

    lprintf(2, "Dataset::loadBase(): loading %d base vectors starting at %d.\n",
            num, from);
    base_loader.load(m_base, from, num);
    lprintf(3, "Dataset::loadBase(): done.\n");
    return true;
  }

  /// load query vectors from file
  bool loadQuery(const std::string& query_file, KeyT from = 0,
                 KeyT num = std::numeric_limits<KeyT>::max()) {
    freeQuery();
    XVecsLoader<BaseT> query_loader(query_file);
    lprintf(1,
            "Dataset::loadQuery(): opened file containing %d %d-dimensional "
            "vectors.\n",
            query_loader.Num(), query_loader.Dim());
    num = std::min(num, query_loader.Num() - from);
    if (num <= 0) {
      fprintf(stderr,
              "Dataset::loadQuery(): the requested range contains no vectors. "
              "aborting.\n");
      freeQuery();
      return false;
    }
    if (N_query == 0)
      N_query = num;
    else if (N_query != num) {
      fprintf(stderr,
              "Dataset::loadQuery(): size mismatch (expected %d, got %d). "
              "aborting.\n",
              N_query, num);
      freeQuery();
      return false;
    }
    if (D == 0)
      D = query_loader.Dim();
    else if (D != query_loader.Dim()) {
      fprintf(stderr,
              "Dataset::loadQuery(): dimension mismatch (expected %d, got %d). "
              "aborting.\n",
              D, query_loader.Dim());
      freeQuery();
      return false;
    }

    if (static_cast<size_t>(N_query) * static_cast<size_t>(D) >
        static_cast<size_t>(std::numeric_limits<BAddrT>::max())) {
      fprintf(stderr,
              "Dataset::loadQuery(): address type is insufficient to address "
              "the requested dataset. aborting.\n");
      freeQuery();
      return false;
    }

    cudaError_t result = cudaMallocManaged(
        &m_query, static_cast<BAddrT>(N_query) * D * sizeof(BaseT));
    if (result != cudaSuccess) {
      fprintf(stderr,
              "Dataset::loadQuery(): failed to allocate memory. aborting.\n");
      freeQuery();
      return false;
    }

    lprintf(2,
            "Dataset::loadQuery(): loading %d query vectors starting at %d.\n",
            num, from);
    query_loader.load(m_query, from, num);
    lprintf(3, "Dataset::loadQuery(): done.\n");
    return true;
  }

  /// load ground truth indices from file
  bool loadGT(const std::string& gt_file, KeyT from = 0,
              KeyT num = std::numeric_limits<KeyT>::max()) {
    freeGT();
    XVecsLoader<KeyT> gt_loader(gt_file);
    lprintf(1,
            "Dataset::loadGT(): opened file containing %d %d-dimensional "
            "vectors.\n",
            gt_loader.Num(), gt_loader.Dim());
    num = std::min(num, gt_loader.Num() - from);
    if (num <= 0) {
      fprintf(stderr,
              "Dataset::loadGT(): the requested range contains no vectors. "
              "aborting.\n");
      freeGT();
      return false;
    }
    if (N_query == 0)
      N_query = num;
    else if (N_query != num) {
      fprintf(
          stderr,
          "Dataset::loadGT(): size mismatch (expected %d, got %d). aborting.\n",
          N_query, num);
      freeGT();
      return false;
    }
    K_gt = gt_loader.Dim();

    if (static_cast<size_t>(N_query) * static_cast<size_t>(K_gt) >
        static_cast<size_t>(std::numeric_limits<BAddrT>::max())) {
      fprintf(stderr,
              "Dataset::loadGT(): address type is insufficient to address the "
              "requested dataset. aborting.\n");
      freeGT();
      return false;
    }

    cudaError_t result = cudaMallocManaged(
        &m_gt, static_cast<BAddrT>(N_query) * K_gt * sizeof(KeyT));
    if (result != cudaSuccess) {
      fprintf(stderr,
              "Dataset::loadGT(): failed to allocate memory. aborting.\n");
      freeGT();
      return false;
    }

    lprintf(
        2,
        "Dataset::loadGT(): loading %d ground truth vectors starting at %d.\n",
        num, from);
    gt_loader.load(m_gt, from, num);
    lprintf(3, "Dataset::loadGT(): done.\n");
    return true;
  }

  // TODO(fabi): remove?
  void prefetch(int gpuId) const {
    lprintf(1, "Dataset::prefetch() to GPU %d.\n", gpuId);

    // push graph data to the gpu
    cudaMemAdvise(m_base, static_cast<BAddrT>(N_base) * D * sizeof(BaseT),
                  cudaMemAdviseSetAccessedBy, gpuId);
    cudaMemAdvise(m_query, static_cast<BAddrT>(N_query) * D * sizeof(BaseT),
                  cudaMemAdviseSetAccessedBy, gpuId);

    cudaMemAdvise(m_base, static_cast<BAddrT>(N_base) * D * sizeof(BaseT),
                  cudaMemAdviseSetReadMostly, gpuId);
    cudaMemAdvise(m_query, static_cast<BAddrT>(N_query) * D * sizeof(BaseT),
                  cudaMemAdviseSetReadMostly, gpuId);

    cudaMemPrefetchAsync(
        m_base, static_cast<BAddrT>(N_base) * D * sizeof(BaseT), gpuId);
    cudaMemPrefetchAsync(
        m_query, static_cast<BAddrT>(N_query) * D * sizeof(BaseT), gpuId);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
  }
};

#endif  // DATASET_CUH
