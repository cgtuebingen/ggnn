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

#ifndef INCLUDE_GGNN_UTILS_CUDA_KNN_DATASET_CUH_
#define INCLUDE_GGNN_UTILS_CUDA_KNN_DATASET_CUH_

#include <algorithm>
#include <limits>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

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
  BaseT* h_base{nullptr};

  /// query vectors
  BaseT* h_query{nullptr};
  /// ground truth indices in the dataset for the given queries
  KeyT* gt{nullptr};

  /// number of dataset vectors
  int N_base{0};
  /// number of query vectors (and ground truth indices)
  int N_query{0};
  /// dimension of vectors in the dataset and query
  int D{0};
  /// number of nearest neighbors per ground truth entry
  int K_gt{0};

  // indices within the ground truth list per point up to which result ids
  // need to be compared.
  // without duplicates in the dataset, each entry should just be 1 / KQuery
  std::vector<uint8_t> top1DuplicateEnd;
  std::vector<uint8_t> topKDuplicateEnd;

  Dataset(const std::string& basePath, const std::string& queryPath,
          const std::string& gtPath, const size_t N_base = std::numeric_limits<size_t>::max()) {

    VLOG(1) << "N_base: " << N_base;

    bool success = loadBase(basePath, 0, N_base) && loadQuery(queryPath) && loadGT(gtPath);

    if (!success)
      throw std::runtime_error(
          "failed to load dataset (see previous log entries for details).\n");
  }

  //TODO(fabi): cleanup.
  ~Dataset() {
    freeBase();
    freeQuery();
    freeGT();
  }

  Dataset(const Dataset&) = delete;
  Dataset(Dataset&&) = delete;
  Dataset& operator=(const Dataset&) = delete;
  Dataset& operator=(Dataset&&) = delete;

  void freeBase() {
    cudaFreeHost(h_base);
    h_base = nullptr;
    N_base = 0;
    if (!h_query) D = 0;
  }

  void freeQuery() {
    cudaFreeHost(h_query);
    h_query = nullptr;
    if (!gt) N_query = 0;
    if (!h_base) D = 0;
  }

  void freeGT() {
    free(gt);
    gt = nullptr;
    if (!h_query) N_query = 0;
    K_gt = 0;
  }

  /// load base vectors from file
  bool loadBase(const std::string& base_file, size_t from = 0,
                size_t num = std::numeric_limits<size_t>::max()) {
    freeBase();
    XVecsLoader<BaseT> base_loader(base_file);

    num = std::min(num, base_loader.Num() - from);
    CHECK_GT(num, 0) << "The requested range contains no vectors.";

    N_base = num;
    if (D == 0) {
      D = base_loader.Dim();
    }
    CHECK_EQ(D, base_loader.Dim()) << "Dimension mismatch";

    const size_t dataset_max_index =
        static_cast<size_t>(N_base) * static_cast<size_t>(D);
    CHECK_LT(dataset_max_index, std::numeric_limits<BAddrT>::max())
        << "Address type is insufficient to address "
           "the requested dataset. aborting";

    const size_t base_memsize = static_cast<BAddrT>(N_base) * D * sizeof(BaseT);

    CHECK_CUDA(cudaMallocHost(&h_base, base_memsize, cudaHostAllocPortable | cudaHostAllocWriteCombined));

    base_loader.load(h_base, from, num);

    return true;
  }

  /// load query vectors from file
  bool loadQuery(const std::string& query_file, KeyT from = 0,
                 KeyT num = std::numeric_limits<KeyT>::max()) {
    freeQuery();
    XVecsLoader<BaseT> query_loader(query_file);

    num = std::min(num, query_loader.Num() - from);
    CHECK_GT(num, 0) << "The requested range contains no vectors.";

    if (N_query == 0) {
      N_query = num;
    }
    CHECK_EQ(N_query, num) << "Number mismatch";

    if (D == 0) {
      D = query_loader.Dim();
    }
    CHECK_EQ(D, query_loader.Dim()) << "Dimension mismatch";

    const size_t dataset_max_index =
        static_cast<size_t>(N_query) * static_cast<size_t>(D);
    CHECK_LT(dataset_max_index, std::numeric_limits<BAddrT>::max())
        << "Address type is insufficient to address "
           "the requested dataset. aborting";


    const size_t query_memsize = static_cast<BAddrT>(N_query) * D * sizeof(BaseT);

    CHECK_CUDA(cudaMallocHost(&h_query, query_memsize, cudaHostAllocPortable));

    query_loader.load(h_query, from, num);

    return true;
  }

  /// load ground truth indices from file
  bool loadGT(const std::string& gt_file, KeyT from = 0,
              KeyT num = std::numeric_limits<KeyT>::max()) {
    freeGT();


    if (gt_file.empty()) {
      LOG(INFO) << "No ground truth file loaded. Make sure to compute it yourself before evaluating any queries.";

      CHECK_GT(N_query, 0) << "Cannot determine the number of GT entries which need to be computed if the query is not yet loaded.";
      K_gt = 100;

      //TODO(fabi): move out of if branch.
      gt = (KeyT*) malloc(static_cast<BAddrT>(N_query) * K_gt * sizeof(KeyT));
      CHECK(gt);

      return true;
    }

    XVecsLoader<KeyT> gt_loader(gt_file);

    num = std::min(num, gt_loader.Num() - from);
    CHECK_GT(num, 0) << "The requested range contains no vectors.";

    if (N_query == 0) {
      N_query = num;
    }
    CHECK_EQ(N_query, num) << "Number mismatch";

    K_gt = gt_loader.Dim();

    const size_t dataset_max_index =
        static_cast<size_t>(N_query) * static_cast<size_t>(K_gt);
    CHECK_LT(dataset_max_index, std::numeric_limits<BAddrT>::max())
        << "Address type is insufficient to address "
           "the requested dataset. aborting";

    gt = (KeyT*) malloc(static_cast<BAddrT>(N_query) * K_gt * sizeof(KeyT));
    CHECK(gt);

    gt_loader.load(gt, from, num);
    return true;
  }

  template <DistanceMeasure measure, typename ValueT>
  ValueT compute_distance_query(KeyT index, KeyT query) const {
    CHECK_GE(index, 0);
    CHECK_GE(query, 0);
    CHECK_LT(index, N_base);
    CHECK_LT(query, N_query);

    ValueT distance = 0.0f, index_norm = 0.0f, query_norm = 0.0f;
    for (int d=0; d<D; ++d)
    {
      if (measure == Euclidean) {
        distance += (h_query[static_cast<size_t>(query)*D+d]
                    -h_base [static_cast<size_t>(index)*D+d])
                   *(h_query[static_cast<size_t>(query)*D+d]
                    -h_base [static_cast<size_t>(index)*D+d]);
      }
      else if (measure == Cosine) {
        distance   += h_query[static_cast<size_t>(query)*D+d]
                     *h_base [static_cast<size_t>(index)*D+d];
        query_norm += h_query[static_cast<size_t>(query)*D+d]
                     *h_query[static_cast<size_t>(query)*D+d];
        index_norm += h_base [static_cast<size_t>(index)*D+d]
                     *h_base [static_cast<size_t>(index)*D+d];
      }
    }
    if (measure == Euclidean) {
      distance = sqrtf(distance);
    }
    else if (measure == Cosine) {
      if (index_norm*query_norm > 0.0f)
        distance = fabs(1.0f-distance/sqrtf(index_norm*query_norm));
      else
        distance = 1.0f;
    }
    return distance;
  };

  template <DistanceMeasure measure, typename ValueT>
  ValueT compute_distance_base_to_base(KeyT a, KeyT b) const {
    CHECK_GE(a, 0);
    CHECK_GE(b, 0);
    CHECK_LT(a, N_base);
    CHECK_LT(b, N_base);

    ValueT distance = 0.0f, a_norm = 0.0f, b_norm = 0.0f;
    for (int d=0; d<D; ++d)
    {
      if (measure == Euclidean) {
        distance += (h_base[static_cast<size_t>(b)*D+d]-h_base[static_cast<size_t>(a)*D+d])
                   *(h_base[static_cast<size_t>(b)*D+d]-h_base[static_cast<size_t>(a)*D+d]);
      }
      else if (measure == Cosine) {
        distance += h_base[static_cast<size_t>(b)*D+d]*h_base[static_cast<size_t>(a)*D+d];
        b_norm += h_base[static_cast<size_t>(b)*D+d]*h_base[static_cast<size_t>(b)*D+d];
        a_norm += h_base[static_cast<size_t>(a)*D+d]*h_base[static_cast<size_t>(a)*D+d];
      }
    }
    if (measure == Euclidean) {
      distance = sqrtf(distance);
    }
    else if (measure == Cosine) {
      if (a_norm*b_norm > 0.0f)
        distance = fabs(1.0f-distance/sqrtf(a_norm*b_norm));
      else
        distance = 1.0f;
    }
    return distance;
  };

  template <DistanceMeasure measure, typename ValueT>
  void checkForDuplicatesInGroundTruth(const int KQuery) {
    if (!top1DuplicateEnd.empty() || !topKDuplicateEnd.empty())
      return;
    VLOG(2) << "searching for duplicates in the ground truth indices.";

    const float Epsilon = 0.000001f;

    size_t total_num_duplicates_top_1 = 0, total_num_duplicates_top_k = 0;
    uint8_t max_dup_top_1 = 0, max_dup_top_k = 0;

    for (int n = 0; n < N_query; n++) {
      const ValueT gt_dist1 = compute_distance_query<measure, ValueT>(gt[n * K_gt], n);
      uint8_t num_duplicates_top_1 = 0, num_duplicates_top_k = 0;
      for (int k=1; k < K_gt; ++k) {
        const ValueT gt_dist_k = compute_distance_query<measure, ValueT>(gt[n * K_gt + k], n);
        if (gt_dist_k-gt_dist1 > Epsilon)
          break;
        ++num_duplicates_top_1;
      }
      total_num_duplicates_top_1 += num_duplicates_top_1;
      if (num_duplicates_top_1 > max_dup_top_1)
        max_dup_top_1 = num_duplicates_top_1;
      top1DuplicateEnd.push_back(1+num_duplicates_top_1);

      if (KQuery <= K_gt) {
        const ValueT gt_distKQuery = compute_distance_query<measure, ValueT>(gt[n * K_gt + KQuery-1], n);
        for (int k=KQuery; k < K_gt; ++k) {
          const ValueT gt_dist_k = compute_distance_query<measure, ValueT>(gt[n * K_gt + k], n);
          if (gt_dist_k-gt_distKQuery > Epsilon)
            break;
          ++num_duplicates_top_k;
        }

        total_num_duplicates_top_k += num_duplicates_top_k;
        if (num_duplicates_top_k > max_dup_top_k)
          max_dup_top_k = num_duplicates_top_k;
        topKDuplicateEnd.push_back(KQuery+num_duplicates_top_k);
      }
      else
        topKDuplicateEnd.push_back(K_gt);
    }

    VLOG(2) << "found " << total_num_duplicates_top_1 << " duplicates for c@1."
            << " max: " << uint32_t(max_dup_top_1);
    if (KQuery <= K_gt) {
      VLOG(2) << "found " << total_num_duplicates_top_k << " duplicates for c@"
              << KQuery << "."
              << " max: " << uint32_t(max_dup_top_k);
    }
  }
};

#endif  // INCLUDE_GGNN_UTILS_CUDA_KNN_DATASET_CUH_
