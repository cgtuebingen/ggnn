/* Copyright 2025 ComputerGraphics Tuebingen. All Rights Reserved.

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
// converted to GGNN library by: Lukas Ruppert, Deborah Kornwolf

#include <ggnn/base/eval.h>

#include <ggnn/base/def.h>
#include <ggnn/base/lib.h>
#include <ggnn/base/data.cuh>
#include <ggnn/base/dataset.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <span>
#include <stdexcept>

#include <glog/logging.h>

namespace ggnn {

template <typename BaseT, typename ValueT>
ValueT compute_distance(const std::span<const BaseT>& a, const std::span<const BaseT>& b,
                        const DistanceMeasure measure)
{
  CHECK_EQ(a.size(), b.size());
  const size_t D = a.size();
  ValueT distance = 0.0f, a_norm = 0.0f, b_norm = 0.0f;
  for (size_t d = 0; d < D; ++d) {
    if (measure == DistanceMeasure::Euclidean) {
      distance += (static_cast<ValueT>(a[d]) - static_cast<ValueT>(b[d])) *
                  (static_cast<ValueT>(a[d]) - static_cast<ValueT>(b[d]));
    }
    else if (measure == DistanceMeasure::Cosine) {
      distance += static_cast<ValueT>(a[d]) * static_cast<ValueT>(b[d]);
      a_norm += static_cast<ValueT>(a[d]) * static_cast<ValueT>(a[d]);
      b_norm += static_cast<ValueT>(a[d]) * static_cast<ValueT>(a[d]);
    }
  }
  if (measure == DistanceMeasure::Euclidean) {
    distance = std::sqrt(distance);
  }
  else if (measure == DistanceMeasure::Cosine) {
    if (a_norm * b_norm > 0.0f)
      distance = std::fabs(1.0f - distance / std::sqrt(a_norm * b_norm));
    else
      distance = 1.0f;
  }
  return distance;
}

std::ostream& operator<<(std::ostream& os, const Evaluation& eval)
{
  os << "c@1 (=r@1): " << eval.c1;
  if (!std::isnan(eval.c1_dup))
    os << " +duplicates: " << eval.c1_dup << '\n';
  else
    os << " (duplicates unknown)\n";
  os << "c@" << eval.KQuery << ": " << eval.cKQuery;
  if (!std::isnan(eval.cKQuery_dup))
    os << " +duplicates: " << eval.cKQuery_dup << '\n';
  else
    os << " (duplicates unknown)\n";
  os << "r@" << eval.KQuery << ": " << eval.rKQuery;
  if (!std::isnan(eval.rKQuery_dup))
    os << " +duplicates: " << eval.rKQuery_dup;
  else
    os << " (duplicates unknown)";

  return os;
}

template <typename KeyT, typename ValueT>
Evaluator<KeyT, ValueT>::Evaluator(const GenericDataset& base, const GenericDataset& query,
                                   const Dataset<KeyT>& gt, const uint32_t KQuery,
                                   const DistanceMeasure measure)
    : KQuery{KQuery}, measure{measure}, gt{gt.clone()}
{
  if (!base.N || !query.N) {
    LOG(WARNING)
        << "Cannot check for duplicates in ground truth indices: No base and/or query data given.";
    return;
  }
  if (!base.isCPUAccessible() || !query.isCPUAccessible()) {
    LOG(WARNING)
        << "Cannot check for duplicates in ground truth indices: Data is not CPU-accessible.";
    return;
  }

  if (!gt.isCPUAccessible())
    throw std::runtime_error("Ground truth data needs to be given on the CPU for evaluation.");

  if (!gt_duplicates.top1DuplicateEnd.empty() || !gt_duplicates.topKDuplicateEnd.empty())
    return;

  CHECK_EQ(base.type, query.type);

  auto compute_distance_base_to_query = [&](size_t base_idx, size_t query_idx) -> ValueT {
    switch (base.type) {
      case DataType::FLOAT: {
        Dataset<float> b = base.reference();
        Dataset<float> q = query.reference();
        return compute_distance<float, ValueT>(
            {&b.at(static_cast<size_t>(base_idx) * base.D), base.D},
            {&q.at(static_cast<size_t>(query_idx) * query.D), query.D}, measure);
      }
      case DataType::UINT8: {
        Dataset<uint8_t> b = base.reference();
        Dataset<uint8_t> q = query.reference();
        return compute_distance<uint8_t, ValueT>(
            {&b.at(static_cast<size_t>(base_idx) * base.D), base.D},
            {&q.at(static_cast<size_t>(query_idx) * query.D), query.D}, measure);
      }
      default:
        break;
    }
    throw std::runtime_error("unsupported data type");
  };

  VLOG(2) << "searching for duplicates in the ground truth indices.";
  const float Epsilon = 0.000001f;
  size_t total_num_duplicates_top_1 = 0, total_num_duplicates_top_k = 0;
  uint32_t max_dup_top_1 = 0, max_dup_top_k = 0;
  for (uint32_t n = 0; n < query.N; n++) {
    const ValueT gt_dist1 = compute_distance_base_to_query(gt[n * gt.D], n);
    uint32_t num_duplicates_top_1 = 0, num_duplicates_top_k = 0;
    for (uint32_t k = 1; k < gt.D; ++k) {
      const ValueT gt_dist_k = compute_distance_base_to_query(gt[n * gt.D + k], n);
      if (gt_dist_k - gt_dist1 > Epsilon)
        break;
      ++num_duplicates_top_1;
    }
    total_num_duplicates_top_1 += num_duplicates_top_1;
    if (num_duplicates_top_1 > max_dup_top_1)
      max_dup_top_1 = num_duplicates_top_1;
    gt_duplicates.top1DuplicateEnd.push_back(1 + num_duplicates_top_1);
    if (KQuery <= gt.D) {
      const ValueT gt_distKQuery = compute_distance_base_to_query(gt[n * gt.D + KQuery - 1], n);
      for (uint32_t k = KQuery; k < gt.D; ++k) {
        const ValueT gt_dist_k = compute_distance_base_to_query(gt[n * gt.D + k], n);
        if (gt_dist_k - gt_distKQuery > Epsilon)
          break;
        ++num_duplicates_top_k;
      }
      total_num_duplicates_top_k += num_duplicates_top_k;
      if (num_duplicates_top_k > max_dup_top_k)
        max_dup_top_k = num_duplicates_top_k;
      gt_duplicates.topKDuplicateEnd.push_back(KQuery + num_duplicates_top_k);
    }
    else
      gt_duplicates.topKDuplicateEnd.push_back(gt.D);
  }
  VLOG(2) << "found " << total_num_duplicates_top_1 << " duplicates for c@1."
          << " max: " << max_dup_top_1;
  if (KQuery <= gt.D) {
    VLOG(2) << "found " << total_num_duplicates_top_k << " duplicates for c@" << KQuery << "."
            << " max: " << max_dup_top_k;
  }
}

template <typename KeyT, typename ValueT>
Evaluation Evaluator<KeyT, ValueT>::evaluateResults(const Dataset<KeyT>& results)
{
  CHECK_GE(gt.N, results.N);

  if (!gt.D)
    throw std::runtime_error("No ground truth data loaded. cannot compute accuracy.");
  if (!results.isCPUAccessible())
    throw std::runtime_error("Results need to be given on the CPU for evaluation.");

  const bool has_duplicate_info =
      (!gt_duplicates.top1DuplicateEnd.empty() && !gt_duplicates.topKDuplicateEnd.empty());

  uint32_t c1 = 0;
  uint32_t c1_dup = 0;
  uint32_t cKQuery = 0;
  uint32_t cKQuery_dup = 0;
  uint32_t rKQuery = 0;
  uint32_t rKQuery_dup = 0;

  for (uint32_t n = 0; n < results.N; n++) {
    const uint32_t endTop1 = has_duplicate_info ? gt_duplicates.top1DuplicateEnd.at(n) : 1;
    const uint32_t endTopK = has_duplicate_info ? gt_duplicates.topKDuplicateEnd.at(n) : KQuery;

    CHECK_LE(endTopK, gt.D);

    for (uint32_t k_result = 0; k_result < KQuery; k_result++) {
      const KeyT q = results[n * KQuery + k_result];
      for (uint32_t k_gt = 0; k_gt < endTopK; k_gt++) {
        const KeyT gt_key = gt[n * gt.D + k_gt];
        if (q == gt_key) {
          if (!k_gt) {
            if (!k_result)
              ++c1;
            if (k_gt < KQuery)
              ++rKQuery;
            ++rKQuery_dup;
          }
          if (k_gt < endTop1) {
            if (!k_result)
              ++c1_dup;
          }
          if (k_gt < KQuery)
            ++cKQuery;
          ++cKQuery_dup;
          continue;
        }
      }
    }
  }

  const float inv_num_queries = 1.0f / static_cast<float>(results.N);
  const float inv_num_results = 1.0f / static_cast<float>(results.N * KQuery);

  return Evaluation{
      .KQuery = KQuery,
      .c1 = static_cast<float>(c1) * inv_num_queries,
      .c1_dup = has_duplicate_info ? static_cast<float>(c1_dup) * inv_num_queries
                                   : std::numeric_limits<float>::quiet_NaN(),
      .cKQuery = static_cast<float>(cKQuery) * inv_num_results,
      .cKQuery_dup = has_duplicate_info ? static_cast<float>(cKQuery_dup) * inv_num_results
                                        : std::numeric_limits<float>::quiet_NaN(),
      .rKQuery = static_cast<float>(rKQuery) * inv_num_queries,
      .rKQuery_dup = has_duplicate_info ? static_cast<float>(rKQuery_dup) * inv_num_queries
                                        : std::numeric_limits<float>::quiet_NaN(),
  };
}

GGNN_EVAL(GGNN_KEYS, GGNN_VALUES, GGNN_INSTANTIATE_STRUCT, Evaluator);

};  // namespace ggnn
