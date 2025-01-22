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

#ifndef INCLUDE_GGNN_EVAL_H
#define INCLUDE_GGNN_EVAL_H

#include <ggnn/base/def.h>
#include <ggnn/base/fwd.h>

#include <cstdint>
#include <ostream>

#include <vector>

namespace ggnn {

struct GTDuplicates {
  // indices within the ground truth list per point up to which result ids
  // need to be compared.
  // without duplicates in the dataset, each entry should just be 1 / KQuery
  std::vector<uint32_t> top1DuplicateEnd{};
  std::vector<uint32_t> topKDuplicateEnd{};
};

struct Evaluation {
  uint32_t KQuery{};

  float c1{0};
  float c1_dup{0};
  float cKQuery{0};
  float cKQuery_dup{0};
  float rKQuery{0};
  float rKQuery_dup{0};
};

std::ostream& operator<<(std::ostream& os, const Evaluation& eval);

template <typename KeyT, typename ValueT>
struct Evaluator {
  uint32_t KQuery{0};
  DistanceMeasure measure{};
  Dataset<KeyT> gt;
  GTDuplicates gt_duplicates{};

  Evaluator() = default;

  Evaluator(const GenericDataset& base, const GenericDataset& query, const Dataset<KeyT>& gt,
            const uint32_t KQuery, const DistanceMeasure measure);

  [[nodiscard]] Evaluation evaluateResults(const Dataset<KeyT>& results);
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_EVAL_H
