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

#ifndef INCLUDE_GGNN_QUERY_KERNELS_CUH
#define INCLUDE_GGNN_QUERY_KERNELS_CUH

#include <ggnn/base/def.h>
#include <ggnn/base/fwd.h>

#include <cstdint>
#include <memory>

namespace ggnn {

template <typename KeyT, typename ValueT, typename BaseT>
struct GPUInstance;

template <typename KeyT, typename ValueT, typename BaseT>
class QueryKernels {
 public:
  using GPUInstance = ggnn::GPUInstance<KeyT, ValueT, BaseT>;
  using Graph = ggnn::Graph<KeyT, ValueT>;
  using Results = ggnn::Results<KeyT, ValueT>;

  QueryKernels() = default;
  QueryKernels(const DistanceMeasure measure);
  virtual ~QueryKernels() = default;
  QueryKernels(const QueryKernels&) = delete;
  QueryKernels(QueryKernels&&) noexcept = default;
  QueryKernels& operator=(const QueryKernels&) = delete;
  QueryKernels& operator=(QueryKernels&&) noexcept = default;

  virtual void query(const GPUInstance& gpu_instance, const uint32_t shard_id,
                     const Dataset<BaseT>& query, const uint32_t KQuery, const uint32_t max_iters,
                     const float tau_query, Results& results)
  {
    pimpl->query(gpu_instance, shard_id, query, KQuery, max_iters, tau_query, results);
  }
  virtual void bruteForceQuery(const Dataset<BaseT>& base, const Dataset<BaseT>& query,
                               const uint32_t KQuery, Results& results, cudaStream_t stream = 0)
  {
    pimpl->bruteForceQuery(base, query, KQuery, results, stream);
  }

 private:
  std::unique_ptr<QueryKernels> pimpl;
};

};  // namespace ggnn

#endif  // INCLUDE_GGNN_QUERY_KERNELS_CUH
