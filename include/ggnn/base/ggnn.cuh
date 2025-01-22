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

#ifndef INCLUDE_GGNN_GGNN_CUH
#define INCLUDE_GGNN_GGNN_CUH

#include <ggnn/base/def.h>        // IWYU pragma: export
#include <ggnn/base/fwd.h>        // IWYU pragma: export
#include <ggnn/base/dataset.cuh>  // IWYU pragma: export

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ggnn {

/**
 * GGNN multi-GPU wrapper
 *
 * @param KeyT datatype of dataset indices (needs to be able to represent
 * N_base, signed integer required)
 * @param ValueT distance value type
 */
template <typename KeyT, typename ValueT>
class GGNN {
 public:
  using Results = ggnn::Results<KeyT, ValueT>;
  using Graph = ggnn::Graph<KeyT, ValueT>;

  /// Maximum data dimension supported by GGNN
  static constexpr uint32_t MIN_D = 1;
  static constexpr uint32_t MAX_D = 4096;
  /// Maximum number of neighbors supported by GGNN
  static constexpr uint32_t MIN_KBUILD = 2;
  static constexpr uint32_t MAX_KBUILD = 512;

  GGNN();
  GGNN(const GGNN& other) = delete;
  GGNN(GGNN&& other) noexcept = default;
  GGNN& operator=(const GGNN& other) = delete;
  GGNN& operator=(GGNN&& other) noexcept = default;
  virtual ~GGNN() = default;

  /**
   * Set the cache directory for GGNN to work in.
   * The graph will be loaded from and stored to files in this directory if requested or when
   * required due to insufficient memory.
   */
  virtual void setWorkingDirectory(const std::filesystem::path& dir);
  /**
   * Set the maximum amount of CPU memory that GGNN is allowed to use for caching graph shards.
   *
   * When insufficient, shards will be swapped out to the working directory.
   */
  virtual void setCPUMemoryLimit(const size_t memory_limit);

  /**
   * Set the GPUs to use (CUDA device indices).
   */
  virtual void setGPUs(const std::span<const int>& gpu_ids);
  /**
   * Set the GPUs to use (CUDA device indices).
   *
   * (This overload allows to use an initializer list.)
   */
  void setGPUs(const std::vector<int>& gpu_ids)
  {
    setGPUs(std::span<const int>{gpu_ids.cbegin(), gpu_ids.cend()});
  }

  /**
   * Set the size of shards to work on.
   * (Optional, default 0: entire base in one shard).
   * The base datasets needs to be evenly divisible by the shard size.
   * The resulting number of shards needs to be evenly divisible by the number of GPUs.
   */
  virtual void setShardSize(const uint32_t N_shard);

  /**
   * Enable or disable returning results on GPU.
   * When enabled, results are directly returned on the GPU, no copy to CPU is performed.
   * NOTE: Only a single GPU is supported in this mode.
   * NOTE: Querying for K results from N shards will return N*K sorted results per query.
   */
  virtual void setReturnResultsOnGPU(const bool return_results_on_gpu = true);

  /**
   * Set the base dataset.
   */
  virtual void setBase(GenericDataset&& base);

  /**
   * Set a reference to the base dataset.
   * NOTE: The calling code needs to ensure that the base data remains accessible during graph
   * construction and query.
   */
  void setBaseReference(const GenericDataset& base);
  // setting a reference to a temporary would result in undefined behavior
  void setBaseReference(GenericDataset&&) = delete;

  /**
   * Build the GGNN search graph.
   *
   * Requires base to be set.
   */
  virtual void build(const uint32_t KBuild, const float tau_build,
                     const uint32_t refinement_iterations = 2,
                     const DistanceMeasure measure = DistanceMeasure::Euclidean);
  /**
   * Store the GGNN search graph.
   */
  virtual void store();
  /**
   * Load a previously built GGNN search graph.
   *
   * Requires base to be set.
   */
  virtual void load(const uint32_t KBuild);

  /**
   * Query the GGNN search graph.
   * @param query may be given on CPU or GPU
   *
   * Requires base to be set.
   * Requires a graph to be built or loaded first.
   *
   * NOTE: query data type has to match base data type
   */
  [[nodiscard]] virtual Results query(const GenericDataset& query, const uint32_t KQuery,
                                      const float tau_query, const uint32_t max_iterations = 400,
                                      const DistanceMeasure measure = DistanceMeasure::Euclidean);

  /**
   * Run a brute-force query on the base dataset.
   * @param query may be given on CPU or GPU
   *
   * Requires base to be set.
   * NOTE: This function currently supports only a single GPU
   *
   * NOTE: query data type has to match base data type
   */
  [[nodiscard]] virtual Results bfQuery(const GenericDataset& query, const uint32_t KGT = 100,
                                        const DistanceMeasure measure = DistanceMeasure::Euclidean);

  /**
   * Access the GGNN graph.
   * @param global_shard_id The graph shard to be accessed.
   *
   * NOTE: The reference is invalidated when a query is run and shards need to be swapped out.
   */
  [[nodiscard]] virtual const Graph& getGraph(const uint32_t global_shard_id = 0);

 protected:
  GGNN(int) {}

 private:
  std::unique_ptr<GGNN> pimpl;
};  // GGNN

};  // namespace ggnn

#endif  // INCLUDE_GGNN_GGNN_CUH
