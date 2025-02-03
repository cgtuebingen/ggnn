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
#include <ggnn/base/ggnn.cuh>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// only needed for getTotalSystemMemory()
#include <unistd.h>

using namespace ggnn;

DEFINE_string(base, "", "Path to file with base vectors (fvecs/bvecs).");
DEFINE_uint32(subset, 0, "Number of base vectors to use.");
DEFINE_string(query, "", "Path to file with query vectors (fvecs/bvecs).");
DEFINE_string(gt, "", "Path to file with groundtruth vectors (ivecs).");
DEFINE_string(graph_dir, "", "Directory to store and load ggnn graph files.");
DEFINE_uint32(k_build, 24, "Number of neighbors for graph construction");
DEFINE_double(tau_build, 0.5, "Search graph construction slack factor.");
DEFINE_uint32(refinement_iterations, 2, "Number of refinement iterations.");
DEFINE_uint32(k_query, 10, "Number of neighbors to query for.");
DEFINE_uint32(max_iterations, 200, "Maximum number of search iterations per query.");
DEFINE_string(measure, "euclidean", "Distance measure. (euclidean or cosine)");
DEFINE_uint32(shard_size, 0, "Number of vectors per shard.");
DEFINE_string(gpu_ids, "0", "GPU ids, separated by spaces.");
DEFINE_bool(grid_search, false, "Perform queries for a wide range of parameters.");

size_t getTotalSystemMemory()
{
  size_t pages = sysconf(_SC_PHYS_PAGES);
  size_t page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
  google::InstallFailureSignalHandler();

  gflags::SetUsageMessage(
      "GGNN: Graph-based GPU Nearest Neighbor Search\n"
      "by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. "
      "Lensch\n"
      "(c) 2025 Computer Graphics University of Tuebingen");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Reading files";
  CHECK(std::filesystem::exists(FLAGS_base))
      << "File for base vectors has to exist: " << FLAGS_base;
  CHECK(std::filesystem::exists(FLAGS_query))
      << "File for query vectors has to exist: " << FLAGS_query;
  CHECK(std::filesystem::exists(FLAGS_gt))
      << "File for groundtruth vectors has to exist: " << FLAGS_gt;

  CHECK_GE(FLAGS_tau_build, 0) << "tau_build has to be bigger or equal 0.";
  CHECK_GE(FLAGS_refinement_iterations, 0)
      << "The number of refinement iterations has to be non-negative.";

  /// data type for addressing points (needs to be able to represent N)
  using KeyT = int32_t;
  /// data type of computed distances
  using ValueT = float;

  using GGNN = ggnn::GGNN<KeyT, ValueT>;
  using Results = ggnn::Results<KeyT, ValueT>;
  using Evaluator = ggnn::Evaluator<KeyT, ValueT>;

  /// distance measure (Euclidean or Cosine)
  const DistanceMeasure measure = []() {
    if (FLAGS_measure == "euclidean") {
      return DistanceMeasure::Euclidean;
    }
    else if (FLAGS_measure == "cosine") {
      return DistanceMeasure::Cosine;
    }
    LOG(FATAL) << "invalid measure: " << FLAGS_measure;
  }();

  // vector of GPU ids
  std::istringstream iss(FLAGS_gpu_ids);
  std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                   std::istream_iterator<std::string>());

  std::vector<int> gpus;
  for (auto&& r : results) {
    int gpu_id = std::atoi(r.c_str());
    gpus.push_back(gpu_id);
  }

  // base & query datasets
  GenericDataset base = GenericDataset::load(
      FLAGS_base, 0, FLAGS_subset ? FLAGS_subset : std::numeric_limits<uint32_t>::max(), true);
  GenericDataset query =
      GenericDataset::load(FLAGS_query, 0, std::numeric_limits<uint32_t>::max(), true);

  // initialize GGNN
  GGNN ggnn;

  const size_t total_memory = getTotalSystemMemory();
  // guess the available memory (assume 1/8 used elsewhere, subtract dataset)
  const size_t available_memory = total_memory - total_memory / 8 - base.required_size_bytes();
  ggnn.setCPUMemoryLimit(available_memory);

  ggnn.setWorkingDirectory(FLAGS_graph_dir);
  // reference the dataset to avoid a copy
  ggnn.setBaseReference(base);

  // only necessary in multi-GPU mode
  ggnn.setGPUs(gpus);
  ggnn.setShardSize(FLAGS_shard_size);

  // build the graph
  if (!FLAGS_graph_dir.empty() &&
      std::filesystem::is_regular_file(std::filesystem::path{FLAGS_graph_dir} / "part_0.ggnn")) {
    ggnn.load(FLAGS_k_build);
  }
  else {
    ggnn.build(FLAGS_k_build, static_cast<float>(FLAGS_tau_build), FLAGS_refinement_iterations,
               measure);

    if (!FLAGS_graph_dir.empty()) {
      ggnn.store();
    }
  }

  // load or compute ground truth
  const bool loadGT = std::filesystem::is_regular_file(FLAGS_gt);
  Dataset<KeyT> gt = loadGT ? Dataset<KeyT>::load(FLAGS_gt) : Dataset<KeyT>{};

  if (!gt.data()) {
    gt = ggnn.bfQuery(query).ids;
    if (!FLAGS_gt.empty()) {
      LOG(INFO) << "exporting brute-forced ground truth data.";
      gt.store(FLAGS_gt);
    }
  }

  Evaluator eval{base, query, gt, FLAGS_k_query, measure};

  // query
  auto query_function = [&ggnn, &eval, &query, measure, max_iter=FLAGS_max_iterations](const float tau_query) {
    Results results;
    LOG(INFO) << "--";
    LOG(INFO) << "Query with tau_query " << tau_query << " max iterations " << max_iter;
    results = ggnn.query(query, FLAGS_k_query, tau_query, max_iter, measure);
    LOG(INFO) << eval.evaluateResults(results.ids);
  };

  if (FLAGS_grid_search) {
    LOG(INFO) << "--";
    LOG(INFO) << "grid-search:";
    for (int i = 0; i < 70; ++i)
      query_function(static_cast<float>(i) * 0.01f);
    for (int i = 7; i <= 20; ++i)
      query_function(static_cast<float>(i) * 0.1f);
  }
  else {  // by default, just execute a few queries
    LOG(INFO) << "--";
    LOG(INFO)
        << "Querying for 90, 95, 99% R@1 (if running on SIFT1M with default parameters):";
    query_function(0.34f);
    query_function(0.41f);
    query_function(0.51f);
  }

  VLOG(1) << "done!";
  gflags::ShutDownCommandLineFlags();
  return 0;
}
