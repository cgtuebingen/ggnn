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

#include <ggnn/base/def.h>
#include <ggnn/base/eval.h>
#include <ggnn/base/graph.h>
#include <ggnn/base/lib.h>
#include <ggnn/base/dataset.cuh>
#include <ggnn/base/ggnn.cuh>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace ggnn;

namespace nb = nanobind;

using namespace nb::literals;

#if NB_VERSION_MAJOR > 1
// shapes are signed starting with NB 2, nb::any now stands for any type, not any shape
constexpr auto any_size = -1;
#else
constexpr auto any_size = nb::any;
#endif

using KeyT = int32_t;
using ValueT = float;

template <typename T>
using NB2DArrayTorch = nb::ndarray<T, nb::shape<any_size, any_size>, nb::c_contig, nb::pytorch>;
template <typename T>
using NB2DArrayCPU = nb::ndarray<T, nb::shape<any_size, any_size>, nb::c_contig, nb::device::cpu>;
template <typename T>
using NB2DArrayGPU = nb::ndarray<T, nb::shape<any_size, any_size>, nb::c_contig, nb::device::cuda>;

struct GlobalInit {
  GlobalInit()
  {
    google::InitGoogleLogging("GGNN");
    google::LogToStderr();
    // google::SetVLOGLevel("*", 4);
  }
};

static GlobalInit init{};

auto dataset_to_ndarray_view =
    []<typename T>(const Dataset<T>& dataset) -> NB2DArrayTorch<const T> {
  return NB2DArrayTorch<const T>(
      dataset.data(), {dataset.N, dataset.D}, nb::handle(), {}, nb::dtype<T>(),
      dataset.isGPUAccessible() ? nb::device::cuda::value : nb::device::cpu::value, dataset.gpu_id);
};

auto dataset_to_ndarray = []<typename T>(Dataset<T>&& dataset) -> NB2DArrayTorch<T> {
  Dataset<T>* reowned_data = new Dataset<T>{std::move(dataset)};

  const int32_t device_type =
      reowned_data->isGPUAccessible() ? nb::device::cuda::value : nb::device::cpu::value;

  return NB2DArrayTorch<T>(reowned_data->data(), {reowned_data->N, reowned_data->D},
                           nb::capsule(reowned_data,
                                       [](void* data) noexcept -> void {
                                         Dataset<T>* reowned_data =
                                             reinterpret_cast<Dataset<T>*>(data);
                                         delete reowned_data;
                                       }),
                           {}, nb::dtype<T>(), device_type, dataset.gpu_id);
};

auto ndarray_to_dataset = []<typename T>(const NB2DArrayTorch<T>& data) -> Dataset<T> {
  if (data.device_type() == nb::device::cpu::value)
    return Dataset<T>::referenceCPUData(data.data(), data.shape(0), data.shape(1)).clone();
  else if (data.device_type() == nb::device::cuda::value)
    return Dataset<T>::referenceGPUData(data.data(), data.shape(0), data.shape(1), data.device_id())
        .clone();

  throw std::runtime_error("tensor given on unsupported device type.");
};

template <typename T>
consteval const char* get_name_for_dataset_type();

template <>
consteval const char* get_name_for_dataset_type<float>()
{
  return "FloatDataset";
};
template <>
consteval const char* get_name_for_dataset_type<uint8_t>()
{
  return "UCharDataset";
};
template <>
consteval const char* get_name_for_dataset_type<int32_t>()
{
  return "IntDataset";
};

NB_MODULE(GGNN, m)
{
  m.doc() = R"(GGNN: Graph-Based GPU Nearest Neighbor Search,
by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. Lensch,
Computer Graphics Group University of Tübingen,
published in IEEE Transactions on Big Data,
vol. 9, no. 1, pp. 267-279, 1 Feb. 2023,
doi: 10.1109/TBDATA.2022.3161156.

Refactored into a Python library by Lukas Ruppert, Deborah Kornwolf,
Computer Graphics Group University of Tübingen, 2025.

https://github.com/cgtuebingen/ggnn

GGNN performs nearest-neighbor computations on CUDA-capable GPUs.
It supports billion-scale datasets and can execute on multiple GPUs through sharding.
When using just a single GPU, data can be exchanged directly with other code
without copying through CPU memory (e.g., torch tensors).
)";

  m.def("set_log_level", [](int log_level) -> void { google::SetVLOGLevel("*", log_level); });

  nb::enum_<DistanceMeasure>(m, "DistanceMeasure")
      .value("Euclidean", DistanceMeasure::Euclidean)
      .value("Cosine", DistanceMeasure::Cosine);

#define DATASET_CLASS(T)                                                                         \
  nb::class_<Dataset<T>>(m, get_name_for_dataset_type<T>())                                      \
      .def("__init__",                                                                           \
           [](Dataset<T>* new_dataset, const NB2DArrayTorch<T>& data) {                          \
             new (new_dataset) Dataset<T>{ndarray_to_dataset(data)};                             \
           })                                                                                    \
      .def_static("load", &Dataset<T>::load, "file"_a, "from"_a = 0,                             \
                  "num"_a = std::numeric_limits<uint32_t>::max(), "pin_memory"_a = false)        \
      .def("store", &Dataset<T>::store, "file"_a)                                                \
      .def_prop_ro("N", [](const Dataset<T>& data) -> uint64_t { return data.N; })               \
      .def_prop_ro("D", [](const Dataset<uint8_t>& data) -> uint64_t { return data.D; })         \
      .def("numel", [](const Dataset<T>& data) -> size_t { return data.numel(); })               \
      .def("clone",                                                                              \
           [](const Dataset<T>& data) -> NB2DArrayTorch<T> {                                     \
             return dataset_to_ndarray(data.clone());                                            \
           })                                                                                    \
      .def_prop_ro("view", [](const Dataset<T>& data) { return dataset_to_ndarray_view(data); }) \
      .def_prop_ro("device", [](const Dataset<T>& data) -> std::string {                         \
        return data.isGPUAccessible() ? "cuda:" + std::to_string(data.gpu_id) : "cpu";           \
      });                                                                                        \
                                                                                                 \
  nb::implicitly_convertible<NB2DArrayCPU<T>, Dataset<T>>();                                     \
  nb::implicitly_convertible<NB2DArrayGPU<T>, Dataset<T>>();

  GGNN_EVAL(GGNN_BASES, DATASET_CLASS);
  GGNN_EVAL(GGNN_KEYS, DATASET_CLASS);

  nb::class_<GGNN<KeyT, ValueT>>(m, "GGNN")
      .def(nb::init<>())
      // set base
      // .def("set_base", &GGNN<KeyT, ValueT>::setBase, "base"_a)
      .def(
          "set_base",
          [](GGNN<KeyT, ValueT>& ggnn, Dataset<float>&& base) { ggnn.setBase(std::move(base)); },
          "base"_a)
      .def(
          "set_base",
          [](GGNN<KeyT, ValueT>& ggnn, Dataset<uint8_t>&& base) { ggnn.setBase(std::move(base)); },
          "base"_a)
      .def("set_working_directory", &GGNN<KeyT, ValueT>::setWorkingDirectory, "dir"_a)
      .def("set_cpu_memory_limit", &GGNN<KeyT, ValueT>::setCPUMemoryLimit, "memory_limit"_a)
      .def(
          "set_gpus",
          [](GGNN<KeyT, ValueT>& ggnn, const std::vector<int>& gpu_ids) -> void {
            ggnn.setGPUs(gpu_ids);
          },
          "gpu_ids"_a)
      .def("set_shard_size", &GGNN<KeyT, ValueT>::setShardSize, "n_shard"_a)
      .def("set_return_results_on_gpu", &GGNN<KeyT, ValueT>::setReturnResultsOnGPU,
           "return_results_on_gpu"_a = true)

      // graph construction
      .def("build", &GGNN<KeyT, ValueT>::build, "k_build"_a, "tau_build"_a,
           "refinement_iterations"_a = 2, "measure"_a = DistanceMeasure::Euclidean,
           "Build a GGNN graph.")
      .def("load", &GGNN<KeyT, ValueT>::load, "k_build"_a, "Load a GGNN graph.")
      .def("store", &GGNN<KeyT, ValueT>::store, "Store a GGNN graph.")

      // run queries
      .def(
          "query",
          [](GGNN<KeyT, ValueT>& ggnn, const Dataset<float>& query, const uint32_t KQuery,
             const float tau_query, const uint32_t max_iterations, const DistanceMeasure measure) {
            Results results = ggnn.query(query, KQuery, tau_query, max_iterations, measure);

            return std::make_tuple<>(dataset_to_ndarray(std::move(results.ids)),
                                     dataset_to_ndarray(std::move(results.dists)));
          },
          "query"_a, "k_query"_a, "tau_query"_a, "max_iterations"_a = 400,
          "measure"_a = DistanceMeasure::Euclidean, "Run a query and return indices and distances.")
      .def(
          "query",
          [](GGNN<KeyT, ValueT>& ggnn, const Dataset<uint8_t>& query, const uint32_t KQuery,
             const float tau_query, const uint32_t max_iterations, const DistanceMeasure measure) {
            Results results = ggnn.query(query, KQuery, tau_query, max_iterations, measure);

            return std::make_tuple<>(dataset_to_ndarray(std::move(results.ids)),
                                     dataset_to_ndarray(std::move(results.dists)));
          },
          "query"_a, "k_query"_a, "tau_query"_a, "max_iterations"_a = 400,
          "measure"_a = DistanceMeasure::Euclidean, "Run a query and return indices and distances.")
      .def(
          "bf_query",
          [](GGNN<KeyT, ValueT>& ggnn, const Dataset<float>& query, const uint32_t KGT,
             const DistanceMeasure measure) {
            Results results = ggnn.bfQuery(query, KGT, measure);

            return std::make_tuple<>(dataset_to_ndarray(std::move(results.ids)),
                                     dataset_to_ndarray(std::move(results.dists)));
          },
          "query"_a, "k_gt"_a = 100, "measure"_a = DistanceMeasure::Euclidean,
          "Run a brute-force query and indices and distances.")
      .def(
          "bf_query",
          [](GGNN<KeyT, ValueT>& ggnn, const Dataset<uint8_t>& query, const uint32_t KGT,
             const DistanceMeasure measure) {
            Results results = ggnn.bfQuery(query, KGT, measure);

            return std::make_tuple<>(dataset_to_ndarray(std::move(results.ids)),
                                     dataset_to_ndarray(std::move(results.dists)));
          },
          "query"_a, "k_gt"_a = 100, "measure"_a = DistanceMeasure::Euclidean,
          "Run a brute-force query and indices and distances.")

      // access the graph
      .def("get_graph", &GGNN<KeyT, ValueT>::getGraph, "on_gpu_shard_id"_a = 0,
           "Access the GGNN graph.", nb::rv_policy::reference_internal)
      .doc() =
      "GGNN main class. Provides functionality for building, loading, storing, and querying "
      "nearest neighbor graphs on the GPU.";
  ;

  nb::class_<Evaluator<KeyT, ValueT>>(m, "Evaluator")
      .def(nb::init<const Dataset<float>&, const Dataset<float>&, const Dataset<KeyT>&,
                    const uint32_t, const DistanceMeasure>(),
           "base"_a, "query"_a, "gt"_a, "k_query"_a, "measure"_a = DistanceMeasure::Euclidean)
      .def(nb::init<const Dataset<uint8_t>&, const Dataset<uint8_t>&, const Dataset<KeyT>&,
                    const uint32_t, const DistanceMeasure>(),
           "base"_a, "query"_a, "gt"_a, "k_query"_a, "measure"_a = DistanceMeasure::Euclidean)
      .def("evaluate_results", &Evaluator<KeyT, ValueT>::evaluateResults, "results"_a,
           "Evaluate the accuracy of a query result.");

  nb::class_<Evaluation>(m, "Evaluation")
      //.def(nb::init<>())
      .def_rw("k_query", &Evaluation::KQuery)
      .def_rw("c1", &Evaluation::c1)
      .def_rw("c1_dup", &Evaluation::c1_dup)
      .def_rw("c_k_query", &Evaluation::cKQuery)
      .def_rw("c_k_query_dup", &Evaluation::cKQuery_dup)
      .def_rw("r_k_query", &Evaluation::rKQuery)
      .def_rw("r_k_query_dup", &Evaluation::rKQuery_dup)
      .def("__repr__", [](const Evaluation& eval) -> std::string {
        std::stringstream ss;
        ss << eval;
        return ss.str();
      });

  nb::class_<Graph<KeyT, ValueT>>(m, "Graph")
      //.def(nb::init<>())
      .def_ro("graph", &Graph<KeyT, ValueT>::graph)
      .def_ro("selection", &Graph<KeyT, ValueT>::selection)
      .def_ro("translation", &Graph<KeyT, ValueT>::translation)
      .def_ro("nn1_stats", &Graph<KeyT, ValueT>::nn1_stats);
}
