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
#ifndef INCLUDE_GGNN_UTILS_CUDA_KNN_UTILS_CUH_
#define INCLUDE_GGNN_UTILS_CUDA_KNN_UTILS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <glog/logging.h>

enum DistanceMeasure : int {
  Euclidean = 0,
  Cosine = 1
};

template <typename T>
__global__ void launcher(const T kernel) {
  kernel();
}

#define CHECK_CUDA(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    if (abort)
      LOG(FATAL) << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
    else
      LOG(ERROR) << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
  }
}

template <typename T>
float time_launcher(const int log_level, T* kernel, int N, cudaStream_t stream = 0) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  kernel->launch(stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  VLOG(log_level) << milliseconds << " ms for " << N << " queries -> " << milliseconds*1000.0f/N << " us/query \n";
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return milliseconds;
}

template <typename T>
void launcher(const int log_level, T* kernel, int N, cudaStream_t stream = 0) {
  kernel->launch(stream);
}

#endif  // INCLUDE_GGNN_UTILS_CUDA_KNN_UTILS_CUH_
