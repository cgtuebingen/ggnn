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
// Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch
#ifndef CUDA_KNN_CORE_UTILS_CUH_
#define CUDA_KNN_CORE_UTILS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

// TODO(fabi) move to cuda utils
template <typename T>
__global__ void launcher(const T kernel) {
  kernel();
}

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#define CUDA_CALL(x)                                  \
  do {                                                \
    if ((x) != cudaSuccess) {                         \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      return EXIT_FAILURE;                            \
    }                                                 \
  } while (0)
#define CURAND_CALL(x)                                \
  do {                                                \
    if ((x) != CURAND_STATUS_SUCCESS) {               \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      return EXIT_FAILURE;                            \
    }                                                 \
  } while (0)

#define gpuDBG(ans) \
  if (DBG) {        \
    ans             \
  }

template <typename T>
float time_launcher(const int log_level, T* kernel, int N) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  kernel->launch();
  cudaEventRecord(stop);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  lprintf(log_level, "ms: %f for %d queries -> %f ms/query \n", milliseconds,
          N, milliseconds / N);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return milliseconds;
}

template <typename T>
float time_launcher(const int log_level, T& kernel, int N) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  kernel.launch();
  cudaEventRecord(stop);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  lprintf(log_level, "ms: %f for %d queries -> %f ms/query \n", milliseconds,
          N, milliseconds / N);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return milliseconds;
}

/**
 *  KNN UTILS:
 *  KeyT -> id type (int)
 *  ValueT -> distance type (float)
 */

/*
 * DistPair
 */
template <typename DistT, typename IdT>
struct DistPair {
  __device__ __forceinline__ DistPair(const DistT dist, const IdT id)
      : dist(dist), id(id) {}

  const DistT dist;
  const IdT id;
};

template <int A, int B>
struct get_power {
  static const int value = A * get_power<A, B - 1>::value;
};
template <int A>
struct get_power<A, 0> {
  static const int value = 1;
};

#endif  // CUDA_KNN_CORE_UTILS_CUH_
