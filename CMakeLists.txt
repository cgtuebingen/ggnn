CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )

project (GGNN)
set(CMAKE_CXX_STANDARD 14)


find_package(CUDA REQUIRED)

set(CUDA_ARCH_LIST Auto)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

set(EIGEN_INCLUDE_PATH "3rdparty/eigen-git-mirror")
set(CUB_INCLUDE_PATH "3rdparty/cub")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -Xptxas=-v --expt-relaxed-constexpr --default-stream per-thread" )

add_subdirectory("3rdparty/pybind11")
set(GFLAGS_NAMESPACE "google")
add_subdirectory("3rdparty/glog")

message("CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

cuda_add_executable(demo src/demo.cu)
target_include_directories(demo
  PRIVATE
  ${CUDA_NVCC_FLAGS}
  ${EIGEN_INCLUDE_PATH}
  ${CUB_INCLUDE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(demo ${CUDA_curand_LIBRARY} glog::glog)

