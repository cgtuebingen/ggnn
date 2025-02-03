CMAKE_MINIMUM_REQUIRED(VERSION 3.25.2 FATAL_ERROR)

project (GGNN
  VERSION 0.9.1
  LANGUAGES CXX CUDA
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# CUDA toolkit version 12 required for C++20 support
find_package(CUDAToolkit 12 REQUIRED)

# CMake does not reject g++9 which only has incomplete experimental support for C++20
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
        message(FATAL_ERROR "GCC or Clang version 10 or higher required for C++20 support!")
    endif()
else()
    message(WARNING "Compilation has only been tested with GCC and Clang.")
endif()

# glog requires libunwind but does not look in the libunwind folder containing LLVM's version.
# This is a workaround for using e.g. libunwind-18-dev on Ubuntu 24.04.
find_path (Unwind_INCLUDE_DIR NAMES unwind.h libunwind.h PATH_SUFFIXES libunwind DOC "unwind include directory")

include(FetchContent)
# if not installed, fetch glog sources from github
FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG        7b134a5c82c0c0b5698bb6bf7a835b230c5638e4 # release 0.7.1
  FIND_PACKAGE_ARGS
)
FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG        e171aa2d15ed9eb17054558e0b3a6a413bb01067 # release 2.2.2
  FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(glog gflags)

set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags)
find_package(glog REQUIRED)

# optional (required for nanobind bindings)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module)

if (Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

    find_package(nanobind CONFIG)
endif()


# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No release type specified. Setting to 'Release'.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


# Configure CUDA architectures to compile for
if (NOT DEFINED GGNN_CUDA_ARCHITECTURES)
  if (DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    message(STATUS "Parsing TORCH_CUDA_ARCH_LIST: $ENV{TORCH_CUDA_ARCH_LIST}")
    string(REPLACE "." "" TORCH_CUDA_ARCHITECTURES $ENV{TORCH_CUDA_ARCH_LIST})
    string(REPLACE "  " " " TORCH_CUDA_ARCHITECTURES ${TORCH_CUDA_ARCHITECTURES})
    string(REPLACE " " "-real " TORCH_CUDA_ARCHITECTURES "${TORCH_CUDA_ARCHITECTURES} ")
    string(REPLACE "+PTX-real" " " TORCH_CUDA_ARCHITECTURES ${TORCH_CUDA_ARCHITECTURES})
    string(REPLACE " " ";" TORCH_CUDA_ARCHITECTURES ${TORCH_CUDA_ARCHITECTURES})
    set(GGNN_CUDA_ARCHITECTURES ${TORCH_CUDA_ARCHITECTURES} CACHE STRING "CUDA architecture(s) to compile for.")
  else()
    # adjust this based on your available GPUs
    # see https://developer.nvidia.com/cuda-gpus
    # 6.1 - 1080Ti
    # 7.0 - V100
    # 7.5 - 2080Ti
    # 8.6 - 3090
    # 8.9 - 4090
    # 9.0 - H100
    # the following are also allowed:
    # all-major
    # native
    set(GGNN_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architecture(s) to compile for.")
    mark_as_advanced(CMAKE_CUDA_ARCHITECTURES)
    # set_property(CACHE GGNN_CUDA_ARCHITECTURES PROPERTY STRINGS "all-major" "native" "61" "70" "75" "86" "89" "90")
  endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES ${GGNN_CUDA_ARCHITECTURES} CACHE STRING "CUDA architecture(s) to compile for." FORCE)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")


# Set CUDA flags based on build type
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
if (CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -Xptxas=-v")
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas=-v")
endif()

message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")


# make compile commands available to clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# don't hide the include paths in a separate file
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

add_library(GGNNBase SHARED)
target_sources(GGNNBase PRIVATE
    "src/ggnn/base/graph_config.cpp"
    "src/ggnn/base/graph.cpp"

    "src/ggnn/base/data.cu"
    "src/ggnn/base/dataset.cu"
    "src/ggnn/base/gpu_instance.cu"
    "src/ggnn/base/ggnn.cu"
    "src/ggnn/base/result_merger.cpp"
    "src/ggnn/base/eval.cpp"

    "include/ggnn/base/lib.h"
    "include/ggnn/base/result_merger.h"

    "include/ggnn/base/graph_config.h"
    "include/ggnn/base/graph.h"
    "include/ggnn/base/gpu_instance.cuh"
    )
  target_sources(GGNNBase PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS "include"
    FILES
    "include/ggnn/base/def.h"
    "include/ggnn/base/fwd.h"

    "include/ggnn/base/data.cuh"
    "include/ggnn/base/dataset.cuh"
    "include/ggnn/base/ggnn.cuh"
    "include/ggnn/base/eval.h"
    )

add_library(GGNNConstruction SHARED)
target_sources(GGNNConstruction PRIVATE
    "src/ggnn/construction/graph_construction.cu"

    "src/ggnn/construction/graph_buffer.cu"

    "src/ggnn/construction/top_merge_layer.cu"
    "src/ggnn/construction/merge_layer.cu"
    "src/ggnn/construction/wrs_select_layer.cu"
    "src/ggnn/construction/sym_buffer_merge_layer.cu"
    "src/ggnn/construction/sym_query_layer.cu"

    "include/ggnn/construction/graph_buffer.cuh"

    "include/ggnn/construction/top_merge_layer.cuh"
    "include/ggnn/construction/merge_layer.cuh"
    "include/ggnn/construction/wrs_select_layer.cuh"
    "include/ggnn/construction/sym_query_layer.cuh"
    "include/ggnn/construction/sym_buffer_merge_layer.cuh"

    "include/ggnn/cuda_utils/distance.cuh"
    "include/ggnn/cuda_utils/k_best_list.cuh"
    "include/ggnn/cuda_utils/simple_knn_cache.cuh"
    "include/ggnn/cuda_utils/simple_knn_sym_cache.cuh"
    "include/ggnn/cuda_utils/check.cuh"
    )
target_sources(GGNNConstruction PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS "include"
    FILES
    "include/ggnn/construction/graph_construction.cuh"
    )
add_library(GGNNQuery SHARED)
target_sources(GGNNQuery PRIVATE
    "src/ggnn/query/query_kernels.cu"

    "src/ggnn/query/bf_query_layer.cu"
    "src/ggnn/query/query_layer.cu"

    "include/ggnn/query/bf_query_layer.cuh"
    "include/ggnn/query/query_layer.cuh"

    "include/ggnn/cuda_utils/distance.cuh"
    "include/ggnn/cuda_utils/k_best_list.cuh"
    "include/ggnn/cuda_utils/simple_knn_cache.cuh"
    "include/ggnn/cuda_utils/check.cuh"
    )
target_sources(GGNNQuery PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS "include"
    FILES
    "include/ggnn/query/query_kernels.cuh"
    )

target_link_libraries(GGNNBase PRIVATE CUDA::curand glog::glog)
target_link_libraries(GGNNConstruction PRIVATE CUDA::curand glog::glog)
target_link_libraries(GGNNQuery PRIVATE glog::glog)
target_include_directories(GGNNBase PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(GGNNConstruction PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(GGNNQuery PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(GGNNBase PRIVATE GGNNConstruction GGNNQuery)
# look for libGGNNConstruction and libGGNNQuery in the same folder
set_property(TARGET GGNNBase APPEND PROPERTY INSTALL_RPATH "$ORIGIN")

if (nanobind_FOUND)
    message(STATUS "building nanobind module")
    nanobind_add_module(ggnn NB_STATIC NOMINSIZE "src/ggnn/python/nanobind.cu")
    target_link_libraries(ggnn PRIVATE GGNNBase GGNNConstruction GGNNQuery glog::glog)
    nanobind_add_stub(ggnn_stub MODULE ggnn OUTPUT ggnn.pyi MARKER_FILE py.typed DEPENDS ggnn)

    if (PY_BUILD_CMAKE_MODULE_NAME)
        message(STATUS "configuring python module installation")
        set_property(TARGET ggnn APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
        install(TARGETS GGNNBase GGNNConstruction GGNNQuery ggnn
            EXCLUDE_FROM_ALL
            COMPONENT python_modules
            DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME})
        # install glog if we had to compile it ourselves
        if (TARGET glog)
            install(TARGETS glog
                LIBRARY
                EXCLUDE_FROM_ALL
                COMPONENT python_modules
                DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME})
        endif()
        install(FILES ${CMAKE_BINARY_DIR}/py.typed ${CMAKE_BINARY_DIR}/ggnn.pyi
            EXCLUDE_FROM_ALL
            COMPONENT python_modules
            DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME})
    endif()
endif()

if (NOT PY_BUILD_CMAKE_MODULE_NAME)
  if (gflags_FOUND)
    set(files
      "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp-and-cuda/ggnn_main.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp-and-cuda/ggnn_main_gpu_data.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp-and-cuda/ggnn_main_multi_gpu.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp-and-cuda/ggnn_benchmark.cpp"
    )
    foreach(filename ${files})
        get_filename_component(EXECUTABLE_NAME "${filename}" NAME_WLE)
        add_executable(${EXECUTABLE_NAME} ${filename})
        target_link_libraries(${EXECUTABLE_NAME} PRIVATE GGNNBase glog::glog gflags::gflags)
    endforeach()

    target_link_libraries(ggnn_main_gpu_data PRIVATE CUDA::curand)
  else()
    message(STATUS "gflags not found. Skipping example projects.")
  endif()

  # Installation (enables find_package)
  # Introduce variables:
  # * CMAKE_INSTALL_LIBDIR
  # * CMAKE_INSTALL_BINDIR
  # * CMAKE_INSTALL_INCLUDEDIR
  include(GNUInstallDirs)
  # Include module with function 'write_basic_package_version_file'
  include(CMakePackageConfigHelpers)

  # directories where to store the generated files
  set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
  set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
  # Configuration
  set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
  set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
  set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
  set(namespace "${PROJECT_NAME}::")

  write_basic_package_version_file("${version_config}" COMPATIBILITY SameMajorVersion)
  configure_package_config_file(
    "cmake/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
  )

  install(
    TARGETS GGNNBase GGNNConstruction GGNNQuery
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILE_SET public_headers
  )
  install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
  )
  install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
  )
endif()
