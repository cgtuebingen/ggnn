CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

project (Example
  LANGUAGES CXX
)

# GGNN requires at least C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(GGNN REQUIRED)

add_executable(example ggnn_main.cpp)

# link to the GGNN library
target_link_libraries(example PRIVATE GGNN::GGNNBase)
