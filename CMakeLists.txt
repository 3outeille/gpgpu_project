cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.5.2.tar.gz
  )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )

FetchContent_Declare(
  Thrust
  URL https://github.com/NVIDIA/thrust/archive/1.16.0.tar.gz
  )


if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)
FetchContent_MakeAvailable(Thrust)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
# find_package(CUDA REQUIRED)
find_package(TBB REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


add_library(renderer)
target_compile_features(renderer PUBLIC cxx_std_14)
target_link_libraries(renderer spdlog::spdlog)


target_sources(renderer PRIVATE src/render.hpp src/render_cpu.cpp src/matrix.hpp src/matrix.cpp src/render.cu src/matrix.cu src/matrix.cuh)
set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(harris src/harris.cpp)
target_compile_features(harris PUBLIC cxx_std_17)
target_link_libraries(harris PRIVATE Thrust::Thrust PNG::PNG CLI11::CLI11 spdlog::spdlog renderer)

add_executable(bench src/bench.cpp)
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench PRIVATE Thrust::Thrust spdlog::spdlog benchmark renderer)
