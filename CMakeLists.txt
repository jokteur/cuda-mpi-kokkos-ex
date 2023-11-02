cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)

project(cuda-mpi-kokkos-test)

# ------
# Kokkos
# ------
set(Kokkos_ENABLE_CUDA ON)
set(Kokkos_ENABLE_OPENMP ON)
set(Kokkos_ARCH_AMPERE80 ON)
add_subdirectory(kokkos)

# ---
# MPI
# ---
find_package(MPI)

message(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(example example.cpp)
target_include_directories(example PUBLIC kokkos MPI::MPI_CXX)
target_link_libraries(example kokkos MPI::MPI_CXX)