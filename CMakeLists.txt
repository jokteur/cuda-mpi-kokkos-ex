cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)

# comment / uncomment the first part / second part

# -------------------
# Project with Kokkos
# -------------------

project(cuda-mpi-kokkos-test)
find_package(MPI)
add_subdirectory(kokkos)

add_executable(example example.cpp)
target_include_directories(example PUBLIC kokkos MPI::MPI_CXX)
target_link_libraries(example kokkos MPI::MPI_CXX)

# -----------------
# Project pure Cuda
# -----------------

# project(cuda-mpi-kokkos-test LANGUAGES CXX CUDA)
# find_package(MPI)
# add_executable(no_kokkos no_kokkos.cu no_kokkos.h
#   main.cpp
# )
# set_target_properties( no_kokkos PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
# target_include_directories(no_kokkos PUBLIC MPI::MPI_CXX)
# target_link_libraries(no_kokkos MPI::MPI_CXX)
