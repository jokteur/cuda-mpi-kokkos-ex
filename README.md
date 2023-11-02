# cuda-mpi-kokkos-example

`git clone https://github.com/jokteur/cuda-mpi-kokkos-ex.git --recurse-submodules`

Contains working and bugged example of multiple GPU parallelism with MPI, Cuda and Kokkos.

To build:
```
module load gcc
module load openmpi
module load cuda

cd cuda-mpi-kokkos-ex
mkdir build
cd build
cmake .. -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_AMPERE80=ON
make -j
```