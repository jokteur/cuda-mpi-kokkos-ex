#include <Kokkos_Core.hpp>
#include <iostream>
#include <mpi.h>
#include <stdio.h>


#define BLOCKSIZE 1024

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void times2(double* a, double* b) {
    int i = blockIdx.x;
    b[i] = 2 * a[i];
}

int main(int argc, char** argv) {
    int mpi_rank, mpi_size;

    size_t N = 4194304;
    size_t num_iterations = 500000;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    Kokkos::initialize(argc, argv);
    {
        cudaSetDevice(mpi_rank % 4);
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        double* A = (double*)malloc(sizeof(double) * N);
        double* B = (double*)malloc(sizeof(double) * N);
        for (size_t i = 0; i < N; i++) {
            A[i] = 1.5;
            B[i] = 0;
        }
        double* dA, * dB;
        cudaMalloc(&dA, sizeof(double) * N); checkCUDAError("Error allocating dA");
        cudaMemcpy(dA, A, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Error copying A");
        cudaMalloc(&dB, sizeof(double) * N); checkCUDAError("Error allocating dB");
        cudaMemcpy(dB, B, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Error copying B");

        Kokkos::Timer timer;

        if (mpi_rank == 0)
            std::cout << "===== Working parallel ===== " << std::endl;

        for (size_t i = 0; i < num_iterations;i++) {
            times2 << <N / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (dA, dB);
            times2 << <N / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (dB, dA);
        }
        Kokkos::fence();
        // cudaDeviceSynchronize();
        std::cout << "From rank:" << mpi_rank << ", time to execution: " << timer.seconds() << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        timer.reset();

        if (mpi_rank == 0)
            std::cout << "===== Bugged parallel (local sync) ===== " << std::endl;
        for (size_t i = 0; i < num_iterations;i++) {
            times2 << <N / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (dA, dB);
            times2 << <N / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (dB, dA);
            cudaStreamSynchronize(Kokkos::Cuda().cuda_stream());
        }
        Kokkos::fence();
        std::cout << "From rank:" << mpi_rank << ", time to execution: " << timer.seconds() << std::endl;
    }
    Kokkos::finalize();
    MPI_Finalize();
}