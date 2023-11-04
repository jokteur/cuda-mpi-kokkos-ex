// #include <Kokkos_Core.hpp>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>


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

void loop() {
    size_t num_iterations = 500000;
    int mpi_rank, mpi_size;

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Set cuda device from mpi rank
    cudaSetDevice(mpi_rank);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t N = 4194304;

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


    if (mpi_rank == 0)
        std::cout << "===== Working parallel ===== " << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_iterations;i++) {
        times2 << <N / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (dA, dB);
        times2 << <N / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (dB, dA);
    }
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "From rank:" << mpi_rank << ", time to execution: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0 << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0)
        std::cout << "===== Bugged parallel (local sync) ===== " << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_iterations;i++) {
        times2 << <N / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (dA, dB);
        times2 << <N / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (dB, dA);
        cudaStreamSynchronize(stream);
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "From rank:" << mpi_rank << ", time to execution: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0 << std::endl;
    // Kokkos::fence();
    cudaFree(dA);
    cudaFree(dB);
    free(A);
    free(B);
}