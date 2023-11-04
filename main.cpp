
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <chrono>

#include "no_kokkos.h"

int main(int argc, char** argv) {
    int mpi_rank, mpi_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    {
        loop();
    }
    MPI_Finalize();
}