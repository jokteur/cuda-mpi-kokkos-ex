#include <Kokkos_Core.hpp>
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    int mpi_rank, mpi_size;

    size_t num_particles = 5000000;
    size_t num_iterations = 100000;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    Kokkos::initialize(argc, argv);
    {
        Kokkos::Timer timer;
        if (mpi_rank == 0)
            std::cout << "===== Working parallel ===== " << std::endl;
        Kokkos::View<double*> array("test_array", num_particles);
        Kokkos::parallel_for("fill", num_particles, KOKKOS_LAMBDA(const size_t& n){
            array(n) = 1;Kokkos::log(n);
        });
        for (size_t i = 0; i < num_iterations;i++) {
            Kokkos::parallel_for("op", num_particles, KOKKOS_LAMBDA(const size_t& n) {
                array(n) += Kokkos::cos(array(n));
            });
            // Simulate two kernels, for equivalence to the next tests
            Kokkos::parallel_for("op", num_particles, KOKKOS_LAMBDA(const size_t& n) {
                array(n) += Kokkos::sin(array(n));
            });
        }
        Kokkos::fence();
        std::cout << "From rank:" << mpi_rank << ", time to execution: " << timer.seconds() << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        timer.reset();

        if (mpi_rank == 0)
            std::cout << "===== Bugged parallel (local reduction) ===== " << std::endl;
        double sum = 0.0;
        for (size_t i = 0; i < num_iterations;i++) {
            Kokkos::parallel_for("op", num_particles, KOKKOS_LAMBDA(const size_t& n) {
                array(n) += Kokkos::cos(array(n));
            });
            double result;
            Kokkos::parallel_reduce("sum", num_particles, KOKKOS_LAMBDA(const size_t& n, double& lsum) {
                lsum += array(n);
            }, result);
            sum += result;
        }
        Kokkos::fence();
        std::cout << "From rank:" << mpi_rank << ", time to execution: " << timer.seconds() << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        timer.reset();

        if (mpi_rank == 0)
            std::cout << "===== Bugged parallel (simple local fence, no reduction) ===== " << std::endl;
        for (size_t i = 0; i < num_iterations;i++) {
            Kokkos::parallel_for("op", num_particles, KOKKOS_LAMBDA(const size_t& n) {
                array(n) += Kokkos::cos(array(n));
            });
            // To compensate for no reduction, we add a normal numerical kernel to keep the timing consistent
            Kokkos::parallel_for("op", num_particles, KOKKOS_LAMBDA(const size_t& n) {
                array(n) += Kokkos::sin(array(n));
            });
            Kokkos::fence();
        }
        Kokkos::fence();
        std::cout << "From rank:" << mpi_rank << ", time to execution: " << timer.seconds() << std::endl;
    }
    Kokkos::finalize();
    MPI_Finalize();
}