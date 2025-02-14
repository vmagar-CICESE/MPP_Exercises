#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Modify global_sum code so that we do the computations with MPI_Reduce.
 * Add global_sum variable to store the result of the reduction.
 * Initialize local_sum with rank.
 * Use MPI_Reduce to compute the global sum with:
 * local_value: Input value from each process
 * global_sum: Output value from the root process
 * MPI_COMM_WORLD: communicator
 * MPI_SUM: operation
 * MPI_INT: datatype
 * MPI_ROOT: rank
 * 0: Root process rank
 * Compile with make, run with sbatch cirrusrun.job
 */


int main(int argc, char *argv[]) {
    int rank, size;
    int local_value, global_sum; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Initialize local_value with rank
    local_value = rank;

    // Perform glocal sum reduction with MPI_Reduce. We don't need to check for rank 0 
    MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print the result on rank 0    
    if (rank == 0){
        printf("Global sum: %d\n", global_sum);
    }
    
    MPI_Finalize();
    return 0;
}