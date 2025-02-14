/**
 * Implementation of a ring-based global sum reduction using:
 * - Non-blocking MPI communications
 * - 1D Cartesian topology with periodic boundaries
 * - MPI_Cart_shift for neighbor rank computation
 *
 * INPUT PARAMETERS:
 * - argc, argv: Command line arguments
 * - size: Number of processes in the Cartesian grid
 * - ndims: Number of dimensions (1 for ring topology)
 * - periods: Periodicity of boundaries (true for ring)
 *
 * OUTPUT:
 * - Global sum computed across all processes
 *
* COMPILE WITH: make
* RUN WITH: sbatch cirrusrun.job
*/

#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) {

    /* Process rank, total number of processes, and neighbor ranks in Cartesian topology */
    int rank, size, left_rank, right_rank; 

    
    /* Local sum and received value from neighbor */
    int local_sum, received;

    /* Request handles for non-blocking communications */
    MPI_Request send_request, recv_request;
    MPI_Status status;

    /* Cartesian topology parameters */
    MPI_Comm cart_comm; 
    int ndims = 1; // 1D ring topology
    int periods[1] = {1}; // Periodic boundaries (true)
    int dims[1]; // array holding dimension size
    int reorder = 1; // Allow reordering of process ranks

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    dims[0] = size;  /* Set dimension size to number of processes */
    
    /* Create Cartesian communicator for ring topology */
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);

    /* Get nearest neighbors using Cartesian shift */
    MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);

    /* Initialize local sum with process rank */
    local_sum = rank;
    int value_to_send = rank;

    /* Ring communication: rotate values size-1 times to complete global sum */
    for (int i = 0; i < size - 1; i++) {
        /* Post non-blocking receive before send to avoid deadlock */
        MPI_Irecv(&received, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &recv_request);
        
        /* Post non-blocking synchronous send to right neighbor */
        MPI_Issend(&value_to_send, 1, MPI_INT, right, 0, MPI_COMM_WORLD, &send_request);

        /* Wait for both operations to complete before proceeding */
        MPI_Wait(&recv_request, &status);
        MPI_Wait(&send_request, &status);

        /* Update local sum with received value and prepare for next iteration */
        local_sum += received;
        value_to_send = received;
    }
    
    /* Print final result for each process */
    printf("Rank %d (Cart) final sum: %d\n", rank, local_sum);
    
    /* Clean up and exit */
    MPI_Comm_free(&cart_comm);   
    MPI_Finalize();
    return 0;
}