/*
 * Implementation of a ring-based global sum reduction using non-blocking MPI communications.
 * Each process sends its value around the ring, accumulating the global sum.
 */

#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) {

    /* Process rank and total number of processes */ 
    int rank, size; 
    
    /* Left and right neighbor ranks in ring topology */   
    int left_rank, right_rank;
    
    /* Local sum and received value from neighbor */
    int local_sum, received;

    /* Request handles for non-blocking communications */
    MPI_Request send_request, recv_request;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Set up ring topology: each process connects to left and right neighbors */
    left_rank = (rank - 1 + size) % size; // Left neighbor wraps around to size-1 for rank 0
    right_rank = (rank + 1) % size; // Right neighbor wraps around to 0 for rank size-1

    /* Initialize local sum with process rank */
    local_sum = rank;
    int value_to_send = local_sum;

    /* Ring communication: rotate values size-1 times to complete global sum */
    for (int i = 0; i < size - 1; i++) {
        /* Post non-blocking receive before send to avoid deadlock */
        MPI_Irecv(&received, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, &recv_request);
        
        /* Post non-blocking synchronous send to right neighbor */
        MPI_Issend(&value_to_send, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, &send_request);

        /* Wait for both operations to complete before proceeding */
        MPI_Wait(&recv_request, &status);
        MPI_Wait(&send_request, &status);

        /* Update local sum with received value and prepare for next iteration */
        local_sum += received;
        value_to_send = received;
    }
    
    /* Print final result for each process */
    printf("Rank %d final sum: %d\n", rank, local_sum);
    
    MPI_Finalize();
    return 0;
}