/*
 * Implementation of a ring-based global sum reduction using non-blocking MPI communications.
 * Each process sends its value around the ring, accumulating the global sum.
 * 
 * Key changes made:
 *
 * Added compound structure definition
 * Created MPI derived datatype using MPI_Type_create_struct
 * Used MPI_Get_address to calculate member displacements
 * Initialized floating-point value to (rank + 1)²
 * Modified communication to use compound type
 * Updated sum operations for both integer and double values
 * Added cleanup for MPI derived datatype
 */
#include <mpi.h>
#include <stdio.h>
#include <math.h>

#define TAG 0

/* Define the compound structure */
struct compound {
    int ival;
    double dval;
};

int main(int argc, char *argv[]) {
    int rank, size;
    int left_rank, right_rank;
    MPI_Request send_request, recv_request;
    MPI_Status status;

    /* Variables for creating MPI derived datatype */
    MPI_Datatype compound_type;
    MPI_Datatype type[2] = {MPI_INT, MPI_DOUBLE};    // Array of member types
    int blocklen[2] = {1, 1};                        // One of each type
    MPI_Aint disp[2], base;                          // For member displacements
    struct compound dummy;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Calculate displacements for struct members */
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.ival, &disp[0]);
    MPI_Get_address(&dummy.dval, &disp[1]);
    disp[0] = disp[0] - base;    // Displacement of int member
    disp[1] = disp[1] - base;    // Displacement of double member

    /* Create and commit the MPI derived datatype */
    MPI_Type_create_struct(2, blocklen, disp, type, &compound_type);
    MPI_Type_commit(&compound_type);

    /* Set up ring topology */
    left_rank = (rank - 1 + size) % size;
    right_rank = (rank + 1) % size;

    /* Initialize local values */
    struct compound local_val, received;
    local_val.ival = rank;
    local_val.dval = pow(rank + 1, 2);
    struct compound value_to_send = local_val;

    /* Ring communication */
    for (int i = 0; i < size - 1; i++) {
        MPI_Irecv(&received,      // Buffer to store received data
            1,              // Number of elements to receive
            compound_type,   // MPI datatype we created
            left_rank,      // Source rank (who we're receiving from)
            TAG,              // Message tag
            MPI_COMM_WORLD, // Communicator
            &recv_request); // Request handle for completion check
            MPI_Issend(&value_to_send,  // Buffer containing data to send
                1,               // Number of elements to send
                compound_type,    // MPI datatype we created
                right_rank,      // Destination rank
                TAG,               // Message tag
                MPI_COMM_WORLD,  // Communicator
                &send_request);  // Request handle for completion check

        MPI_Wait(&recv_request, &status);
        MPI_Wait(&send_request, &status);

        /* Update local sums */
        local_val.ival += received.ival;
        local_val.dval += received.dval;
        value_to_send = received;
    }

    printf("Rank %d final sums: ival=%d, dval=%f\n", 
           rank, local_val.ival, local_val.dval);

    /* Clean up */
    MPI_Type_free(&compound_type);
    MPI_Finalize();
    return 0;
}
