#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

int main(int argc, char **argv) {

	int size, rank;
	int *src=NULL, *ring_dst=NULL, *tmp_dst=NULL, *orig_dst=NULL, *tree_dst=NULL;
	int *dst_buffer, *src_buffer;
	int i, j;
	MPI_Status status[2];
	MPI_Request request[2];
	struct timeval start, end;
	float total_time=0;
	int max_size=0;
	int index;
	int slice;
	int stages;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	max_size = 4*size;
	printf("Allreduce benchmark\n");

	/* allocate memory */
	src = (int *)malloc(max_size*sizeof(int));
	tmp_dst = (int *)malloc(max_size*sizeof(int));
	orig_dst = (int *)malloc(max_size*sizeof(int));
	tree_dst = (int *)malloc(max_size*sizeof(int));
	if (src == NULL || tmp_dst == NULL || orig_dst == NULL || tree_dst == NULL) {
		perror("cannot allocate memory\n");
		exit(0);
	}

	/* initialize the buffers */
	for (i=0; i < max_size; i++) {
		src[i] = rank+i;
		printf("rank=%d src index=%d has value %2d\n", rank, i, src[i]);
	}
		
	/* MPI allreduce */
	MPI_Allreduce (src, orig_dst, max_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	/* TODO: Implement rabenseifner tree algorithm allreduce */

	// number of steps for reduce scatter / all gather = log(size)
	// chunk_size = max_size / (pow(2, i))
	// peer = max_size ^ (pow(2, i))

	for(i = 0; i < max_size; i++) {
		tree_dst[i] = src[i];
		tmp_dst[i] = 0;
	}

	// reduce scatter
	MPI_Barrier(MPI_COMM_WORLD);

	int steps = (int) log2(size);
	int buffer_ptr = 0;
	for(i = 0; i < steps; i++) {
		int chunk_size = max_size / ((int)pow(2, i + 1));
		int peer = rank ^ ((int)pow(2, i));

		if(peer < rank) {	// right node
			// send the first half
			MPI_Isend(tree_dst + buffer_ptr, chunk_size, MPI_INT, peer, peer, MPI_COMM_WORLD, &request[0]);
			buffer_ptr += chunk_size;
		}
		else {
			// send the second half
			MPI_Isend(tree_dst + buffer_ptr + chunk_size, chunk_size, MPI_INT, peer, peer, MPI_COMM_WORLD, &request[0]);
		}

		MPI_Irecv(tmp_dst, chunk_size, MPI_INT, peer, rank, MPI_COMM_WORLD, &request[1]);

		MPI_Waitall(2, request, status);

		for(int j = 0; j < chunk_size; j++) {
			tree_dst[buffer_ptr + j] += tmp_dst[j];
		}
	}

	// all gather
	MPI_Barrier(MPI_COMM_WORLD);

	for(i = steps - 1; i >= 0; i--) {
		int chunk_size = max_size / ((int)pow(2, i + 1));
		int peer = rank ^ ((int)pow(2, i));

		MPI_Isend(tree_dst + buffer_ptr, chunk_size, MPI_INT, peer, peer, MPI_COMM_WORLD, &request[0]);

		if(peer < rank) {	// right node
			// receive in first half
			MPI_Irecv(tree_dst + buffer_ptr - chunk_size, chunk_size, MPI_INT, peer, rank, MPI_COMM_WORLD, &request[1]);
			buffer_ptr -= chunk_size;
		}
		else {
			// receive in second half
			MPI_Irecv(tree_dst + buffer_ptr + chunk_size, chunk_size, MPI_INT, peer, rank, MPI_COMM_WORLD, &request[1]);
		}

		MPI_Waitall(2, request, status);
	}

	if(memcmp(orig_dst, tree_dst, max_size*sizeof(int))) {
		printf("TREE algorithm FAILED correctness\n");
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for (j=0; j < size; j++) {
		if (j==rank) {
		printf("TREE algorithm rank=%d ", rank);
		for (i=0; i < max_size; i++) {
			printf("%d ", tree_dst[i]);
		}
		printf("\n");
		}
		fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for (j=0; j < size; j++) {
		if (j==rank) {
		printf("MPI  algorithm rank=%d ", rank);
		for (i=0; i < max_size; i++) {
			printf("%d ", orig_dst[i]);
		}
		printf("\n");
		}
		fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}
