// IMT2020065 (Shridhar Sharma)

// Works only for power of 2 nodes
// Compile with -> mpicc -O3 allReduce_doubleBinaryTree.c -lm
// Run with -> mpirun -np [number of nodes] ./a.out
			 
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {

	int size, rank;
	int max_size;
	int* src = NULL, *orig_dst=NULL, *temp_buff=NULL, *dbt_dst=NULL;
	MPI_Status status[3];
	MPI_Request request[3];
	struct timeval start, end;
	float total_time = 0.0;
	int i, j;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	max_size = 1024;

	// allocate memory
	src = (int *)malloc(max_size * sizeof(int));
	orig_dst = (int *)malloc(max_size * sizeof(int));
	temp_buff = (int *)malloc(max_size * sizeof(int));
	dbt_dst = (int *)malloc(max_size * sizeof(int));
	if(src == NULL || orig_dst == NULL || temp_buff == NULL || dbt_dst == NULL) {
		perror("Cannot allocate memory\n");
		exit(0);
	}

	// initialize buffers
	// printf("Rank = %d has src values = ", rank);
	for(i = 0; i < max_size; i++) {
		src[i] = rank + i;
		dbt_dst[i] = src[i];
		temp_buff[i] = 0;
		// printf("%2d ", src[i]);
	}
	// printf("\n");

	// MPI AllReduce
	gettimeofday(&start, NULL);
	MPI_Allreduce(src, orig_dst, max_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&end, NULL);
	total_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));
	if(rank == 0) {
		printf("MPI_Allreduce (%d nodes, %d bytes message size) Time taken : %f (usecs)\n", size, max_size * 4, total_time);
	}

	// Double Binary Tree Algorithm (Works only for power of 2 nodes)
	
	// ========================= Reduce Scatter ================================

	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&start, NULL);

	int height = 0, peer = 0;
	int steps = (int) log2(size) - 1;
	int index = ((rank % 2 == 0) ? rank : rank + 1);	// This index is used to make calculations easier for even and odd nodes in both trees
	if(rank > 0) {
		while((index & (1 << height)) == 0)
			height++;	
	}

	// First step is to send data from leaf nodes (slightly different from other iterations)
	index = ((rank % 2 == 0) ? rank : rank - 1);		
	peer = ((index % 4 == 0) ? rank + 1 : rank - 1);	// calculates if the current node is left or right child of it's parent

	MPI_Isend(dbt_dst, max_size, MPI_INT, peer, peer, MPI_COMM_WORLD, &request[0]);

	// Nodes at height 1 will receive data from leaf nodes
	if(height == 1) {
		MPI_Irecv(dbt_dst, max_size, MPI_INT, rank - height, MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
		MPI_Irecv(temp_buff, max_size, MPI_INT, rank + height, MPI_ANY_TAG, MPI_COMM_WORLD, &request[2]);
		MPI_Waitall(3, request, status);

		#pragma omp paraller for
		for(j = 0; j < max_size; j++) {
			dbt_dst[j] += temp_buff[j];
		}
	}

	else {
		MPI_Waitall(1, request, status);	
	}

	int child_offset = 0;	// Child offset is used to calculate the indices of child nodes
	index = ((rank % 2 == 0) ? rank : rank + 1);

	for(i = 1; i < steps; i++) {

		if(height == i) {
			// send
			peer = ((index - (1 << height)) % (1 << (2 + height))) == 0 ? rank + (1 << height) : rank - (1 << height);
			MPI_Isend(dbt_dst, max_size, MPI_INT, peer, peer, MPI_COMM_WORLD, &request[0]);
			MPI_Waitall(1, request, status);
		}

		if(height == i + 1) {
			// receive
			child_offset = (1 << (height - 1));
			MPI_Irecv(dbt_dst, max_size, MPI_INT, rank - child_offset, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
			MPI_Irecv(temp_buff, max_size, MPI_INT, rank + child_offset, MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
			MPI_Waitall(2, request, status);

			#pragma omp paraller for
			for(j = 0; j < max_size; j++) {
				dbt_dst[j] += temp_buff[j];
			}
		}
	}

	// Now the roots of both trees should send data to node 0
	// Alternatively we can send it to both node 0 and (size - 1).
	if(rank == (size / 2) || rank == (size / 2 - 1)) {
		MPI_Isend(dbt_dst, max_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &request[0]);
		MPI_Waitall(1, request, status);
	}

	if(rank == 0) {
		MPI_Irecv(dbt_dst, max_size, MPI_INT, size / 2, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
		MPI_Irecv(temp_buff, max_size, MPI_INT, size / 2 - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
		MPI_Waitall(2, request, status);

		#pragma omp paraller for
		for(j = 0; j < max_size; j++) {
			dbt_dst[j] += temp_buff[j];
		}
	}

	// Rank 0 will send reduced data back to roots and Rank (size - 1)
	if(rank == 0) {
		MPI_Isend(dbt_dst, max_size, MPI_INT, size / 2, 0, MPI_COMM_WORLD, &request[0]);
		MPI_Isend(dbt_dst, max_size, MPI_INT, size / 2 - 1, 0, MPI_COMM_WORLD, &request[1]);
		MPI_Isend(dbt_dst, max_size, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &request[2]);
		MPI_Waitall(3, request, status);
	}

	if(rank == (size / 2) || rank == (size / 2 - 1) || rank == size - 1) {
		MPI_Irecv(dbt_dst, max_size, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
		MPI_Waitall(1, request, status);
	}

	// MPI_Barrier(MPI_COMM_WORLD);
	// ========================= All Gather ================================

	for(i = steps; i > 1; i--) {

		if(height == i) {
			// send
			child_offset = (1 << (height - 1));
			MPI_Isend(dbt_dst, max_size, MPI_INT, rank - child_offset, rank, MPI_COMM_WORLD, &request[0]);
			MPI_Isend(dbt_dst, max_size, MPI_INT, rank + child_offset, rank, MPI_COMM_WORLD, &request[1]);
			MPI_Waitall(2, request, status);
		}

		if(height == i - 1) {
			// receive
			peer = ((index - (1 << height)) % (1 << (2 + height))) == 0 ? rank + (1 << height) : rank - (1 << height);
			MPI_Irecv(dbt_dst, max_size, MPI_INT, peer, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
			MPI_Waitall(1, request, status);
		}
	}

	// ========================== Timing and Correctness ==================================
	if(rank == 0) {
		gettimeofday(&end, NULL);
		total_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));
		printf("Double Binary Tree (%d nodes, %d bytes message size) Time taken : %f (usecs)\n", size, max_size * 4, total_time);
	}

	if(memcmp(orig_dst, dbt_dst, max_size*sizeof(int))) {
		printf("Rank : %d failed correctness\n", rank);
	}
	
	MPI_Finalize();
	return 0;
}