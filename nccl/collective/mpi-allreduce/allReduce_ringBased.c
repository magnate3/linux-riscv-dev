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

	MPI_Comm_size(MPI_COMM_WORLD, &size);		// get size (number of processes spawned)
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);		// get the rank of current process
	max_size = 4*size;
	printf("Allreduce benchmark\n");

	/* allocate memory */
	src = (int *)malloc(max_size*sizeof(int));
	ring_dst = (int *)malloc(max_size*sizeof(int));
	tmp_dst = (int *)malloc(max_size*sizeof(int));
	orig_dst = (int *)malloc(max_size*sizeof(int));
	if (src == NULL || ring_dst == NULL || tmp_dst == NULL || orig_dst == NULL) {
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

	// =========== Implement ring based allreduce algorithm ===============

	// (n - 1) steps for reduce scatter
	// (n - 1) steps for all gather
	// chunk size = (max_size / size)
	// process i will send chunk (r-i+n)%n from (i+1)%n
	// process i will receive chunk (r-i-1+n)%n from (i-1+n)%n

	for(i = 0; i < max_size; i++) {
		ring_dst[i] = src[i];
		tmp_dst[i] = 0;
	}

	int send_idx, recv_idx, send_chunk, recv_chunk;

	// reduce scatter
	MPI_Barrier(MPI_COMM_WORLD);
	int chunk_size = max_size / size;
	for(i = 0; i < size - 1; i++) {
		send_idx = (rank+1)%size;
		recv_idx = (rank-1+size)%size;
		send_chunk = (rank-i+size)%size;
		recv_chunk = (rank-i-1+size)%size;

		MPI_Isend(ring_dst + send_chunk*chunk_size, chunk_size, MPI_INT, send_idx, send_idx, MPI_COMM_WORLD, &request[0]);
		MPI_Irecv(tmp_dst + recv_chunk*chunk_size, chunk_size, MPI_INT, recv_idx, rank, MPI_COMM_WORLD, &request[1]);

		MPI_Waitall(2, request, status);

		for(int m = recv_chunk*chunk_size; m < recv_chunk*chunk_size + chunk_size; m++) {
			*(ring_dst + m) = *(ring_dst + m) + *(tmp_dst + m);
		}
	}

	// all gather
	MPI_Barrier(MPI_COMM_WORLD);
	for(i = 0; i < size - 1; i++) {
		send_idx = (rank+1)%size;
		recv_idx = (rank-1+size)%size;
		send_chunk = (rank+1-i+size)%size;
		recv_chunk = (rank+1-i-1+size)%size;

		MPI_Isend(ring_dst + send_chunk*chunk_size, chunk_size, MPI_INT, send_idx, send_idx, MPI_COMM_WORLD, &request[0]);
		MPI_Irecv(ring_dst + recv_chunk*chunk_size, chunk_size, MPI_INT, recv_idx, rank, MPI_COMM_WORLD, &request[1]);

		MPI_Waitall(2, request, status);
	}

	// ============ Check Correctness ===============

	if(memcmp(orig_dst, ring_dst, max_size*sizeof(int))) {
		printf("RING algorithm FAILED correctness\n");
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for (j=0; j < size; j++) {
		if (j==rank) {
		printf("RING algorithm rank=%d ", rank);
		for (i=0; i < max_size; i++) {
			printf("%d ", ring_dst[i]);
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
