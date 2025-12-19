#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_SIZE (1024*1024*4)
#define ITERATIONS 10
#define WARMUP     10
#define WINDOW     32
int main(int argc, char **argv) {

	int size, rank, src_rank=0, dst_rank=1;
	char *src=NULL, *dst=NULL;
	int i, j;
	MPI_Status status[WINDOW];
	MPI_Request request[WINDOW];
	struct timeval start, end;
	float total_time=0;
	float total_bw=0;

	if(argc > 1) src_rank = atoi(argv[1]);
	if(argc > 2) dst_rank = atoi(argv[2]);
	for (i=0; i<WINDOW; i++)
		request[i] = MPI_REQUEST_NULL;

	printf("Bandwidth benchmark between src_rank=%d and dst_rank=%d\n", src_rank, dst_rank);

	/* allocate memory */
	src = (char *)malloc(MAX_SIZE);
	dst = (char *)malloc(MAX_SIZE);
	if (src == NULL || dst == NULL) {
		perror("cannot allocate memory\n");
		exit(0);
	}

	/* bandwidth benchmark is applicable only for two ranks */
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	/* warmup */
	for(int size=1; size <= MAX_SIZE; size *= 2) {
		for (i=0; i < WARMUP; i++) {
			if (rank == src_rank) {
				for (j = 0; j < WINDOW; j++) {
					MPI_Isend(src, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &request[j]); 
				}
				MPI_Recv(dst, 1, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &status[0]); 
			}
			else if(rank == dst_rank) {
				for (j = 0; j < WINDOW; j++) {
					MPI_Irecv(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &request[j]); 
				}
				MPI_Send(src, 1, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD); 	// non blocking because we need to wait for all the non-blocking calls to finish
			}  

			MPI_Waitall(WINDOW, request, status);
		}
	}
	/* benchmark */
	if(rank == src_rank)
		printf("%10s   %10s MB/s\n", "Size", "Bandwidth");
	for(int size=1; size <= MAX_SIZE; size *= 2) {
		/* synchronize all ranks */
		MPI_Barrier(MPI_COMM_WORLD);

		gettimeofday(&start, NULL);
		for (i=0; i < ITERATIONS; i++) {
			if (rank == src_rank) {
				for (j = 0; j < WINDOW; j++) {
					MPI_Isend(src, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &request[j]); 
				}
				MPI_Recv(dst, 1, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &status[0]); 
			}
			else if(rank == dst_rank) {
				for (j = 0; j < WINDOW; j++) {
					MPI_Irecv(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &request[j]); 
				}
				MPI_Send(src, 1, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD); 
			}    
			MPI_Waitall(WINDOW, request, status);
		}
		gettimeofday(&end, NULL);
		total_time = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
		total_time /= ITERATIONS;
		total_bw = (size*WINDOW)/total_time;	// in MB because time is in microseconds
		if(rank == src_rank)
		printf("%10d   %10.2f\n", size, total_bw);
	}

	MPI_Finalize();
	return 0;
}
