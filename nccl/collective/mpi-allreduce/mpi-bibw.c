#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_SIZE (1024*1024*4)
#define ITERATIONS 20
#define WARMUP     10
#define WINDOW     32
int main(int argc, char **argv) {

  	int size, rank, src_rank=0, dst_rank=1;
  	char *src=NULL, *dst=NULL;
  	int i, j;
  	MPI_Status status[WINDOW];
  	MPI_Request send_request[WINDOW];		
  	MPI_Request recv_request[WINDOW];		
  	struct timeval start, end;
  	float total_time=0;
  	float total_bibw=0;

  	if(argc > 1) src_rank = atoi(argv[1]);
  	if(argc > 2) dst_rank = atoi(argv[2]);
  	for (i=0; i<WINDOW; i++) {
  	    send_request[i] = MPI_REQUEST_NULL;
  	    recv_request[i] = MPI_REQUEST_NULL;
  	}

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
				MPI_Irecv(dst, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &recv_request[j]);
				MPI_Isend(src, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &send_request[j]); 		// both non-blocking for bidirectional bandwidth
			}
		}
		else if(rank == dst_rank) {
			for (j = 0; j < WINDOW; j++) {
				MPI_Irecv(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &recv_request[j]); 
				MPI_Isend(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &send_request[j]); 
			}
		}    
		MPI_Waitall(WINDOW, send_request, status);
		MPI_Waitall(WINDOW, recv_request, status);
		}
	}
	/* benchmark */
	if(rank == src_rank)
		printf("%10s   %10s MB/s\n", "Size", "Bidirectional Bandwidth");
	for(int size=1; size <= MAX_SIZE; size *= 2) {
		/* synchronize all ranks */
		MPI_Barrier(MPI_COMM_WORLD);

		gettimeofday(&start, NULL);
		for (i=0; i < ITERATIONS; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == src_rank) {
			for (j = 0; j < WINDOW; j++) {
				MPI_Irecv(dst, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &recv_request[j]);
				MPI_Isend(src, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &send_request[j]); 
			}
		}
		else if(rank == dst_rank) {
			for (j = 0; j < WINDOW; j++) {
				MPI_Isend(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &send_request[j]); 
				MPI_Irecv(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &recv_request[j]); 
			}
		}    
		MPI_Waitall(WINDOW, send_request, status);
		MPI_Waitall(WINDOW, recv_request, status);
		}
		gettimeofday(&end, NULL);
		total_time = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
		total_time /= ITERATIONS;
		total_bibw = (2*size*WINDOW)/total_time;
		if(rank == src_rank)
		printf("%10d   %10.2f\n", size, total_bibw);
	}

	MPI_Finalize();
	return 0;
}
