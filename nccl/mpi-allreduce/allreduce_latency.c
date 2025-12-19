#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_SIZE (1024*1024*4)
#define ITERATIONS 1000
#define WARMUP     10
int main(int argc, char **argv) {

	int size, rank;
	int *src=NULL, *dst=NULL;
	int i;
	MPI_Status status;
	struct timeval start, end;
	float total_time=0;
	int max_size=0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	max_size = size;

	printf("Allreduce benchmark\n");

	/* allocate memory */
	src = (int *)malloc(max_size*sizeof(int));
	dst = (int *)malloc(max_size*sizeof(int));
	if (src == NULL || dst == NULL) {
		perror("cannot allocate memory\n");
		exit(0);
	}

	for (i=0; i < max_size; i++) {
		src[i] = rank;
		// printf("rank=%d src index=%d has value %2d\n", rank, i, src[i]);
	}

	/* warmup */
		MPI_Barrier(MPI_COMM_WORLD);
		for (i=0; i < WARMUP; i++) {
			MPI_Allreduce (src, dst, max_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		}

	/* benchmark */

		printf("%10s   %10s usecs\n", "Size", "Latency (usecs)");
		/* synchronize all ranks */
		MPI_Barrier(MPI_COMM_WORLD);

		gettimeofday(&start, NULL);
		MPI_Barrier(MPI_COMM_WORLD);
		for (i=0; i < ITERATIONS; i++) {
			MPI_Allreduce (src, dst, max_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  
		}

		gettimeofday(&end, NULL);
		total_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));
		total_time /= ITERATIONS;
		printf("%10d   %10.2f\n", size, total_time);

	MPI_Finalize();
	return 0;
}