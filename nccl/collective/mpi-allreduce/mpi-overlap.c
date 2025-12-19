#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_SIZE (1024*1024*128)
#define ITERATIONS 10
int main(int argc, char **argv) {

  int size, rank, src_rank=0, dst_rank=1;
  char *src=NULL, *dst=NULL;
  int i, j, m , n;
  MPI_Status status;
  MPI_Request send_request, recv_request;

  struct timeval start, end;
  float total_time=0;
  double a=1.98, b=1.97, c=1.0;
  float compute_time, comms_time, overlap;

  if(argc > 1) src_rank = atoi(argv[1]);
  if(argc > 2) dst_rank = atoi(argv[2]);

  printf("Overlap benchmark between src_rank=%d and dst_rank=%d\n", src_rank, dst_rank);

  /* allocate memory */
  src = (char *)malloc(MAX_SIZE);
  dst = (char *)malloc(MAX_SIZE);
  if (src == NULL || dst == NULL) {
     perror("cannot allocate memory\n");
     exit(0);
  }

  /* overlap benchmark is applicable only for two ranks */
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);

  for (i=0; i < ITERATIONS; i++) {
    for(m=0;m<100;m++)
      for(n=0;n<1000;n++)
         for(j=0;j<1000;j++)
             c += a*b;
  }

  gettimeofday(&start, NULL);
  for (i=0; i < ITERATIONS; i++) {
    for(m=0;m<100;m++)
      for(n=0;n<1000;n++) 
         for(j=0;j<1000;j++)
             c += a*b;
  }
  gettimeofday(&end, NULL);
  compute_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));
  compute_time /= ITERATIONS;

  printf("Compute Time %f usecs %f\n", compute_time, c);

  /* benchmark */
  if(rank == src_rank)
    printf("%10s   %10s usecs\n", "Size", "Latency (usecs) Overlap (%)");
  for(int size=1024*1024*16; size <= MAX_SIZE; size *= 2) {
    /* synchronize all ranks */
    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&start, NULL);
    for (i=0; i < ITERATIONS; i++) {
       if (rank == src_rank) {
         MPI_Irecv(dst, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &recv_request);
         MPI_Isend(dst, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &send_request);
       }
       else if(rank == dst_rank) {
         MPI_Irecv(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &recv_request);
         MPI_Isend(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &send_request);
       }
       MPI_Wait(&send_request, &status);
       MPI_Wait(&recv_request, &status);
    }
    gettimeofday(&end, NULL);
    comms_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));
    comms_time /= ITERATIONS;

    gettimeofday(&start, NULL);
    for (i=0; i < ITERATIONS; i++) {
#if 1
       MPI_Barrier(MPI_COMM_WORLD);
       if (rank == src_rank) {
         MPI_Irecv(dst, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &recv_request);
         MPI_Isend(dst, size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &send_request);
       }
       else if(rank == dst_rank) {
         MPI_Irecv(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &recv_request);
         MPI_Isend(dst, size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &send_request);
       }
#endif
       for(m=0;m<100;m++)
          for(n=0;n<1000;n++)
             for(j=0;j<1000;j++)
                c += a*b;
       MPI_Wait(&send_request, &status);
       MPI_Wait(&recv_request, &status);
    }
    gettimeofday(&end, NULL);

    total_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));
    total_time /= ITERATIONS;
    overlap = (((comms_time - (total_time - compute_time)) / comms_time) * 100.0); 
    if(overlap < 0) overlap = 0;

    printf("[rank:%2d] size=%10d total=%10.2f comms=%10.2f compute=%10.2f overlap=%10.2f %f\n", rank, size, total_time, comms_time, compute_time, overlap, c);
  }

  MPI_Finalize();
  return 0;
}
