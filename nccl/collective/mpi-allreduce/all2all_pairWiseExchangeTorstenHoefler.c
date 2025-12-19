#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_RANK (1024)
int main(int argc, char **argv) {

  int size, rank, peer;
  int *src=NULL, *dst=NULL, *tmp_dst=NULL;
  int i;
  MPI_Status status[MAX_RANK];
  MPI_Request send_request[MAX_RANK];
  MPI_Request recv_request[MAX_RANK];
  struct timeval start, end;
  float total_time=0;
  int max_size=0;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  max_size = size;

  /* allocate memory */
  src = (int *)malloc(max_size*sizeof(int));
  dst = (int *)malloc(max_size*sizeof(int));
  tmp_dst = (int *)malloc(max_size*sizeof(int));
  if (src == NULL || dst == NULL || tmp_dst == NULL) {
     perror("cannot allocate memory\n");
     exit(0);
  }

  if(!rank) printf("MPI All2All benchmark\n");
  printf("rank=%d has src value ", rank);
  for (i=0; i < max_size; i++) {
      src[i]     = (size*rank)+i;
      tmp_dst[i] = 0;
      dst[i]     = 0;
      printf("%d ", src[i]);
  }
  printf("\n");
     
  MPI_Alltoall (src, max_size/size, MPI_INT, dst, max_size/size, MPI_INT, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  printf("rank=%d has dst value ", rank);
  for (i=0; i < max_size; i++) {
      printf("%d ", dst[i]);
  }
  printf("\n");
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(5);
  printf("\n");

  /* linear exchange algorithm */
  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) printf("All2All linear exchange algorithm\n");
  printf("rank=%d has src value ", rank);
  for (i=0; i < max_size; i++) {
      tmp_dst[i] = 0;
      dst[i]     = 0;
      printf("%d ", src[i]);
  }
  printf("\n");

  MPI_Barrier(MPI_COMM_WORLD);

  for (i=0; i < size; i++) {
      peer = ((rank%2 + i%2)%2)*(size/2)+(rank/2 + i/2)%(size/2);	// modified from i+1
      //printf("rank=%d step=%d peer=%d\n", rank, i, peer);
      MPI_Irecv(&tmp_dst[peer], max_size/size, MPI_INT, peer, rank, MPI_COMM_WORLD, &recv_request[peer]);
      MPI_Isend(&src[peer], max_size/size, MPI_INT, peer, peer, MPI_COMM_WORLD, &send_request[peer]);
  }
  MPI_Waitall(size, send_request, status);
  MPI_Waitall(size, recv_request, status);
  MPI_Barrier(MPI_COMM_WORLD);

  printf("rank=%d has tmp dst value ", rank);
  for (i=0; i < max_size; i++) {
      printf("%d ", tmp_dst[i]);
  }
  printf("\n");

  /*  TODO: Implement pair-wise exchange */
  /*  TODO: Implement optimal half bisection bandwidth algorithm */

  MPI_Finalize();
  return 0;
}
