#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#define WINDOW 64

int main(int argc, char **argv) {

  int size, rank;
  int *src=NULL, *dst=NULL;
  int *ring_dst=NULL, *tmp=NULL;
  int send_peer, recv_peer;
  int i, j;
  int base_index;
  int stride;
  struct timeval start, end;
  float total_time=0;
  int max_size=0;
  MPI_Status status[WINDOW];
  MPI_Request send_request[WINDOW];
  MPI_Request recv_request[WINDOW];


  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  max_size = size;
  printf("Allreduce benchmark\n");

  /* allocate memory */
  src = (int *)malloc(max_size*sizeof(int));
  dst = (int *)malloc(max_size*sizeof(int));
  ring_dst = (int *)malloc(max_size*sizeof(int));
  tmp = (int *)malloc(max_size*sizeof(int));
  if (src == NULL || dst == NULL || ring_dst == NULL || tmp == NULL) {
     perror("cannot allocate memory\n");
     exit(0);
  }

  printf("max size=%d, size=%d\n", max_size, size);
  for (i=0; i < max_size; i++) {
      src[i] = rank*size + i;
      ring_dst[i] = 0;
      printf("rank=%d src index=%d has value %2d\n", rank, i, src[i]);
  }
   
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce (src, dst, max_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (i=0; i < max_size; i++) {
      printf("rank=%d dst index=%d has value %2d\n", rank, i, dst[i]);
  }


  memcpy(tmp, src, sizeof(int)*max_size);
  memcpy(ring_dst, src, sizeof(int)*max_size);

  stride = max_size/size;
  base_index = rank*stride;

  /* reduce-scatter phase */
  for(i=0;i<size-1;i++) {
    send_peer = (rank + 1) % size;
    recv_peer = (rank - 1 + size) % size;
   
    /* FIXME tmp/src index */
    MPI_Isend(&ring_dst[base_index], stride, MPI_INT, send_peer, 1, MPI_COMM_WORLD, &send_request[0]);
    
    /* FIXME the tmp array index */
    MPI_Irecv(&tmp[0], stride, MPI_INT, recv_peer, 1, MPI_COMM_WORLD, &recv_request[0]);
   
    MPI_Waitall(1, send_request, status);
    MPI_Waitall(1, recv_request, status);
    base_index = (base_index - stride + max_size)%max_size;


    /* reduction */
    for(j=0;j<stride;j++) {
    /* FIXME the ring_dst and tmp array index */
       ring_dst[base_index+j] += tmp[j];
     }
  }


  /* allgather phase */
for(i=0;i<size-1;i++) {
    send_peer = (rank + 1) % size;
    recv_peer = (rank - 1 + size) % size;
    
    /* FIXME tmp index */
    MPI_Isend(&ring_dst[base_index], stride, MPI_INT, send_peer, 1, MPI_COMM_WORLD, &send_request[0]);
    /* FIXME the ring_dst array index */
    MPI_Irecv(&ring_dst[(base_index - stride + max_size)%max_size], stride, MPI_INT, recv_peer, 1, MPI_COMM_WORLD, &recv_request[0]);
    
    MPI_Waitall(1, send_request, status);
    MPI_Waitall(1, recv_request, status);
    base_index = (base_index - stride + max_size)%max_size;
}

  if(memcmp(dst, ring_dst, max_size*sizeof(int)))
	printf("Ring Algo ERROR\n");
  else 
  	printf("Ring Algo allreduce successful\n");

  MPI_Finalize();
  return 0;
}

