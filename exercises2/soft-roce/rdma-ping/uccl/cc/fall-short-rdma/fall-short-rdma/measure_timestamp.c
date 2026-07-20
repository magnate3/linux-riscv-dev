#include<stdio.h>
#include <rdma/rdma_cma.h>

#include "rudp.h"
int main(int argc, char **argv){
	struct ibv_device **dev_list = ibv_get_device_list(NULL) ;
	struct ibv_device *ib_dev = dev_list[0];
	struct ibv_context *ctx = ibv_open_device(ib_dev);
	struct timespec start, end;
	int i,j;
	for (j=0;j<2000;j++){
		long long unsigned min=1000000000L;
		long long unsigned max = 0L; 
		long long unsigned avg = 0L;
		for(i=0;i <2000; i++){
			clock_gettime(CLOCK_MONOTONIC, &start);
			query_hardware_time(ctx);
			clock_gettime(CLOCK_MONOTONIC, &end);

			long long unsigned diff = (long long unsigned)(BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec);
	/*		if(diff < min)
				min = diff;
			if (diff >max)
				max = diff;
			avg += diff;
	*/
			printf("%llu\n", diff);
		}

	//	printf("max: %llu ns, min: %lluns, avg: %llu ns \n", max, min, avg/i);
	}
}
