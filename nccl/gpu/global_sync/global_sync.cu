#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <sm_11_atomic_functions.h>


#define MAX_NUM_BLOCKS	40

// Synchronization code is based on paper: 
// Shucai Xiao and Wu-chun Feng. "Inter-Block GPU Communication via Fast Barrier Synchronization". 
// Proceedings of the 24th IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2010
// 
__device__ volatile int g_mutex;
__device__ volatile int ArrayIn[MAX_NUM_BLOCKS];
__device__ volatile int ArrayOut[MAX_NUM_BLOCKS];

__global__ void init_sync(){
	g_mutex = 0;
	
	for(int i = threadIdx.x; i < MAX_NUM_BLOCKS; i+=blockDim.x){
		ArrayIn[i] = 0;
		ArrayOut[i] = 0;	
	}

};

__device__ void __gpu_sync(int goalVal){
	// Thread id in block
	int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;

	// only thread 0 is used for synchronization
	if(tid_in_block == 0){
		atomicAdd((int*)&g_mutex, 1);
		
		while(g_mutex != goalVal){
			// Do nothing. Volatile g_mutex guarantees 
			// that this loop is not moved away
		}
	}
	__syncthreads();
}

__device__ void __gpu_sync_lock_free(int goalVal){
	// Thread id in block
	int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
	int nBlockNum = gridDim.x * gridDim.y;
	int bid = blockIdx.x * gridDim.y + blockIdx.y;

	// only thread 0 is used for synchronization
	if(tid_in_block == 0){
		ArrayIn[bid] = goalVal;
	}

	if(bid == 1){
		// Assuming that there are more threads than blocks. Modify it.
		if(tid_in_block < nBlockNum){
			while(ArrayIn[tid_in_block] != goalVal){
				// Do nothing here
			}
		}
		__syncthreads();
	
		if(tid_in_block < nBlockNum){
			ArrayOut[tid_in_block] = goalVal;
		}
	}

	if(tid_in_block == 0){
		while(ArrayOut[tid_in_block] != goalVal){
			// Do nothing here
		}
	}
	__syncthreads();
}
