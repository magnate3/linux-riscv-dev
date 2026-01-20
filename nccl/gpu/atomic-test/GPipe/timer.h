#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GpuTimer
{
	long long int start;
	long long int stop;

    double get_time()
    {
    	struct timespec time;
    	clock_gettime(CLOCK_REALTIME, &time);
    	return (double)time.tv_sec + (double)time.tv_nsec * 1.0e-9 ;
    }

    __host__ GpuTimer()
    {
    	start = 0;
    	stop = 0;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
    }

    __host__ ~GpuTimer()
    {
        // cudaEventDestroy(start);
        // cudaEventDestroy(stop);
    }

    __device__ void Start()
    {
        // cudaEventRecord(start, 0);
    	start = clock64();
    }

    __device__ void Stop()
    {
        // cudaEventRecord(stop, 0);
    	stop = clock64();
    }

    __host__ long long int Elapsed()
    {
        long long int elapsed = stop - start;
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

#endif  /* GPU_TIMER_H__ */
