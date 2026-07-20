#define _GNU_SOURCE
#include <sys/sysinfo.h>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <linux/kernel.h>
#include <linux/module.h>

#ifndef _LIB_OF_MEMORY_
#define _LIB_OF_MEMORY_

typedef struct {
    unsigned long size,resident,share,text,lib,data,dt;
} ProcStatm;

//int sysinfo(struct sysinfo *info);

//struct sysinfo {
//               long uptime;             /* Seconds since boot */
//               unsigned long loads[3];  /* 1, 5, and 15 minute load averages */
//               unsigned long totalram;  /* Total usable main memory size */
//               unsigned long freeram;   /* Available memory size */
//               unsigned long sharedram; /* Amount of shared memory */
//               unsigned long bufferram; /* Memory used by buffers */
//               unsigned long totalswap; /* Total swap space size */
//               unsigned long freeswap;  /* swap space still available */
//               unsigned short procs;    /* Number of current processes */
//               unsigned long totalhigh; /* Total high memory size */
//               unsigned long freehigh;  /* Available high memory size */
//               unsigned int mem_unit;   /* Memory unit size in bytes */
//               char _f[20-2*sizeof(long)-sizeof(int)]; /* Padding to 64 bytes */
//           };

void ProcStat_init(ProcStatm *result) {
    const char* statm_path = "/proc/self/statm";
    FILE *f = fopen(statm_path, "r");
    if(!f) {
        perror(statm_path);
        abort();
    }
    if(7 != fscanf(
        f,
        "%lu %lu %lu %lu %lu %lu %lu",
        &(result->size),
        &(result->resident),
        &(result->share),
        &(result->text),
        &(result->lib),
        &(result->data),
        &(result->dt)
    )) {
        perror(statm_path);
        abort();
    }
    fclose(f);
}

/**
 * int_sqrt - rough approximation to sqrt
 * @x: integer of which to calculate the sqrt
 *
 * A very rough approximation to the sqrt() function.
 */
unsigned long int_sqrt(unsigned long x)
{
	unsigned long op, res, one;

	op = x;
	res = 0;
    int BITS_PER_LONG = 64;

	one = 1UL << (BITS_PER_LONG - 2);
	while (one > op)
		one >>= 2;

	while (one != 0) {
		if (op >= res + one) {
			op = op - (res + one);
			res = res +  2 * one;
		}
		res /= 2;
		one /= 4;
	}
	return res;
}

void badness(double time, long page_size, 
                ProcStatm proc_statm){
    ProcStat_init(&proc_statm);
    double total_vm = ((double)proc_statm.size * page_size) / (1024 * 1024);
    printf("/proc/self/statm size resident %f MiB, page_size %ld\n",
                total_vm, page_size
            );
    double minutes = time / 60;
    if(time < 1)
	time = time * 1000;
    double _badness = total_vm / (int_sqrt((long)time) * //pow(minutes, 1.0/4)
                        sqrt(sqrt(minutes))
                        );
    printf("Badness bd = %f\n", _badness);
}
           
unsigned long mem_avail()
{
  struct sysinfo info;
  
  if (sysinfo(&info) < 0)
    return 0;
    
  return info.freeram;
}

#endif