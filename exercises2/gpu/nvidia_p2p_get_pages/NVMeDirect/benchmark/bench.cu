/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#ifdef GPUD
extern "C"
{
#include <nvmed/nvmed.h>
}
#include <errno.h>
#endif
#ifdef PERFSTAT
extern "C"
{
#include <donard/perfstats.h>
}
#endif
/*#ifdef NONBLOCKING
#define NUM_OF_THREADS 4
#include <pthread.h>
void *load_file(void *x);
struct thread_para
{
	char *filename;
	unsigned long offset;
	unsigned long length;
	float *dest;
};
#endif
*/
void Bench(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	Bench( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file> <num_of_elements>\n", argv[0]);

}
#ifdef GPUD
#define NVMED_INIT(a)	nvmed_init(a);
#define NVMED_SEND(a,b,c,d) nvmed_send(a,b,c,d)
#define NVMED_RECV(a,b,c,d) nvmed_recv(a,b,c,d)
#endif

#ifdef PIPELINE
#define CHUNK 4096*1024*1024
#endif
void Bench( int argc, char** argv) 
{

    struct timeval time_start, time_end, total_start, total_end;
    size_t filesize;
	int input_fd;
    char *input_f;

    gettimeofday(&total_start, NULL);
    gettimeofday(&time_start, NULL);

    FILE *fp;
    unsigned long num_of_nodes;
    int input_time,input_time_wo_init = 0;
    float *h_elements = NULL;
    float *ch_elements = NULL;
    int i;
	if(argc < 2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];
	input_fd = open(input_f,O_RDONLY);
	num_of_nodes = atol(argv[2]);

   	gettimeofday(&time_start, NULL);
#ifdef GPUD
	NVMED_INIT(1);
  	gettimeofday(&time_end, NULL);
	input_time_wo_init = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	fprintf(stderr,"HGProfile: NVMED_INIT %d\n",input_time_wo_init);

	float* d_elements = NULL;
	filesize = sizeof(float)*num_of_nodes;
	cudaMalloc( (void**) &d_elements, filesize);
	filesize = NVMED_SEND(input_fd,(void**) &d_elements,0,0);
//	h_elements = (float*) malloc(sizeof(float)*num_of_nodes);
//	filesize = nvmed_send_direct(input_fd,(void**) &h_elements,0,0);
	//fprintf(stderr,"After NVMeDSend\n");
	gettimeofday(&time_end, NULL);
//	input_time_wo_init = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
//	fprintf(stderr,"HGProfile: nvmed_send %d\n",input_time_wo_init);
/*	float *d_elements_2;
	d_elements_2 = d_elements;
	d_elements_2 += num_of_nodes/2;
	NVMED_SEND(input_f,(void**) &d_elements_2,0,sizeof(float)*(num_of_nodes/2));
*/
//    	gettimeofday(&time_end, NULL);
	input_time_wo_init = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)  - input_time_wo_init);
	fprintf(stderr,"HGProfile: nvmed_send %d\n",input_time_wo_init);
	fprintf(stderr,"filesize: %llu\n",filesize);
	
#else
	filesize = sizeof(float)*num_of_nodes;
	float* d_elements = NULL;

	cudaMalloc( (void**) &d_elements, sizeof(float)*num_of_nodes);
	h_elements = (float*) malloc(sizeof(float)*num_of_nodes);
    	gettimeofday(&time_end, NULL);
	input_time_wo_init = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	fprintf(stderr,"HGProfile: cudaMalloc %d\n",input_time_wo_init);
	fp = fopen(input_f,"r");
	if(!fp)
	{
		fprintf(stderr,"Error Reading graph file\n");
		return;
	}
//	cudaMallocHost((void **)&h_elements, sizeof(float)*num_of_nodes);
//fprintf(stderr,"host: %p, gpu: %p\n",h_elements, d_elements);
	fread(h_elements,sizeof(float),num_of_nodes,fp);
	fclose(fp);
	gettimeofday(&time_end, NULL);
	input_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	fprintf(stderr,"HGProfile: FileInput %d Bandwidth %lf (%lf wo init) MB/Sec\n",input_time, (double)filesize/(input_time*1.024), (double)filesize/((input_time-input_time_wo_init)*1.024));
	cudaMemcpy( d_elements, h_elements, sizeof(float)*num_of_nodes, cudaMemcpyHostToDevice);
#endif
	gettimeofday(&time_end, NULL);
	input_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
//	fprintf(stderr,"filesize: %llu\n",filesize);
	fprintf(stderr,"HGProfile: File to GPU %d Bandwidth %lf (%lf wo init) MB/Sec\n",input_time, (double)filesize/(input_time*1.024), (double)filesize/((input_time-input_time_wo_init)*1.024));
    	gettimeofday(&time_start, NULL);	
#ifdef OUTPUT
#ifdef GPUD
	h_elements = (float *)malloc(filesize);
//	NVMED_RECV("/mnt/princeton/h1tseng/bench_output.nvmed.txt", d_elements, sizeof(float)*num_of_nodes, 0);

//	cudaMemcpy( h_elements, d_elements, sizeof(float)*num_of_nodes, cudaMemcpyDeviceToHost);
	int fd = open("/mnt/intel/htseng3/bench_output.nvmed.txt", O_WRONLY | O_TRUNC | O_CREAT, 0777 ^ (umask(0)));
	NVMED_RECV(fd, d_elements, sizeof(float)*num_of_nodes, 0);

//	FILE *fpo = fdopen(fd,"wb");
	//fwrite(h_elements, sizeof(float), num_of_nodes, fpo);
//	fclose(fpo);
    fsync(fd);
	close(fd);

    	gettimeofday(&time_end, NULL);
	input_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	fprintf(stderr,"HGProfile: GPU to File %d Bandwidth %lf (%lf wo init) MB/Sec\n",input_time, (double)filesize/(input_time*1.024), (double)filesize/((input_time)*1.024));
//	fprintf(stderr,"HGProfile: GPU to File %d\n",((total_end.tv_sec * 1000000 + total_end.tv_usec) - (total_start.tv_sec * 1000000 + total_start.tv_usec)));
#else

//	for(i=0;i<50;i++)
//	        fprintf(stderr,"%f\t",h_elements[num_of_nodes-i-1]);
//	fprintf(stderr,"\n");	

        if(h_elements == NULL)
	        h_elements = (float *)malloc(sizeof(float)*num_of_nodes);
	cudaMemcpy( h_elements, d_elements, sizeof(float)*num_of_nodes, cudaMemcpyDeviceToHost);
    int fd = open("/mnt/intel/htseng3/bench_output.baseline.txt", O_CREAT|O_RDWR|O_SYNC, 0777 ^ (umask(0)));
	FILE *fpo = fdopen(fd,"wb");
	fwrite(h_elements, sizeof(float), num_of_nodes, fpo);
	fclose(fpo);
    fsync(fd);
	close(fd);
	gettimeofday(&time_end, NULL);
	input_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	fprintf(stderr,"HGProfile: GPU to File %d Bandwidth %lf (%lf wo init) MB/Sec\n",input_time, (double)filesize/(input_time*1.024), (double)filesize/((input_time)*1.024));

#endif
#endif
   	gettimeofday(&total_end, NULL);
	fprintf(stderr,"HGProfile: Total %ld\n",((total_end.tv_sec * 1000000 + total_end.tv_usec) - (total_start.tv_sec * 1000000 + total_start.tv_usec)));
#ifdef VERIFY
	fp = fopen(input_f,"r");
	if(!fp)
	{
		fprintf(stderr,"Error Reading graph file\n");
		return;
	}
	ch_elements = (float*) malloc(sizeof(float)*num_of_nodes);
	fread(ch_elements,sizeof(float),num_of_nodes,fp);
	fclose(fp);
//	fprintf(stderr,"0: %f %f\n", h_elements[0],ch_elements[0]);
	
	for(i=0;i<num_of_nodes;i++)
	{
		if(ch_elements[i] != h_elements[i])
		{
	        fprintf(stderr,"%d %f %f\t",i, h_elements[i],ch_elements[i]);
	        break;
		}
	}
	fprintf(stderr,"\n");
#endif
#ifdef GPUD
	nvmed_deinit();
#endif
	return;
}
