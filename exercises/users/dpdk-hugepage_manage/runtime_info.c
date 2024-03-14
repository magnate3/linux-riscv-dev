#define _GNU_SOURCE
#include "runtime_info.h"
#include "sysfs_ops.h"
#include "hugepage_memory.h"
#include <linux/limits.h>
#include <stdio.h>

int socket_id = -1;
int core_id = -1;
int cpu_id = -1;

int get_cpu_id()
{
	int i,cpu_nb;
	cpu_set_t cpuset;

	//get cpu set
	if(sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == -1){
		perror("Can not get cpu affinity information:");
		return -1;
	}
	
	//get logical cpu number
	cpu_nb = sysconf(_SC_NPROCESSORS_CONF);
	
	for(i=0; i<cpu_nb; i++){
		if(CPU_ISSET(i, &cpuset)){
			return i;
		}
	}
	return -1;
}

int get_core_id(int cpu_id)
{	
	char path[PATH_MAX];
	unsigned long val;
	
	int len = sprintf(path,"/sys/devices/system/cpu/cpu%d/topology/core_id",cpu_id);	
	if(len <=0 || len>=sizeof(path))
	{
		printf("Read core_id file error in copy path...\n");
		return -1;
	}
	
	if( parse_sysfs_value(path, &val) )
	{
		printf("Read core_id error in file reading...\n");		
	}
	
	return (int)val;
}

int get_socket_id(int cpu_id)
{
	char path[PATH_MAX];
	int socket_id;
	
	for(socket_id = 0; socket_id < MAX_SOCKET_NB; socket_id ++){
		sprintf(path, "/sys/devices/system/node/node%d/cpu%d", socket_id, cpu_id);
		if(access(path, F_OK) == 0){
			return socket_id;
		}
	}
	return -1;
}

int get_runtime_info()
{
	int ret;
	
	ret = get_cpu_id();
	if(ret >= 0)
		cpu_id = ret;
	else
		return -1;
	
	ret = get_core_id(cpu_id);
	if(ret >= 0)
		core_id = ret;
	else
		return -1;
	
	ret = get_socket_id(cpu_id);
	if(ret >= 0)
		socket_id = ret;
	else
		return -1;	
	
	return 0;
}

int get_cur_cpu_id()
{
	return cpu_id;
}

int get_cur_core_id()
{
	return core_id;
}

int get_cur_socket_id()
{
	return socket_id;
}
