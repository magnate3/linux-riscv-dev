
/* 1) Includes */
#include <stdio.h>
#include <unistd.h>
#include <net/if.h> // for if_nametoindex
#include <time.h>
#include "bpf.h"
#include "libbpf.h"

#define DEFAULT_IFACE "enx00e04c3662aa"
#define MAP_XDP_PROGS_NAME "packet_processing_progs"
/* 3) Implementation */
int main(int argc, char *argv[])
{
	const char *ifname = DEFAULT_IFACE;
	const char *xdp_program_name = NULL;

	unsigned int ifindex = 0;
	struct bpf_program *prog = NULL;
	struct bpf_object *obj = NULL;
	int fd = -1, jmp_tbl_fd = -1, main_fd = -1;
	uint32_t key = 0, next_key=0;
	uint64_t value = 0;
	struct timespec ts;

	/* 1) Get ETH interface index */
	if(!(ifindex = if_nametoindex(ifname)))
	{
		printf("Error: finding device %s\n failed", ifname);
		return -1;
	}


	/* 2) Verify XDP object file existence */
	if(!(obj = bpf_object__open_file("xdp_kern.o", NULL)) || libbpf_get_error(obj))
	{
		printf("Error: opening BPF object file failed\n");
		return -1;
	}
	/* 4) Load XDP object file */
	if(bpf_object__load(obj))
	{
		printf("Error: loading BPF obj file failed\n");
		return -1;
	}

	/* 5) Find XDP progs map file descriptor */		
	if((jmp_tbl_fd = bpf_object__find_map_fd_by_name(obj, "packet_processing_progs")) < 0)
	{
		printf("Error: table " MAP_XDP_PROGS_NAME " not found\n");
		return -1;
	}

	/* 6) Fill XDP Programs Map by Iterating XDP Sections*/
	bpf_object__for_each_program(prog, obj)
	{
		xdp_program_name = bpf_program__section_name(prog);
		fd = bpf_program__fd(prog);
		if(!strcmp(xdp_program_name, "xdp_classifier"))
		{
			main_fd = fd;
		}
		printf(MAP_XDP_PROGS_NAME " entry key -> name -> fd\n: %d -> %s -> %d\n", key, xdp_program_name, fd);
		if(bpf_map_update_elem(jmp_tbl_fd, &key, &fd, BPF_ANY)<0)
		{
			printf("Error: making entry for %s\n", xdp_program_name);
			fd = -1;
			return -1;
		}
		++key;
	}

	/* 6.4) Verify main program found */
	if(fd < 0 || main_fd < 0)
	{
		printf("Error: didn't find main program\n" );
		return -1;
	}
	
	/* 7) Link Main XDP Prog to ETH Interface*/
	if(bpf_set_link_xdp_fd(ifindex,main_fd,0))
	{
		printf("Error: attaching xdp program to device\n");
		return -1;
	}
	/* 8) Loading Process Succeeded */	
	printf("Program attached and running.\nPress Ctrl-C to stop followed by make unload\n");
	while(true){
		sleep(60);
	}
	return -1;
}
