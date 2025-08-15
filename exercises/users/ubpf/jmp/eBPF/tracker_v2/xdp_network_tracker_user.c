
/* 1) Includes */
#include <stdio.h>
#include <unistd.h>
#include <net/if.h> // for if_nametoindex
#include <time.h>
#include "bpf.h"
#include "libbpf.h"

/* 2) Defines */
#define BLOCK_TIME 5000000000 
#define DEFAULT_IFACE "ens33"
#define QUERY_MAP_PIN_PATH "/sys/fs/bpf/query"
#define MAP_ILLEGAL_DOMAINS_PIN_PATH "/sys/fs/bpf/map_illegal_domains"
#define MAP_BLOCKED_REQUESTERS_PIN_PATH "/sys/fs/bpf/map_blocked_requesters"
#define MAP_PACKETS_COUNTERS_PIN_PATH "/sys/fs/bpf/map_packets_counters"
#define QUERY_TBL "query"
#define MAP_ILLEGAL_DOMAINS_NAME "map_illegal_domains"
#define MAP_BLOCKED_REQUESTERS_NAME "map_blocked_requesters"
#define MAP_PACKETS_COUNTERS_NAME "map_packets_counters"
#define MAP_XDP_PROGS_NAME "map_xdp_progs"


/* 3) Implementation */
int main(int argc, char *argv[])
{
	const char *ifname = DEFAULT_IFACE;
	const char *xdp_program_name = NULL;

	unsigned int ifindex = 0;
	struct bpf_program *prog = NULL;
	struct bpf_object *obj = NULL;
	struct bpf_map *query = NULL;
	struct bpf_map *map_illegal_domains = NULL;
	struct bpf_map *map_blocked_requesters = NULL;
	struct bpf_map *map_packets_counters = NULL;
	int fd = -1, jmp_tbl_fd = -1, main_fd = -1,blocked_requesters_fd = -1, deleted_map_entries = 0;
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
	if(!(obj = bpf_object__open_file("xdp_network_tracker.o", NULL)) || libbpf_get_error(obj))
	{
		printf("Error: opening BPF object file failed\n");
		return -1;
	}

	/* 3) Pin Maps*/
	/* 3.1) Query Map */
	if(!(query = bpf_object__find_map_by_name(obj, QUERY_TBL)))
	{
		printf("Error: table " QUERY_TBL " not found\n");
		return -1;
	}

	/* 3.1.1) Pin Map */
	if(bpf_map__set_pin_path(query, QUERY_MAP_PIN_PATH))
	{
		printf("Error: pinning " QUERY_TBL " to \"%s\" failed\n", QUERY_MAP_PIN_PATH);
		return -1;
	}
	
	/* 3.2) illegal Domains Map */
	if(!(map_illegal_domains = bpf_object__find_map_by_name(obj, MAP_ILLEGAL_DOMAINS_NAME)))
	{
		printf("Error: table " MAP_ILLEGAL_DOMAINS_NAME " not found\n");
		return -1;
	}

	/* 3.2.1) Pin Map */
	if(bpf_map__set_pin_path(map_illegal_domains, MAP_ILLEGAL_DOMAINS_PIN_PATH)){
		printf("Error: pinning " MAP_ILLEGAL_DOMAINS_NAME " to \"%s\" failed\n", MAP_ILLEGAL_DOMAINS_PIN_PATH);
		return -1;
	}

	/* 3.3) MAP Blocked requesters */
	if(!(map_blocked_requesters = bpf_object__find_map_by_name(obj, MAP_BLOCKED_REQUESTERS_NAME)))
	{
		printf("Error: table " MAP_BLOCKED_REQUESTERS_NAME " not found\n");
		return -1;
	}

	/* 3.3.1) Pin Map */
	if(bpf_map__set_pin_path(map_blocked_requesters, MAP_BLOCKED_REQUESTERS_PIN_PATH))
	{
		printf("Error: pinning " MAP_BLOCKED_REQUESTERS_NAME " to \"%s\" failed\n", MAP_BLOCKED_REQUESTERS_PIN_PATH);
		return -1;
	}

	/* 3.4) Packets Counters Map*/
	if(!(map_packets_counters = bpf_object__find_map_by_name(obj, MAP_PACKETS_COUNTERS_NAME)))
	{
		printf("Error: table " MAP_PACKETS_COUNTERS_NAME " not found\n");
		return -1;
	}

	/* 3.4.1) Pin Map */
	if(bpf_map__set_pin_path(map_packets_counters, MAP_PACKETS_COUNTERS_PIN_PATH))
	{
		printf("Error: pinning " MAP_PACKETS_COUNTERS_NAME " to \"%s\" failed\n", MAP_PACKETS_COUNTERS_PIN_PATH);
		return -1;
	}

	/* 4) Load XDP object file */
	if(bpf_object__load(obj))
	{
		printf("Error: loading BPF obj file failed\n");
		return -1;
	}

	/* 5) Find XDP progs map file descriptor */		
	if((jmp_tbl_fd = bpf_object__find_map_fd_by_name(obj, MAP_XDP_PROGS_NAME)) < 0)
	{
		printf("Error: table " MAP_XDP_PROGS_NAME " not found\n");
		return -1;
	}

	/* 6) Fill XDP Programs Map by Iterating XDP Sections*/
	bpf_object__for_each_program(prog, obj)
	{
		xdp_program_name = bpf_program__section_name(prog);
		fd = bpf_program__fd(prog);
		if(!strcmp(xdp_program_name, "xdp-packet-preprocess"))
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
	if((blocked_requesters_fd = bpf_object__find_map_fd_by_name(obj, MAP_BLOCKED_REQUESTERS_NAME)) < 0){
		printf("Error: table " MAP_BLOCKED_REQUESTERS_NAME " not found\n");
		return -1;
	}

	/* 8) Loading Process Succeeded */	
	printf("Program attached and running.\nPress Ctrl-C to stop followed by make unload\n");
	while(true){
		sleep(60);
		deleted_map_entries = 0;
		key = -1;
		printf("Checking blocked_requesters_fd\n");
		while(bpf_map_get_next_key(blocked_requesters_fd, &key, &next_key) == 0){
			//printf("Got key %d next: %d \n", key, next_key);
			int res = bpf_map_lookup_elem(blocked_requesters_fd, &key, &value);
			if(res < 0){//first iteration key will be -1
				key = next_key;
				continue;
			}
			clock_gettime(CLOCK_MONOTONIC, &ts);
			if(ts.tv_nsec - value > BLOCK_TIME){
				deleted_map_entries++;
				uint32_t to_delete = key;
				key = next_key;
				bpf_map_delete_elem(blocked_requesters_fd, &to_delete);
			}
		}
		//delete last element considered a special case, since upon last key
		//bpf_map_get_next_key returns -1
		int res = bpf_map_lookup_elem(blocked_requesters_fd, &key, &value);
		if(res < 0){
			key = next_key;
			continue;
		}
		clock_gettime(CLOCK_MONOTONIC, &ts);
		if(ts.tv_nsec - value > BLOCK_TIME){
			deleted_map_entries++;
			uint32_t to_delete = key;
			key = next_key;
			bpf_map_delete_elem(blocked_requesters_fd, &to_delete);
		}
		printf("Deleted: %d entries from map_blocked_requesters\n", deleted_map_entries);
	}
	return -1;
}
