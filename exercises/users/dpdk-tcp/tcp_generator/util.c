#include "util.h"

int distribution;
char output_file[MAXSTRLEN];

// Sample the value using Exponential Distribution
double sample(double lambda) {
   	double u = ((double) rte_rand()) / ((uint64_t) -1);

	return -log(1 - u) / lambda;
}

// Convert string type into int type
static uint32_t process_int_arg(const char *arg) {
	char *end = NULL;

	return strtoul(arg, &end, 10);
}

// Allocate all nodes for incoming packets (+ 20%)
void allocate_incoming_nodes() {
	uint64_t rate_per_queue = rate/nr_queues;
	uint64_t nr_elements_per_queue = (rate_per_queue * duration * nr_executions) * 1.2;

	incoming_array = (node_t**) malloc(nr_queues * sizeof(node_t*));
	if(incoming_array == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot alloc the incoming array.\n");
	}

        uint64_t i = 0;
	for(i = 0; i < nr_queues; i++) {
		incoming_array[i] = (node_t*) malloc(nr_elements_per_queue * sizeof(node_t));
		if(incoming_array[i] == NULL) {
			rte_exit(EXIT_FAILURE, "Cannot alloc the incoming array.\n");
		}
	}

	incoming_idx_array = (uint64_t*) malloc(nr_queues * sizeof(uint64_t));
	if(incoming_idx_array == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot alloc the incoming_idx array.\n");
	}

	for(i = 0; i < nr_queues; i++) {
		incoming_idx_array[i] = 0;
	}
} 

// Allocate and create an array for all interarrival packets for rate specified.
void create_interarrival_array() {
	uint64_t rate_per_queue = rate/nr_queues;
	double lambda;
	if(distribution == UNIFORM_VALUE) {
		lambda = (1.0/(double)rate_per_queue) * (double)BILLION;
	} else if(distribution == EXPONENTIAL_VALUE) {
		lambda = 1.0/((double)BILLION/(double)rate_per_queue);
	} else {
		rte_exit(EXIT_FAILURE, "Cannot define the interarrival distribution.\n");
	}
	
	uint64_t nr_elements_per_queue = 2 * rate_per_queue * duration;

	interarrival_array = (uint64_t**) malloc(nr_queues * sizeof(uint64_t*));
	if(interarrival_array == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot alloc the interarrival_gap array.\n");
	}

        uint64_t i = 0;
	for(i = 0; i < nr_queues; i++) {
		interarrival_array[i] = (uint64_t*) malloc(nr_elements_per_queue * sizeof(uint64_t));
		if(interarrival_array[i] == NULL) {
			rte_exit(EXIT_FAILURE, "Cannot alloc the interarrival_gap array.\n");
		}
		
		uint64_t *interarrival_gap = interarrival_array[i];
		if(distribution == UNIFORM_VALUE) {
                        uint64_t j;
			for(j = 0; j < nr_elements_per_queue; j++) {
				interarrival_gap[j] = lambda;
			}
		} else {
                        uint64_t j;
			for(j = 0; j < nr_elements_per_queue; j++) {
				interarrival_gap[j] = sample(lambda);
			}
		}
	} 
}

// Allocate and create an array for all flow indentier to send to the server
void create_flow_indexes_array() {
	uint32_t nbits = (uint32_t) log2(nr_queues);
	uint64_t rate_per_queue = rate/nr_queues;
	uint64_t nr_elements_per_queue = rate_per_queue * duration * nr_executions;

	flow_indexes_array = (uint16_t**) malloc(nr_queues * sizeof(uint16_t*));
	if(flow_indexes_array == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot alloc the flow_indexes array.\n");
	}

        uint64_t i = 0;
	for(i = 0; i < nr_queues; i++) {
		flow_indexes_array[i] = (uint16_t*) malloc(nr_elements_per_queue * sizeof(uint16_t));
		if(flow_indexes_array[i] == NULL) {
			rte_exit(EXIT_FAILURE, "Cannot alloc the flow_indexes array.\n");
		}
		uint16_t *flow_indexes = flow_indexes_array[i];
                int j = 0;
		for(j = 0; j < nr_elements_per_queue; j++) {
			flow_indexes[j] = ((rte_rand() << nbits) | i) % nr_flows;
		}
	}
}

// Clean up all allocate structures
void clean_heap() {
	free(incoming_array);
	free(incoming_idx_array);
	free(flow_indexes_array);
	free(interarrival_array);
	free(throughputs);
}

// Usage message
static void usage(const char *prgname) {
	printf("%s [EAL options] -- \n"
		"  -d DISTRIBUTION: <uniform|exponential>\n"
		"  -r RATE: rate in pps\n"
		"  -f FLOWS: number of flows\n"
		"  -q QUEUES: number of queues\n"
		"  -s SIZE: frame size in bytes\n"
		"  -t TIME: time in seconds to send packets\n"
		"  -c FILENAME: name of the configuration file\n"
		"  -o FILENAME: name of the output file\n",
		prgname
	);
}

// Parse the argument given in the command line of the application
int app_parse_args(int argc, char **argv) {
	int opt, ret;
	char **argvopt;
	char *prgname = argv[0];

	nr_executions = 2;
	argvopt = argv;
	while ((opt = getopt(argc, argvopt, "d:r:f:s:q:p:t:c:o:")) != EOF) {
		switch (opt) {
		// distribution
		case 'd':
			if(strcmp(optarg, "uniform") == 0) {
				// Uniform distribution
				distribution = UNIFORM_VALUE;
			} else if(strcmp(optarg, "exponential") == 0) {
				// Exponential distribution
				distribution = EXPONENTIAL_VALUE;
			} else {
				usage(prgname);
				rte_exit(EXIT_FAILURE, "Invalid arguments.\n");
			}
			break;
			
		// rate (pps)
		case 'r':
			rate = process_int_arg(optarg);
			break;

		// flows
		case 'f':
			nr_flows = process_int_arg(optarg);
			break;

		// frame size (bytes)
		case 's':
			frame_size = process_int_arg(optarg);
			tcp_payload_size = (frame_size - sizeof(struct rte_ether_hdr) - sizeof(struct rte_ipv4_hdr) - sizeof(struct rte_tcp_hdr));
			break;

		// queues
		case 'q':
			nr_queues = process_int_arg(optarg);
			min_lcores = 3 * nr_queues + 1;
			break;

		// duration (s)
		case 't':
			duration = process_int_arg(optarg);
			break;

		// config file name
		case 'c':
			process_config_file(optarg);
			break;
		
		// output mode
		case 'o':
			strcpy(output_file, optarg);
			break;

		default:
			usage(prgname);
			rte_exit(EXIT_FAILURE, "Invalid arguments.\n");
		}
	}

	if(optind >= 0) {
		argv[optind-1] = prgname;
	}

	if(nr_flows < nr_queues) {
		rte_exit(EXIT_FAILURE, "The number of flows should be bigger than the number of queues.\n");
	}

	ret = optind-1;
	optind = 1;

	return ret;
}

// Wait for the duration parameter
void wait_timeout() {
	uint64_t t0 = get_time_ns();
	while(get_time_ns() - t0 < duration * nr_executions * BILLION) { }
	/* set quit flag for all internal cores */
	quit_rx = 1;
	quit_tx = 1;
	quit_rx_ring = 1;
	// uint64_t t0 = rte_rdtsc();
	// while((rte_rdtsc() - t0) < (duration * nr_executions * 1000000 * TICKS_PER_US)) { }

	// /* wait for remaining */
	// t0 = rte_rdtsc_precise();
	// while((rte_rdtsc() - t0) < (5 * 1000000 * TICKS_PER_US)) { }
	
	// /* set quit flag for all internal cores */
	// quit_rx = 1;
	// quit_tx = 1;
	// quit_rx_ring = 1;
}

// Compare two double values (for qsort function)
int cmp_func(const void * a, const void * b) {
	double da = (*(double*)a);
	double db = (*(double*)b);

	return (da - db) > ( (fabs(da) < fabs(db) ? fabs(db) : fabs(da)) * EPSILON);
}

// Print stats into output file
void print_stats_output() {
	// open the file
	FILE *fp = fopen(output_file, "w");
	if(fp == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot open the output file.\n");
	}

        uint32_t i = 0;
	for(i = 0; i < nr_queues; i++) {
		// get the pointers
		node_t *incoming = incoming_array[i];
		uint32_t incoming_idx = incoming_idx_array[i];

		// drop the first 50% packets for warming up
		uint64_t j = 0.5 * incoming_idx;

		// print the RTT latency in (ns)
		node_t *cur;
		for(; j < incoming_idx; j++) {
			cur = &incoming[j];
			if(cur->timestamp_rx < cur->timestamp_tx){
				continue;
			}
			fprintf(fp, "%lu\t%lu\n",
				cur->flow_id,
				(cur->timestamp_rx - cur->timestamp_tx)
				// ((uint64_t)((cur->timestamp_rx - cur->timestamp_tx)/((double)TICKS_PER_US/1000)))
			);
		}
	}

	// close the file
	fclose(fp);


	strcat(output_file, ".tput");
	FILE *fp_tput = fopen(output_file, "w");
	if(fp_tput == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot open the throughput tracking log file.\n");
	}
	/* drop the first 50% for warming up */
	i = duration * (1000 / THROUGHPUT_INTERVAL);
	for(; i < duration * (1000 / THROUGHPUT_INTERVAL) * nr_executions; i++) {
		fprintf(fp_tput, "%u %lu\n", i * THROUGHPUT_INTERVAL, throughputs[i]);
	}
	/* close the file */
	fclose(fp_tput);
}

// Process the config file
void process_config_file(char *cfg_file) {
	// open the file
	struct rte_cfgfile *file = rte_cfgfile_load(cfg_file, 0);
	if(file == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot load configuration profile %s\n", cfg_file);
	}

	// load ethernet addresses
	char *entry = (char*) rte_cfgfile_get_entry(file, "ethernet", "src");
	if(entry) {
		rte_ether_unformat_addr((const char*) entry, &src_eth_addr);
	}
	entry = (char*) rte_cfgfile_get_entry(file, "ethernet", "dst");
	if(entry) {
		rte_ether_unformat_addr((const char*) entry, &dst_eth_addr);
	}

	// load ipv4 addresses
	entry = (char*) rte_cfgfile_get_entry(file, "ipv4", "src");
	if(entry) {
		uint8_t b3, b2, b1, b0;
		sscanf(entry, "%hhd.%hhd.%hhd.%hhd", &b3, &b2, &b1, &b0);
		src_ipv4_addr = IPV4_ADDR(b3, b2, b1, b0);
	}
	entry = (char*) rte_cfgfile_get_entry(file, "ipv4", "dst");
	if(entry) {
		uint8_t b3, b2, b1, b0;
		sscanf(entry, "%hhd.%hhd.%hhd.%hhd", &b3, &b2, &b1, &b0);
		dst_ipv4_addr = IPV4_ADDR(b3, b2, b1, b0);
	}

	// load TCP destination port
	entry = (char*) rte_cfgfile_get_entry(file, "tcp", "dst");
	if(entry) {
		uint16_t port;
		sscanf(entry, "%hu", &port);
		dst_tcp_port = port;
	}

	// local server info
	entry = (char*) rte_cfgfile_get_entry(file, "server", "nr_servers");
	if(entry) {
		uint16_t n;
		sscanf(entry, "%hu", &n);
		nr_servers = n;
	}

	// close the file
	rte_cfgfile_close(file);
}

// Fill the data into packet payload properly
inline void fill_payload_pkt(struct rte_mbuf *pkt, uint32_t idx, uint64_t value) {
	uint8_t *payload = (uint8_t*) rte_pktmbuf_mtod_offset(pkt, uint8_t*, sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr));

	((uint64_t*) payload)[idx] = value;
}

// return current time in ns
uint64_t get_time_ns()
{
	struct timespec ts;
	clock_gettime( CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_nsec + ((uint64_t)ts.tv_sec * BILLION);
}


// Prepare throughput tracking array
void prepare_throughput_tracking() {
    uint64_t nr_ms = duration * (1000 / THROUGHPUT_INTERVAL) * nr_executions * 1.2;

    throughputs = (uint64_t*) malloc(nr_ms * sizeof(uint64_t));
    if(throughputs == NULL) {
        rte_exit(EXIT_FAILURE, "Cannot alloc the throughputs array.\n");
    }

    uint64_t i = 0;
    for(i = 0; i < nr_ms; i++) {
		throughputs[i] = 0;
		// printf("%lu\n", interarrival_gap[i]);
    } 
	// printf("TICKS_PER_US: %lu", TICKS_PER_US);
	// exit(1);
}
