/*
 * Copyright (c) 2005 Mellanox Technologies. All rights reserved.
 *
 */

#include "rdma_mt.h"


static void usage(const char *argv0)
{
	printf("Usage:\n");
	printf("  %s                      start a server and wait for connection\n", argv0);
	printf("  %s [Options] <host>     connect to server at <host>\n", argv0);
	printf("\n");
	printf("Options:\n");

	printf("  -h, --help");
	printf("    Show this help screen.\n");

	printf("  -a, --ip=<ip address>");
	printf("    IP address of server.\n");

	printf("  -d, --ib-dev=<dev>");
	printf("    Use IB device <dev> (default first device found)\n");

	printf("  -D, --direction=<direction>");
	printf("    Direction of RDMA transaction, 0: BI, 1: Client -> Server, 2: Server -> Client.\n");

	printf("  -e, --events");
	printf("    Sleep on CQ events (default poll)\n");
		
	printf("  -g, --sq_sig_all=<direction> \n");

	printf("  -G, --debug_level=<debug_level>\n");
	
	printf("  -i, --ib-port=<port>");
	printf("    Use port <port> of IB device (default %d)\n", 1);

	printf("  -I, --interval=<interval>");
	printf("    Sleeping time(us) between 2 RDMA transactions.\n");

	printf("  -n, --num_of_iter=<num_of_iter> \n");

	printf("  -o, --opcode=<opcode>");
	printf("    RDMA Operation Code, 0: RDMA_WRITE, 1: RDMA_WRITE_IMM, 2: SEND, 3: SEND_IMM, 4: RDMA_READ, 5: CAS, 6: FAA.\n");

	printf("  -p, --port=<port>");
	printf("    TCP port of server.\n");

	printf("  -s, --sl=<sl> \n");

	printf("  -t, --qp_type=<qp_type>");
	printf("    Type of QP, 2:RC, 3:UC, 4:UD. Currently only RC is supported.\n");

	printf("  -t, --num_of_thread=<num_of_thread>");
	printf("    The number of threads which will be created.\n");

	printf("Examples:\n");
	printf("    Server:.\n");
	printf("        ./rdma_mt_send_lat -d mlx4_0 -i 2 -G 4\n");
	printf("    Client:.\n");
	printf("        ./rdma_mt_send_lat -d mlx4_0 -i 2 -G 4 -o 2 -t 2  192.168.2.6 \n");

	putchar('\n');
}


static void init_user_param(struct user_param_t *user_param)
{
	user_param->server_ip      = NULL;
	user_param->hca_id         = "mlx4_0";
	user_param->ib_port        = 1;
	user_param->ip             = "127.0.0.1";
	user_param->tcp_port       = 18752;
	user_param->num_of_iter    = 1000;
	user_param->num_of_thread  = 1;
	user_param->num_of_oust    = 256;
	user_param->size_per_sg    = DEF_SG_SIZE;
	user_param->sq_sig_all     = 1;
	user_param->max_send_sge   = 1;
	user_param->sl             = DEF_SL;
	user_param->interval       = 0;
	user_param->direction      = 0;
	user_param->use_event      = 0;

	user_param->path_mtu       = IBV_MTU_256;
	user_param->qp_timeout     = 0x12;
	user_param->qp_retry_count = 6;
	user_param->qp_rnr_timer   = DEF_RNR_NAK_TIMER;
	user_param->qp_rnr_retry   = DEF_PKEY_IX;
	user_param->comp_timeout   = 6;
	user_param->rr_post_delay  = 6;

	user_param->cq_size        = 1024;
}


static struct option options[] =
{
	{ .name = "help",           .has_arg = 0, .val = 'h' },
	{ .name = "ip",             .has_arg = 1, .val = 'a' },
	{ .name = "ib-dev",         .has_arg = 1, .val = 'd' },
	{ .name = "direction",      .has_arg = 1, .val = 'D' },
	{ .name = "direction",      .has_arg = 0, .val = 'e' },
	{ .name = "sq_sig_all",     .has_arg = 1, .val = 'g' },
	{ .name = "debug_level",    .has_arg = 1, .val = 'G' },
	{ .name = "ib-port",        .has_arg = 1, .val = 'i' },
	{ .name = "interval",       .has_arg = 1, .val = 'I' },
	{ .name = "num_of_iter",    .has_arg = 1, .val = 'n' },
	{ .name = "opcode",         .has_arg = 1, .val = 'o' },
	{ .name = "num_of_outs",    .has_arg = 1, .val = 'O' },
	{ .name = "server_port",    .has_arg = 1, .val = 'p' },
	{ .name = "sl",             .has_arg = 1, .val = 's' },
	{ .name = "qp_type",        .has_arg = 1, .val = 't' },
	{ .name = "num_of_thread",  .has_arg = 1, .val = 'T' },	
    { 0 }
};


static int parser(struct user_param_t *user_param, char *argv[], int argc)
{
	int c;

	init_user_param(user_param);

	while (1) {
        c = getopt_long(argc,argv,"a:d:D:g:G:i:I:n:o:O:p:s:t:T:hev",options,NULL);
        if (c == -1) {
			break;
        }

        switch (c) {
			case 'h': usage(argv[0]);
					  return 1;
			case 'a': user_param->ip            = strdup(optarg);          break;
			case 'd': user_param->hca_id        = strdup(optarg);          break;
			case 'D': user_param->direction     = strtol(optarg, NULL, 0); break;
			case 'e': user_param->use_event     = 1;                       break;
			case 'g': user_param->sq_sig_all    = strtol(optarg, NULL, 0); break;
			case 'G': user_param->debug_level   = strtol(optarg, NULL, 0); break;
			case 'i': user_param->ib_port       = strtol(optarg, NULL, 0); break;
			case 'I': user_param->interval      = strtol(optarg, NULL, 0); break;
			case 'n': user_param->num_of_iter   = strtol(optarg, NULL, 0); break;
			case 'o': user_param->opcode        = strtol(optarg, NULL, 0); break;
			case 'O': user_param->num_of_oust   = strtol(optarg, NULL, 0); break;
			case 'p': user_param->tcp_port      = strtol(optarg, NULL, 0); break;
			case 's': user_param->sl            = strtol(optarg, NULL, 0); break;
			case 't': user_param->qp_type       = strtol(optarg, NULL, 0); break;
			case 'T': user_param->num_of_thread = strtol(optarg, NULL, 0); break;

			default:
				printf(" Invalid Command or flag.\n");
				usage(argv[0]);
				return 1;
		 }
	}

	Debug = user_param->debug_level;
    if (optind == argc - 1) {
		user_param->server_ip = strdup(argv[optind]);
    } else if (optind < (argc - 1)) {
        ERROR(" Invalid command line. Please check command rerun.\n");
        return 1;
    }

	if (user_param->server_ip && (user_param->opcode != IBV_WR_SEND)) {
		ERROR("Invalid opcode, only SEND is supported.\n");
		usage(argv[0]);
		return 1;
	}

	if (user_param->server_ip && (user_param->qp_type != IBV_QPT_RC) /*&& (user_param->qp_type != IBV_QPT_UD)*/) {
		ERROR(" Invalid QP type, only RC is supported.\n");
		usage(argv[0]);
		return 1;
	}

    return 0;
}


static struct rdma_resource_t rdma_resource;
uint32_t Debug = 0;
uint32_t volatile stop_all = 0;


static void do_signal(int dunno)
{
	switch (dunno) { 
	case 1: 
		printf("Get a signal -- SIGHUP ");
		stop_all = 1;
		break; 
	case 2:
	case 3:
	default:
		break;
	}

	return; 
}


static void signal_init()
{
	signal(SIGHUP, do_signal);
	return;
}


static void UNUSED dump_args(struct rdma_resource_t* rdma_resource)
{
	struct user_param_t *user_param = &(rdma_resource->user_param);

	DEBUG("Input parameters after sync:\n");
	DEBUG("\tnum_of_thread:%d\n", user_param->num_of_thread);
	DEBUG("\tqp_num       :%d\n", user_param->num_of_qp);
	DEBUG("\tcq_num       :%d\n", user_param->num_of_cq);
	DEBUG("\tsrq_num      :%d\n", user_param->num_of_srq);
	DEBUG("\tts_mask      :%d\n", user_param->ts_mask);
	DEBUG("\tnum_of_iter  :%d\n", user_param->num_of_iter);
	DEBUG("\topcode       :%d\n", user_param->opcode);
}


int main(int argc, char *argv[])
{
	int rc;
	int i = 0;
	struct sock_t sock;
	struct sock_bind_t sock_bind;
	struct user_param_t *user_param = &(rdma_resource.user_param);

	rc = parser(&(rdma_resource.user_param), argv, argc);
	if (rc) {
		return rc;
	}

    sock_bind.socket_fd  = -1;
    sock_bind.counter    = 0;
	sock_bind.is_daemon  = (user_param->server_ip == NULL) ? 1 : 0;
	sock_bind.port       = user_param->tcp_port;
	if (user_param->server_ip) {
		strcpy(sock_bind.ip, user_param->server_ip);
	}

	signal_init();
	sock_init(&sock);

	rc = rdma_resource_init(&sock, &rdma_resource);
	if (rc) {
		goto failure_0;
	}

	for (i = 0; i < LAT_LEVEL; i++) {
		rdma_resource.lat[i]  = 0;
	}
	rdma_resource.max_lat = 0;
	rdma_resource.min_lat = 0x7fffffff;

	rc = start_rdma_threads(&sock, &rdma_resource, &sock_bind);
	if (rc != 0) {
		ERROR("Testing failed");
	}

	rdma_resource_destroy(&rdma_resource);

failure_0:
	return rc;
}

