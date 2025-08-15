/*
 * Copyright (c) 2005 Ammasso, Inc. All rights reserved.
 * Copyright (c) 2006 Open Grid Computing, Inc. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "rping.h"

int debug = 0;

void usage(char *name)
{
	printf("%s -s [-vVd] [-S size] [-C count] [-a addr] [-p port]\n", 
	       basename(name));
	printf("%s -c [-vVd] [-S size] [-C count] -a addr [-p port]\n", 
	       basename(name));
	printf("\t-c\t\tclient side\n");
	printf("\t-s\t\tserver side.  To bind to any address with IPv6 use -a ::0\n");
	printf("\t-v\t\tdisplay ping data to stdout\n");
	printf("\t-V\t\tvalidate ping data\n");
	printf("\t-d\t\tdebug printfs\n");
	printf("\t-S size \tping data size\n");
	printf("\t-C count\tping count times\n");
	printf("\t-a addr\t\taddress\n");
	printf("\t-p port\t\tport\n");
	printf("\t-P\t\tpersistent server mode allowing multiple connections\n");
}


int main(int argc, char *argv[])
{
	struct rping_cb *cb;
	int op;
	int ret = 0;
	int persistent_server = 0;

	cb = malloc(sizeof(*cb));
	if (!cb)
		return -ENOMEM;

	memset(cb, 0, sizeof(*cb));
	cb->server = -1;
	cb->state = IDLE;
	cb->size = 64;
	cb->sin.ss_family = PF_INET;
	cb->port = htons(7174);
	sem_init(&cb->sem, 0, 0);

	opterr = 0;
	while ((op=getopt(argc, argv, "a:Pp:C:S:t:scvVd")) != -1) {
		switch (op) {
		case 'a':
			ret = get_addr(optarg, (struct sockaddr *) &cb->sin);
			break;
		case 'P':
			persistent_server = 1;
			break;
		case 'p':
			cb->port = htons(atoi(optarg));
			DEBUG_LOG("port %d\n", (int) atoi(optarg));
			break;
		case 's':
			cb->server = 1;
			DEBUG_LOG("server\n");
			break;
		case 'c':
			cb->server = 0;
			DEBUG_LOG("client\n");
			break;
		case 'S':
			cb->size = atoi(optarg);
			if ((cb->size < RPING_MIN_BUFSIZE) ||
			    (cb->size > (RPING_BUFSIZE - 1))) {
				fprintf(stderr, "Invalid size %d "
				       "(valid range is %Zd to %d)\n",
				       cb->size, RPING_MIN_BUFSIZE, RPING_BUFSIZE);
				ret = EINVAL;
			} else
				DEBUG_LOG("size %d\n", (int) atoi(optarg));
			break;
		case 'C':
			cb->count = atoi(optarg);
			if (cb->count < 0) {
				fprintf(stderr, "Invalid count %d\n",
					cb->count);
				ret = EINVAL;
			} else
				DEBUG_LOG("count %d\n", (int) cb->count);
			break;
		case 'v':
			cb->verbose++;
			DEBUG_LOG("verbose\n");
			break;
		case 'V':
			cb->validate++;
			DEBUG_LOG("validate data\n");
			break;
		case 'd':
			debug++;
			break;
		default:
			usage("rping");
			ret = EINVAL;
			goto out;
		}
	}
	if (ret)
		goto out;

	if (cb->server == -1) {
		usage("rping");
		ret = EINVAL;
		goto out;
	}

	cb->cm_channel = rdma_create_event_channel();
	if (!cb->cm_channel) {
		perror("rdma_create_event_channel");
		ret = errno;
		goto out;
	}

	ret = rdma_create_id(cb->cm_channel, &cb->cm_id, cb, RDMA_PS_TCP);
	if (ret) {
		perror("rdma_create_id");
		goto out2;
	}
	DEBUG_LOG("created cm_id %p\n", cb->cm_id);

	ret = pthread_create(&cb->cmthread, NULL, cm_thread, cb);
	if (ret) {
		perror("pthread_create");
		goto out2;
	}

	if (cb->server) {
		if (persistent_server)
			ret = rping_run_persistent_server(cb);
		else
			ret = rping_run_server(cb);
	} else {
		ret = rping_run_client(cb);
	}

	DEBUG_LOG("destroy cm_id %p\n", cb->cm_id);
	rdma_destroy_id(cb->cm_id);
out2:
	rdma_destroy_event_channel(cb->cm_channel);
out:
	free(cb);
	return ret;
}
