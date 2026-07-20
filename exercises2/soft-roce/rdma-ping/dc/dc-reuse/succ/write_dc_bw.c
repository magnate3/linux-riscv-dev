/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2005 Mellanox Technologies Ltd.  All rights reserved.
 * Copyright (c) 2009 HNR Consulting.  All rights reserved.
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
os*        disclaimer in the documentation and/or other materials
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
 *
 * $Id$
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "perftest_parameters.h"
#include "perftest_resources.h"
#include "perftest_communication.h"

/******************************************************************************
 ******************************************************************************/
static int dc_ctx_modify_qp_to_rts(struct ibv_qp *qp,
		struct ibv_qp_attr *attr,
		struct perftest_parameters *user_param,
		struct pingpong_dest *dest,
		struct pingpong_dest *my_dest,
		int qp_index)
{
	int num_of_qps = user_param->num_of_qps;
	int num_of_qps_per_port = user_param->num_of_qps / 2;
	int is_dc_server_side = 0;
	int flags = IBV_QP_STATE;
	int ooo_flags = 0;

	attr->qp_state = IBV_QPS_RTS;
	attr->ah_attr.src_path_bits = 0;

	/* in xrc with bidirectional,
	 * there are send qps and recv qps. the actual number of send/recv qps
	 * is num_of_qps / 2.
	 */
	if ((user_param->connection_type == DC || user_param->use_xrc) && (user_param->duplex || user_param->tst == LAT)) {
		num_of_qps /= 2;
		num_of_qps_per_port = num_of_qps / 2;
	}
	is_dc_server_side = ((!(user_param->duplex || user_param->tst == LAT) &&
						 (user_param->machine == SERVER)) ||
						  ((user_param->duplex || user_param->tst == LAT) &&
						 (qp_index >= num_of_qps)));
	/* first half of qps are for ib_port and second half are for ib_port2
	 * in xrc with bidirectional, the first half of qps are xrc_send qps and
	 * the second half are xrc_recv qps. the first half of the send/recv qps
	 * are for ib_port1 and the second half are for ib_port2
	 */
	if (user_param->dualport == ON && (qp_index % num_of_qps >= num_of_qps_per_port))
		attr->ah_attr.port_num = user_param->ib_port2;
	else
		attr->ah_attr.port_num = user_param->ib_port;

	if (user_param->connection_type != RawEth) {
		attr->ah_attr.dlid = (user_param->dlid) ? user_param->dlid : dest->lid;
		attr->ah_attr.sl = user_param->sl;

		if (((attr->ah_attr.port_num == user_param->ib_port) && (user_param->gid_index == DEF_GID_INDEX))
				|| ((attr->ah_attr.port_num == user_param->ib_port2) && (user_param->gid_index2 == DEF_GID_INDEX) && user_param->dualport)) {

			attr->ah_attr.is_global = 0;
		} else {

			attr->ah_attr.is_global  = 1;
			attr->ah_attr.grh.dgid = dest->gid;
			attr->ah_attr.grh.sgid_index = (attr->ah_attr.port_num == user_param->ib_port) ? user_param->gid_index : user_param->gid_index2;
			attr->ah_attr.grh.hop_limit = 0xFF;
			attr->ah_attr.grh.traffic_class = user_param->traffic_class;
		}
		if (user_param->connection_type != UD && user_param->connection_type != SRD) {
			if (user_param->connection_type == DC) {
				attr->path_mtu = user_param->curr_mtu;
				flags |= IBV_QP_AV | IBV_QP_PATH_MTU;
				if (is_dc_server_side)
				{
					attr->min_rnr_timer = MIN_RNR_TIMER;
					flags |= IBV_QP_MIN_RNR_TIMER;
				} //DCT
			}
			else {
				attr->path_mtu = user_param->curr_mtu;
				attr->dest_qp_num = dest->qpn;
				attr->rq_psn = dest->psn;

				flags |= (IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN);

				if (user_param->connection_type == RC || user_param->connection_type == XRC) {

					attr->max_dest_rd_atomic = my_dest->out_reads;
					attr->min_rnr_timer = MIN_RNR_TIMER;
					flags |= (IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC);
				}
			}
		}
	}
	else if (user_param->raw_qos) {
		attr->ah_attr.sl = user_param->sl;
		flags |= IBV_QP_AV;
	}

	#ifdef HAVE_OOO_ATTR
		ooo_flags |= IBV_QP_OOO_RW_DATA_PLACEMENT;
	#endif

	if (user_param->use_ooo)
		flags |= ooo_flags;
	//return ibv_modify_qp(qp, attr, flags);
	return 0;
}

int dc_ctx_connect(struct pingpong_context *ctx,
		struct pingpong_dest *dest,
		struct perftest_parameters *user_param,
		struct pingpong_dest *my_dest)
{
#if 0
	int i;
	struct ibv_qp_attr attr;
	int xrc_offset = 0;
	int flags = 0;

	if((user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT)) {
		xrc_offset = user_param->num_of_qps / 2;
	}
	for (i=0; i < user_param->num_of_qps; i++) {
		memset(&attr, 0, sizeof attr);
		if ((i >= xrc_offset) && (user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT))
			xrc_offset = -1*xrc_offset;
	        attr.ah_attr.grh.dgid = dest[xrc_offset + i].gid;
		//fprintf(stdout, " modify dest QP gid %0x\n",dest[xrc_offset + i].gid);
	        if (0!= ibv_modify_qp(ctx->qp[i], &attr, flags))
		{
			fprintf(stderr, "Failed to modify QP %d dest\n",ctx->qp[i]->qp_num);

		}
         }
#elif 0

	int i;
	int xrc_offset = 0;
	if((user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT)) {
		xrc_offset = user_param->num_of_qps / 2;
	}
	for (i=0; i < user_param->num_of_qps; i++) {
		if (NULL == ctx->dc_ah[i])
		{
			fprintf(stderr, "Failed to modify QP ah %d dest,dc ah is NULL\n",ctx->qp[i]->qp_num);
			return -1;
		}
	}
	for (i=0; i < user_param->num_of_qps; i++) {
		ibv_destroy_ah(ctx->ah[i]);
		ctx->ah[i] = ctx->dc_ah[i];
		if((user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT))
			xrc_offset = user_param->num_of_qps / 2;
         }
#elif 1
	int i;
	int xrc_offset = 0;
	struct ibv_qp_attr attr;
	if((user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT)) {
		xrc_offset = user_param->num_of_qps / 2;
	}
	for (i=0; i < user_param->num_of_qps; i++) {
		if(dc_ctx_modify_qp_to_rts(ctx->qp[i], &attr, user_param, &dest[xrc_offset + i], &my_dest[i], i)) {
			fprintf(stderr, "Failed to modify QP %d to RTR\n",ctx->qp[i]->qp_num);
			return FAILURE;
		}
		ibv_destroy_ah(ctx->ah[i]);
	ctx->ah[i] = ibv_create_ah(ctx->pd,&(attr.ah_attr));

	if (!ctx->ah[i]) {
		fprintf(stderr, "Failed to create AH\n");
		return FAILURE;
	}
		if((user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT))
			xrc_offset = user_param->num_of_qps / 2;
         }
#else
	int i;
	struct ibv_ah_attr  ah_attr;
	int xrc_offset = 0;
	int flags = 0;
        ah_attr.is_global  = 1;
	//ah_attr.dlid = (user_param->dlid) ? user_param->dlid : dest->lid;
	ah_attr.port_num = user_param->ib_port;
        //ah_attr.grh.dgid = dest->gid;
        ah_attr.grh.sgid_index =  user_param->gid_index ;
        //ah_attr.grh.sgid_index = (attr->ah_attr.port_num == user_param->ib_port) ? user_param->gid_index : user_param->gid_index2;
        ah_attr.grh.hop_limit = 0xFF;
        ah_attr.grh.traffic_class = user_param->traffic_class;
	if((user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT)) {
		xrc_offset = user_param->num_of_qps / 2;
	}
	for (i=0; i < user_param->num_of_qps; i++) {
		memset(&ah_attr, 0, sizeof ah_attr);
		if ((i >= xrc_offset) && (user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT))
			xrc_offset = -1*xrc_offset;
		ibv_destroy_ah(ctx->ah[i]);
	        ah_attr.grh.dgid = dest[xrc_offset + i].gid;
	        ah_attr.dlid = dest[xrc_offset + i].lid;
		ctx->ah[i] = ibv_create_ah(ctx->pd, &ah_attr);
		//fprintf(stdout, " modify dest QP gid %0x\n",dest[xrc_offset + i].gid);
	        //if (0!= ibv_modify_qp(ctx->qp[i], &attr, flags))
		if (NULL == ctx->ah[i])
		{
			fprintf(stderr, "Failed to modify QP ah %d dest\n",ctx->qp[i]->qp_num);

			return -1;
		}

		if((user_param->use_xrc || user_param->connection_type == DC) && (user_param->duplex || user_param->tst == LAT))
			xrc_offset = user_param->num_of_qps / 2;
         }
	          
	   
#endif
	 return 0;
}
int main(int argc, char *argv[])
{
	int				ret_parser, i = 0, rc;
	struct ibv_device		*ib_dev = NULL;
	struct pingpong_context		ctx;
	struct pingpong_dest		*my_dest,*rem_dest;
	struct perftest_parameters	user_param;
	struct perftest_comm		user_comm;
	struct bw_report_data		my_bw_rep, rem_bw_rep;
	unsigned char again = 1;

	/* init default values to user's parameters */
	memset(&user_param,0,sizeof(struct perftest_parameters));
	memset(&user_comm,0,sizeof(struct perftest_comm));
	memset(&ctx,0,sizeof(struct pingpong_context));

	user_param.verb    = WRITE;
	user_param.tst     = BW;
	strncpy(user_param.version, VERSION, sizeof(user_param.version));

	/* Configure the parameters values according to user arguments or default values. */
	ret_parser = parser(&user_param,argv,argc);
	if (ret_parser) {
		if (ret_parser != VERSION_EXIT && ret_parser != HELP_EXIT)
			fprintf(stderr," Parser function exited with Error\n");
		return FAILURE;
	}

	if((user_param.connection_type == DC || user_param.use_xrc) && user_param.duplex) {
		user_param.num_of_qps *= 2;
	}

	/* Finding the IB device selected (or default if none is selected). */
	ib_dev = ctx_find_dev(&user_param.ib_devname);
	if (!ib_dev) {
		fprintf(stderr," Unable to find the Infiniband/RoCE device\n");
		return FAILURE;
	}

	/* Getting the relevant context from the device */
	ctx.context = ctx_open_device(ib_dev, &user_param);
	if (!ctx.context) {
		fprintf(stderr, " Couldn't get context for the device\n");
		return FAILURE;
	}

	/* Verify user parameters that require the device context,
	 * the function will print the relevent error info. */
	if (verify_params_with_device_context(ctx.context, &user_param))
	{
		fprintf(stderr, " Couldn't get context for the device\n");
		return FAILURE;
	}

	/* See if MTU and link type are valid and supported. */
	if (check_link(ctx.context,&user_param)) {
		fprintf(stderr, " Couldn't get context for the device\n");
		return FAILURE;
	}

	/* copy the relevant user parameters to the comm struct + creating rdma_cm resources. */
	if (create_comm_struct(&user_comm,&user_param)) {
		fprintf(stderr," Unable to create RDMA_CM resources\n");
		return FAILURE;
	}

	if (user_param.output == FULL_VERBOSITY && user_param.machine == SERVER) {
		printf("\n************************************\n");
		printf("* Waiting for client to connect... *\n");
		printf("************************************\n");
	}

	/* Initialize the connection and print the local data. */
	if (establish_connection(&user_comm)) {
		fprintf(stderr," Unable to init the socket connection\n");
		dealloc_comm_struct(&user_comm,&user_param);
		return FAILURE;
	}
	sleep(1);
	exchange_versions(&user_comm, &user_param);
	check_version_compatibility(&user_param);
	check_sys_data(&user_comm, &user_param);

	/* See if MTU and link type are valid and supported. */
	if (check_mtu(ctx.context,&user_param, &user_comm)) {
		fprintf(stderr, " Couldn't get context for the device\n");
		dealloc_comm_struct(&user_comm,&user_param);
		return FAILURE;
	}

	MAIN_ALLOC(my_dest , struct pingpong_dest , user_param.num_of_qps , return_error);
	memset(my_dest, 0, sizeof(struct pingpong_dest)*user_param.num_of_qps);
	MAIN_ALLOC(rem_dest , struct pingpong_dest , user_param.num_of_qps , free_my_dest);
	memset(rem_dest, 0, sizeof(struct pingpong_dest)*user_param.num_of_qps);

	/* Allocating arrays needed for the test. */
	if(alloc_ctx(&ctx,&user_param)){
		fprintf(stderr, "Couldn't allocate context\n");
		dealloc_comm_struct(&user_comm,&user_param);
		goto free_mem;
	}

	/* Create RDMA CM resources and connect through CM. */
	if (user_param.work_rdma_cm == ON) {
		rc = create_rdma_cm_connection(&ctx, &user_param, &user_comm,
			my_dest, rem_dest);
		if (rc) {
			fprintf(stderr,
				"Failed to create RDMA CM connection with resources.\n");
			dealloc_comm_struct(&user_comm,&user_param);
			dealloc_ctx(&ctx, &user_param);
			goto free_mem;
		}
	} else {
		/* create all the basic IB resources (data buffer, PD, MR, CQ and events channel) */
		if (ctx_init(&ctx, &user_param)) {
			fprintf(stderr, " Couldn't create IB resources\n");
			dealloc_comm_struct(&user_comm,&user_param);
			dealloc_ctx(&ctx, &user_param);
			goto free_mem;
		}
	}

	/* Set up the Connection. */
	if (set_up_connection(&ctx,&user_param,my_dest)) {
		fprintf(stderr," Unable to set up socket connection\n");
		goto destroy_context;
	}

	/* Print basic test information. */
	ctx_print_test_info(&user_param);

	for (i=0; i < user_param.num_of_qps; i++) {

		if (ctx_hand_shake(&user_comm,&my_dest[i],&rem_dest[i])) {
			fprintf(stderr," Failed to exchange data between server and clients\n");
			goto destroy_context;
		}
	}

	if (user_param.work_rdma_cm == OFF) {
		if (ctx_check_gid_compatibility(&my_dest[0], &rem_dest[0])) {
			fprintf(stderr,"\n Found Incompatibility issue with GID types.\n");
			fprintf(stderr," Please Try to use a different IP version.\n\n");
			goto destroy_context;
		}
	}

	if (user_param.work_rdma_cm == OFF) {
		if (ctx_connect(&ctx,rem_dest,&user_param,my_dest)) {
			fprintf(stderr," Unable to Connect the HCA's through the link\n");
			goto destroy_context;
		}
	}

	if (user_param.connection_type == DC)
	{
		/* Set up connection one more time to send qpn properly for DC */
		if (set_up_connection(&ctx, &user_param, my_dest))
		{
			fprintf(stderr," Unable to set up socket connection\n");
			goto destroy_context;
		}
	}

	/* Print this machine QP information */
	for (i=0; i < user_param.num_of_qps; i++)
		ctx_print_pingpong_data(&my_dest[i],&user_comm);

	user_comm.rdma_params->side = REMOTE;

	for (i=0; i < user_param.num_of_qps; i++) {

		if (ctx_hand_shake(&user_comm,&my_dest[i],&rem_dest[i])) {
			fprintf(stderr," Failed to exchange data between server and clients\n");
			goto destroy_context;
		}

		ctx_print_pingpong_data(&rem_dest[i],&user_comm);
	}

	/* An additional handshake is required after moving qp to RTR. */
	if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
		fprintf(stderr," Failed to exchange data between server and clients\n");
		goto destroy_context;
	}

	if (user_param.output == FULL_VERBOSITY) {
		if (user_param.report_per_port) {
			printf(RESULT_LINE_PER_PORT);
			printf((user_param.report_fmt == MBS ? RESULT_FMT_PER_PORT : RESULT_FMT_G_PER_PORT));
		}
		else {
			printf(RESULT_LINE);
			printf((user_param.report_fmt == MBS ? RESULT_FMT : RESULT_FMT_G));
		}

		printf((user_param.cpu_util_data.enable ? RESULT_EXT_CPU_UTIL : RESULT_EXT));
	}

	/* For half duplex tests, server just waits for client to exit */
	if (user_param.machine == SERVER && !user_param.duplex) {

		if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr," Failed to exchange data between server and clients\n");
			goto free_mem;
		}

		xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
		print_full_bw_report(&user_param, &rem_bw_rep, NULL);

		if (ctx_close_connection(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr,"Failed to close connection between server and client\n");
			goto free_mem;
		}

		if (user_param.output == FULL_VERBOSITY) {
			if (user_param.report_per_port)
				printf(RESULT_LINE_PER_PORT);
			else
				printf(RESULT_LINE);
		}

		if (user_param.work_rdma_cm == ON) {
			if (destroy_ctx(&ctx,&user_param)) {
				fprintf(stderr, "Failed to destroy resources\n");
				goto destroy_cm_context;
			}

			user_comm.rdma_params->work_rdma_cm = OFF;
			free(my_dest);
			free(rem_dest);
			return destroy_ctx(user_comm.rdma_ctx,user_comm.rdma_params);
		}

		free(my_dest);
		free(rem_dest);
		return destroy_ctx(&ctx,&user_param);
	}

	if (user_param.test_method == RUN_ALL) {

		for (i = 1; i < 24 ; ++i) {

			user_param.size = (uint64_t)1 << i;
			ctx_set_send_wqes(&ctx,&user_param,rem_dest);

			if (user_param.perform_warm_up) {
				if(perform_warm_up(&ctx, &user_param)) {
					fprintf(stderr, "Problems with warm up\n");
					goto free_mem;
				}
			}

			if(user_param.duplex) {
				if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
					fprintf(stderr,"Failed to sync between server and client between different msg sizes\n");
					goto free_mem;
				}
			}

			if(run_iter_bw(&ctx,&user_param)) {
				fprintf(stderr," Failed to complete run_iter_bw function successfully\n");
				goto free_mem;
			}

			if (user_param.duplex && (atof(user_param.version) >= 4.6)) {
				if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
					fprintf(stderr,"Failed to sync between server and client between different msg sizes\n");
					goto free_mem;
				}
			}

			print_report_bw(&user_param,&my_bw_rep);

			if (user_param.duplex) {
				xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
				print_full_bw_report(&user_param, &my_bw_rep, &rem_bw_rep);
			}
		}

	} else if (user_param.test_method == RUN_REGULAR) {

		ctx_set_send_wqes(&ctx,&user_param,rem_dest);

		if (user_param.verb != SEND) {

			if (user_param.perform_warm_up) {
				if(perform_warm_up(&ctx, &user_param)) {
					fprintf(stderr, "Problems with warm up\n");
					goto free_mem;
				}
			}
		}

		if(user_param.duplex) {
			if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
				fprintf(stderr,"Failed to sync between server and client between different msg sizes\n");
				goto free_mem;
			}
		}

		if(run_iter_bw(&ctx,&user_param)) {
			fprintf(stderr," Failed to complete run_iter_bw function successfully\n");
			goto free_mem;
		}

		print_report_bw(&user_param,&my_bw_rep);

		if (user_param.duplex) {
			xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
			print_full_bw_report(&user_param, &my_bw_rep, &rem_bw_rep);
		}

		if (user_param.report_both && user_param.duplex) {
			printf(RESULT_LINE);
			printf("\n Local results: \n");
			printf(RESULT_LINE);
			printf((user_param.report_fmt == MBS ? RESULT_FMT : RESULT_FMT_G));
			printf((user_param.cpu_util_data.enable ? RESULT_EXT_CPU_UTIL : RESULT_EXT));
			print_full_bw_report(&user_param, &my_bw_rep, NULL);
			printf(RESULT_LINE);

			printf("\n Remote results: \n");
			printf(RESULT_LINE);
			printf((user_param.report_fmt == MBS ? RESULT_FMT : RESULT_FMT_G));
			printf((user_param.cpu_util_data.enable ? RESULT_EXT_CPU_UTIL : RESULT_EXT));
			print_full_bw_report(&user_param, &rem_bw_rep, NULL);
		}
	} else if (user_param.test_method == RUN_INFINITELY) {

		ctx_set_send_wqes(&ctx,&user_param,rem_dest);

		if(run_iter_bw_infinitely(&ctx,&user_param)) {
			fprintf(stderr," Error occurred while running infinitely! aborting ...\n");
			goto free_mem;
		}
	}

	if (user_param.output == FULL_VERBOSITY) {
		if (user_param.report_per_port)
			printf(RESULT_LINE_PER_PORT);
		else
			printf(RESULT_LINE);
	}

	/* For half duplex tests, server just waits for client to exit */
	if (user_param.machine == CLIENT && !user_param.duplex) {

		if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr," Failed to exchange data between server and clients\n");
			goto free_mem;
		}

		xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
	}

	/* Closing connection. */
	if (ctx_close_connection(&user_comm,&my_dest[0],&rem_dest[0])) {
		fprintf(stderr,"Failed to close connection between server and client\n");
		goto free_mem;
	}

	if(again && (user_param.connection_type == DC || user_param.connection_type == UC))
	{
	    printf("***********again ******************* \n");
	    const char *ip_str = "10.22.116.222";
	     struct in_addr ip_addr;
	    uint32_t ip_int = 0;
	    inet_pton(AF_INET, ip_str, &ip_addr);
	    ip_int = ntohl(ip_addr.s_addr);
	    again = 0;
	    close(user_comm.rdma_params->sockfd);
	    // tcp establish_connection
	    /* Initialize the connection and print the local data. */
	    user_param.port += 1;
	    user_param.remote_ip = ip_int;
	    user_param.server_ip = ip_int;
	    user_param.machine = CLIENT;
            //user_comm.rdma_params->port +=1;
	    strncpy(user_comm.rdma_params->servername,"10.22.116.222",strlen("10.22.116.222"));
            user_comm.rdma_params->port +=1;
	    if (establish_connection(&user_comm)) {
	    	fprintf(stderr," Unable to init the socket connection\n");
	    	dealloc_comm_struct(&user_comm,&user_param);
	    	return FAILURE;
	    }
	    exchange_versions(&user_comm, &user_param);
	    check_version_compatibility(&user_param);
	    check_sys_data(&user_comm, &user_param);
	    /* See if MTU and link type are valid and supported. */
	    if (check_mtu(ctx.context,&user_param, &user_comm)) {
	    	fprintf(stderr, " Couldn't get context for the device\n");
	    	dealloc_comm_struct(&user_comm,&user_param);
	    	return FAILURE;
	    }

	    ctx_print_test_info(&user_param);
	    /* Print this machine QP information */
	    for (i=0; i < user_param.num_of_qps; i++)
	    	ctx_print_pingpong_data(&my_dest[i],&user_comm);

	    user_comm.rdma_params->side = REMOTE;

	    usleep(50000);
	    fprintf(stdout," all qps hand shake \n");
	for (i=0; i < user_param.num_of_qps; i++) {

		if (ctx_hand_shake(&user_comm,&my_dest[i],&rem_dest[i])) {
			fprintf(stderr," Failed to exchange data between server and clients\n");
			goto destroy_context;
		}
	}

	if (user_param.work_rdma_cm == OFF) {
		if (ctx_check_gid_compatibility(&my_dest[0], &rem_dest[0])) {
			fprintf(stderr,"\n Found Incompatibility issue with GID types.\n");
			fprintf(stderr," Please Try to use a different IP version.\n\n");
			goto destroy_context;
		}
	}

	if (user_param.work_rdma_cm == OFF) {
		// not need to modify QP to RTR
		//if (ctx_connect(&ctx,rem_dest,&user_param,my_dest)) {
		//	fprintf(stderr," Unable to Connect the HCA's through the link\n");
		//	goto destroy_context;
		//}
	}

#if 1
	if (user_param.connection_type == DC)
	{
	  // change dest gidx attr.ah_attr.grh.dgid 
	  if (dc_ctx_connect(&ctx,rem_dest,&user_param,my_dest)) {
	    		fprintf(stderr," Unable to Connect the HCA's through the link\n");
	 		goto destroy_context;
	  }
	}
#endif
	/* Print this machine QP information */
	for (i=0; i < user_param.num_of_qps; i++)
		ctx_print_pingpong_data(&my_dest[i],&user_comm);

	user_comm.rdma_params->side = REMOTE;

	for (i=0; i < user_param.num_of_qps; i++) {

		if (ctx_hand_shake(&user_comm,&my_dest[i],&rem_dest[i])) {
			fprintf(stderr," Failed to exchange data between server and clients\n");
			goto destroy_context;
		}

		ctx_print_pingpong_data(&rem_dest[i],&user_comm);
	}
#if 0
	if (user_param.connection_type == DC)
	{
	  // change dest gidx attr.ah_attr.grh.dgid 
	  if (dc_ctx_connect(&ctx,rem_dest,&user_param,my_dest)) {
	    		fprintf(stderr," Unable to Connect the HCA's through the link\n");
	 		goto destroy_context;
	  }
	}
#endif
	/* An additional handshake is required after moving qp to RTR. */
	if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
		fprintf(stderr," Failed to exchange data between server and clients\n");
		goto destroy_context;
	}

	if (user_param.output == FULL_VERBOSITY) {
		if (user_param.report_per_port) {
			printf(RESULT_LINE_PER_PORT);
			printf((user_param.report_fmt == MBS ? RESULT_FMT_PER_PORT : RESULT_FMT_G_PER_PORT));
		}
		else {
			printf(RESULT_LINE);
			printf((user_param.report_fmt == MBS ? RESULT_FMT : RESULT_FMT_G));
		}

		printf((user_param.cpu_util_data.enable ? RESULT_EXT_CPU_UTIL : RESULT_EXT));
	}

	/* For half duplex tests, server just waits for client to exit */
	if (user_param.machine == SERVER && !user_param.duplex) {

		if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr," Failed to exchange data between server and clients\n");
			goto free_mem;
		}

		xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
		print_full_bw_report(&user_param, &rem_bw_rep, NULL);

		if (ctx_close_connection(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr,"Failed to close connection between server and client\n");
			goto free_mem;
		}

		if (user_param.output == FULL_VERBOSITY) {
			if (user_param.report_per_port)
				printf(RESULT_LINE_PER_PORT);
			else
				printf(RESULT_LINE);
		}

		if (user_param.work_rdma_cm == ON) {
			if (destroy_ctx(&ctx,&user_param)) {
				fprintf(stderr, "Failed to destroy resources\n");
				goto destroy_cm_context;
			}

			user_comm.rdma_params->work_rdma_cm = OFF;
			free(my_dest);
			free(rem_dest);
			return destroy_ctx(user_comm.rdma_ctx,user_comm.rdma_params);
		}

		free(my_dest);
		free(rem_dest);
		return destroy_ctx(&ctx,&user_param);
	}
	printf("********* begin to iter ************* \n");
	if (user_param.test_method == RUN_ALL) {

		for (i = 1; i < 24 ; ++i) {

			user_param.size = (uint64_t)1 << i;
			ctx_set_send_wqes(&ctx,&user_param,rem_dest);

			if (user_param.perform_warm_up) {
				if(perform_warm_up(&ctx, &user_param)) {
					fprintf(stderr, "Problems with warm up\n");
					goto free_mem;
				}
			}

			if(user_param.duplex) {
				if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
					fprintf(stderr,"Failed to sync between server and client between different msg sizes\n");
					goto free_mem;
				}
			}

			if(run_iter_bw(&ctx,&user_param)) {
				fprintf(stderr," Failed to complete run_iter_bw function successfully\n");
				goto free_mem;
			}

			if (user_param.duplex && (atof(user_param.version) >= 4.6)) {
				if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
					fprintf(stderr,"Failed to sync between server and client between different msg sizes\n");
					goto free_mem;
				}
			}

			print_report_bw(&user_param,&my_bw_rep);

			if (user_param.duplex) {
				xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
				print_full_bw_report(&user_param, &my_bw_rep, &rem_bw_rep);
			}
		}

	} else if (user_param.test_method == RUN_REGULAR) {

		ctx_set_send_wqes(&ctx,&user_param,rem_dest);

		if (user_param.verb != SEND) {

			if (user_param.perform_warm_up) {
				if(perform_warm_up(&ctx, &user_param)) {
					fprintf(stderr, "Problems with warm up\n");
					goto free_mem;
				}
			}
		}

		if(user_param.duplex) {
			if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
				fprintf(stderr,"Failed to sync between server and client between different msg sizes\n");
				goto free_mem;
			}
		}

		if(run_iter_bw(&ctx,&user_param)) {
			fprintf(stderr," Failed to complete run_iter_bw function successfully\n");
			goto free_mem;
		}

		print_report_bw(&user_param,&my_bw_rep);

		if (user_param.duplex) {
			xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
			print_full_bw_report(&user_param, &my_bw_rep, &rem_bw_rep);
		}

		if (user_param.report_both && user_param.duplex) {
			printf(RESULT_LINE);
			printf("\n Local results: \n");
			printf(RESULT_LINE);
			printf((user_param.report_fmt == MBS ? RESULT_FMT : RESULT_FMT_G));
			printf((user_param.cpu_util_data.enable ? RESULT_EXT_CPU_UTIL : RESULT_EXT));
			print_full_bw_report(&user_param, &my_bw_rep, NULL);
			printf(RESULT_LINE);

			printf("\n Remote results: \n");
			printf(RESULT_LINE);
			printf((user_param.report_fmt == MBS ? RESULT_FMT : RESULT_FMT_G));
			printf((user_param.cpu_util_data.enable ? RESULT_EXT_CPU_UTIL : RESULT_EXT));
			print_full_bw_report(&user_param, &rem_bw_rep, NULL);
		}
	} else if (user_param.test_method == RUN_INFINITELY) {

		ctx_set_send_wqes(&ctx,&user_param,rem_dest);

		if(run_iter_bw_infinitely(&ctx,&user_param)) {
			fprintf(stderr," Error occurred while running infinitely! aborting ...\n");
			goto free_mem;
		}
	}
	if (user_param.output == FULL_VERBOSITY) {
		if (user_param.report_per_port)
			printf(RESULT_LINE_PER_PORT);
		else
			printf(RESULT_LINE);
	}

	/* For half duplex tests, server just waits for client to exit */
	if (user_param.machine == CLIENT && !user_param.duplex) {

		if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr," Failed to exchange data between server and clients\n");
			goto free_mem;
		}

		xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
	}

	/* Closing connection. */
	if (ctx_close_connection(&user_comm,&my_dest[0],&rem_dest[0])) {
		fprintf(stderr,"Failed to close connection between server and client\n");
		goto free_mem;
	}
        } //again
	if (!user_param.is_bw_limit_passed && (user_param.is_limit_bw == ON ) ) {
		fprintf(stderr,"Error: BW result is below bw limit\n");
		goto destroy_context;
	}

	if (!user_param.is_msgrate_limit_passed && (user_param.is_limit_bw == ON )) {
		fprintf(stderr,"Error: Msg rate  is below msg_rate limit\n");
		goto destroy_context;
	}

	if (user_param.work_rdma_cm == ON) {
		if (destroy_ctx(&ctx,&user_param)) {
			fprintf(stderr, "Failed to destroy resources\n");
			goto destroy_cm_context;
		}

		user_comm.rdma_params->work_rdma_cm = OFF;
		free(rem_dest);
		free(my_dest);
		return destroy_ctx(user_comm.rdma_ctx,user_comm.rdma_params);
	}

	free(rem_dest);
	free(my_dest);
	return destroy_ctx(&ctx,&user_param);

destroy_context:
	if (destroy_ctx(&ctx,&user_param))
		fprintf(stderr, "Failed to destroy resources\n");
destroy_cm_context:
	if (user_param.work_rdma_cm == ON) {
		user_comm.rdma_params->work_rdma_cm = OFF;
		destroy_ctx(user_comm.rdma_ctx,user_comm.rdma_params);
	}
free_mem:
	free(rem_dest);
free_my_dest:
	free(my_dest);
return_error:
	return FAILURE;
}
