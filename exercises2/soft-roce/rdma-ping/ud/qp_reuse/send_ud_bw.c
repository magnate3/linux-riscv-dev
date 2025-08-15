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
 *
 * $Id$
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "perftest_parameters.h"
#include "perftest_resources.h"
#include "multicast_resources.h"
#include "perftest_communication.h"

/******************************************************************************
 *
 ******************************************************************************/
static int set_mcast_group(struct pingpong_context *ctx,
		struct perftest_parameters *user_param,
		struct mcast_parameters *mcg_params)
{
	struct ibv_port_attr port_attr;

	if (ibv_query_gid(ctx->context,user_param->ib_port,user_param->gid_index,&mcg_params->port_gid)) {
		return FAILURE;
	}

	if (ibv_query_pkey(ctx->context,user_param->ib_port,DEF_PKEY_IDX,&mcg_params->pkey)) {
		return FAILURE;
	}

	if (ibv_query_port(ctx->context,user_param->ib_port,&port_attr)) {
		return FAILURE;
	}
	mcg_params->sm_lid  = port_attr.sm_lid;
	mcg_params->sm_sl   = port_attr.sm_sl;
	mcg_params->ib_port = user_param->ib_port;
	mcg_params->ib_ctx  = ctx->context;
	mcg_params->ib_devname = user_param->ib_devname;

	if (!strcmp(link_layer_str(user_param->link_type),"IB")) {
		/* Request for Mcast group create registery in SM. */
		if (join_multicast_group(SUBN_ADM_METHOD_SET,mcg_params)) {
			fprintf(stderr,"Couldn't Register the Mcast group on the SM\n");
			return FAILURE;
		}
	}
	return 0;
}

/******************************************************************************
 *
 ******************************************************************************/
static int send_set_up_connection(struct pingpong_context *ctx,
		struct perftest_parameters *user_param,
		struct pingpong_dest *my_dest,
		struct mcast_parameters *mcg_params,
		struct perftest_comm *comm)
{

	if (set_up_connection(ctx,user_param,my_dest)) {
		fprintf(stderr," Unable to set up my IB connection parameters\n");
		return FAILURE;
	}

	if (user_param->use_mcg && (user_param->duplex || user_param->machine == SERVER)) {
		int i;
		mcg_params->user_mgid = user_param->user_mgid;
		set_multicast_gid(mcg_params,ctx->qp[0]->qp_num,(int)user_param->machine);
		if (set_mcast_group(ctx,user_param,mcg_params)) {
			return FAILURE;
		}

		for (i=0; i < user_param->num_of_qps; i++) {
			if (ibv_attach_mcast(ctx->qp[i],&mcg_params->mgid,mcg_params->mlid)) {
				fprintf(stderr, "Couldn't attach QP to MultiCast group");
				return FAILURE;
			}
		}

		mcg_params->mcast_state |= MCAST_IS_ATTACHED;
		my_dest->gid = mcg_params->mgid;
		my_dest->lid = mcg_params->mlid;
		my_dest->qpn = QPNUM_MCAST;
	}
	return 0;
}

/******************************************************************************
 *
 ******************************************************************************/
static int send_destroy_ctx(
		struct pingpong_context *ctx,
		struct perftest_parameters *user_param,
		struct mcast_parameters *mcg_params)
{
	if (user_param->use_mcg) {

		if (user_param->duplex || user_param->machine == SERVER) {
			int i;
			for (i=0; i < user_param->num_of_qps; i++) {
				if (ibv_detach_mcast(ctx->qp[i],&mcg_params->mgid,mcg_params->mlid)) {
					fprintf(stderr, "Couldn't attach QP to MultiCast group");
					return FAILURE;
				}
			}
		}

		/* Removal Request for Mcast group in SM if needed. */
		if (!strcmp(link_layer_str(user_param->link_type),"IB")) {
			if (join_multicast_group(SUBN_ADM_METHOD_DELETE,mcg_params)) {
				fprintf(stderr,"Couldn't Unregister the Mcast group on the SM\n");
				return FAILURE;
			}
		}
	}
	return destroy_ctx(ctx,user_param);
}

int restart_stage_ctx_close_connection(struct perftest_comm *comm,
				struct pingpong_dest *my_dest,
						struct pingpong_dest *rem_dest)
{
		/*Signal client is finished.*/
		//if (ctx_hand_shake(comm,my_dest,rem_dest)) {
		//			return 1;
		//				}

			if (!comm->rdma_params->use_rdma_cm && !comm->rdma_params->work_rdma_cm) {
						close(comm->rdma_params->sockfd);
								return 0;
									}

				return 0;
}
int ud_ctx_connect(struct pingpong_context *ctx,
		struct pingpong_dest *dest,
		struct perftest_parameters *user_param,
		struct pingpong_dest *my_dest)
{
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
	 return 0;
}
/******************************************************************************
 *
 ******************************************************************************/
int main(int argc, char *argv[])
{
	struct ibv_device		*ib_dev = NULL;
	struct pingpong_context  	ctx;
	struct pingpong_dest	 	*my_dest  = NULL;
	struct pingpong_dest		*rem_dest = NULL;
	struct perftest_parameters  	user_param;
	struct perftest_comm		user_comm;
	struct mcast_parameters     	mcg_params;
	struct bw_report_data		my_bw_rep, rem_bw_rep;
	int                      	ret_parser, i = 0, rc;
	int                      	size_max_pow = 24;
	int 						error = 1;
	unsigned char again = 1;

	/* init default values to user's parameters */
	memset(&ctx, 0,sizeof(struct pingpong_context));
	memset(&user_param, 0 , sizeof(struct perftest_parameters));
	memset(&mcg_params, 0 , sizeof(struct mcast_parameters));
	memset(&user_comm, 0,sizeof(struct perftest_comm));

	user_param.verb    = SEND;
	user_param.tst     = BW;
	strncpy(user_param.version, VERSION, sizeof(user_param.version));

	/* Configure the parameters values according to user arguments or defalut values. */
	ret_parser = parser(&user_param,argv,argc);
	if (ret_parser) {
		if (ret_parser != VERSION_EXIT && ret_parser != HELP_EXIT)
			fprintf(stderr," Parser function exited with Error\n");
		return FAILURE;
	}
	if((user_param.connection_type == DC || user_param.use_xrc) && user_param.duplex) {
		user_param.num_of_qps *= 2;
	}
	/* Checking that the user did not run with RawEth. for this we have raw_etherent_bw test. */
	if (user_param.connection_type == RawEth) {
		fprintf(stderr," This test cannot run Raw Ethernet QPs (you have chosen RawEth as connection type\n");
		fprintf(stderr," For this we have raw_ethernet_bw test in this package.\n");
		return FAILURE;
	}

	/* Finding the IB device selected (or defalut if no selected). */
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
	if (verify_params_with_device_context(ctx.context, &user_param)) {
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

	exchange_versions(&user_comm, &user_param);
	check_version_compatibility(&user_param);
	check_sys_data(&user_comm, &user_param);

	/* See if MTU and link type are valid and supported. */
	if (check_mtu(ctx.context,&user_param, &user_comm)) {
		fprintf(stderr, " Couldn't get context for the device\n");
		dealloc_comm_struct(&user_comm,&user_param);
		return FAILURE;
	}

	MAIN_ALLOC(my_dest, struct pingpong_dest, user_param.num_of_qps, return_error);
	memset(my_dest, 0, sizeof(struct pingpong_dest)*user_param.num_of_qps);
	MAIN_ALLOC(rem_dest, struct pingpong_dest, user_param.num_of_qps, free_my_dest);
	memset(rem_dest, 0, sizeof(struct pingpong_dest)*user_param.num_of_qps);

	if (user_param.transport_type == IBV_TRANSPORT_IWARP)
		ctx.send_rcredit = 1;

	/* Allocating arrays needed for the test. */
	if (alloc_ctx(&ctx,&user_param)){
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
	if (send_set_up_connection(&ctx,&user_param,my_dest,&mcg_params,&user_comm)) {
		fprintf(stderr," Unable to set up socket connection\n");
		goto destroy_context;
	}

	/* Print basic test information. */
	ctx_print_test_info(&user_param);

	if (ctx.send_rcredit)
		ctx_alloc_credit(&ctx,&user_param,my_dest);

	for (i=0; i < user_param.num_of_qps; i++) {

		/* shaking hands and gather the other side info. */
		if (ctx_hand_shake(&user_comm,&my_dest[i],&rem_dest[i])) {
			fprintf(stderr,"Failed to exchange data between server and clients\n");
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

	/* If credit for available recieve buffers is necessary,
	 * the credit sending is done via RDMA WRITE ops and the ctx_hand_shake above
	 * is used to exchange the rkeys and buf addresses for the RDMA WRITEs
	 */
	if (ctx.send_rcredit)
		ctx_set_credit_wqes(&ctx,&user_param,rem_dest);

	/* Joining the Send side port the Mcast gid */
	if (user_param.use_mcg && (user_param.machine == CLIENT || user_param.duplex)) {

		memcpy(mcg_params.mgid.raw, rem_dest[0].gid.raw, 16);
		if (set_mcast_group(&ctx,&user_param,&mcg_params)) {
			fprintf(stderr," Unable to Join Sender to Mcast gid\n");
			goto destroy_context;
		}
		/*
		 * The next stall in code (50 ms sleep) is a work around for fixing the
		 * the bug this test had in Multicast for the past 1 year.
		 * It appears, that when a switch involved, it takes ~ 10 ms for the join
		 * request to propogate on the IB fabric, thus we need to wait for it.
		 * what happened before this fix was client reaching the post_send
		 * code segment in about 350 ns from here, and the switch(es) dropped
		 * the packet because join request wasn't finished.
		 */
		usleep(50000);
	}

	if (user_param.work_rdma_cm == OFF) {

		/* Prepare IB resources for rtr/rts. */
		if (ctx_connect(&ctx,rem_dest,&user_param,my_dest)) {
			fprintf(stderr," Unable to Connect the HCA's through the link\n");
			goto destroy_context;
		}
	}

	/* shaking hands and gather the other side info. */
	if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
		fprintf(stderr,"Failed to exchange data between server and clients\n");
		goto destroy_context;
	}

	if (user_param.connection_type == DC)
	{
		/* Set up connection one more time to send qpn properly for DC */
		if (set_up_connection(&ctx,&user_param,my_dest)) {
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

	if (user_param.use_event) {

		if (ibv_req_notify_cq(ctx.send_cq, 0)) {
			fprintf(stderr, " Couldn't request CQ notification\n");
			goto destroy_context;
		}

		if (ibv_req_notify_cq(ctx.recv_cq, 0)) {
			fprintf(stderr, " Couldn't request CQ notification\n");
			goto destroy_context;
		}
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
	if (user_param.test_method == RUN_ALL) {
		if (user_param.connection_type == UD)
			size_max_pow = (int)MSG_SZ_2_EXP(MTU_SIZE(user_param.curr_mtu)) + 1;
		else if (user_param.connection_type == SRD)
			size_max_pow = (int)MSG_SZ_2_EXP(user_param.size) + 1;

		for (i = 1; i < size_max_pow ; ++i) {

			user_param.size = (uint64_t)1 << i;

			if (user_param.machine == CLIENT || user_param.duplex)
				ctx_set_send_wqes(&ctx,&user_param,rem_dest);

			if (user_param.machine == SERVER || user_param.duplex) {
				if (ctx_set_recv_wqes(&ctx,&user_param)) {
					fprintf(stderr," Failed to post receive recv_wqes\n");
					goto free_mem;
				}
			}

			if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
				fprintf(stderr,"Failed to exchange data between server and clients\n");
				goto free_mem;
			}

			if (ctx.send_rcredit) {
				int j;
				for (j = 0; j < user_param.num_of_qps; j++)
					ctx.credit_buf[j] = 0;
			}

			if (user_param.duplex) {
				if(run_iter_bi(&ctx,&user_param)){
					error = 17;
					goto free_mem;
				}

			} else if (user_param.machine == CLIENT) {

				if(run_iter_bw(&ctx,&user_param)) {
					error = 17;
					goto free_mem;
				}

			} else	{

				if(run_iter_bw_server(&ctx,&user_param)) {
					error = 17;
					goto free_mem;
				}
			}

			print_report_bw(&user_param,&my_bw_rep);

			if (user_param.duplex && user_param.test_type != DURATION) {
				xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
				print_full_bw_report(&user_param, &my_bw_rep, &rem_bw_rep);
			}
			if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
				fprintf(stderr,"Failed to exchange data between server and clients\n");
				goto free_mem;
			}

			/* Check if last iteration ended well in UC/UD */
			if (user_param.check_alive_exited) {
				break;
			}
		}

	} else if (user_param.test_method == RUN_REGULAR) {

		if (user_param.machine == CLIENT || user_param.duplex)
			ctx_set_send_wqes(&ctx,&user_param,rem_dest);

		if (user_param.machine == SERVER || user_param.duplex) {
			if (ctx_set_recv_wqes(&ctx,&user_param)) {
				fprintf(stderr," Failed to post receive recv_wqes\n");
				goto free_mem;
			}
		}

		if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr,"Failed to exchange data between server and clients\n");
			goto free_mem;
		}

		if (user_param.duplex) {

			if(run_iter_bi(&ctx,&user_param)){
				error = 17;
				goto free_mem;
			}
		} else if (user_param.machine == CLIENT) {

			if(run_iter_bw(&ctx,&user_param)) {
				error = 17;
				goto free_mem;
			}

		} else if(run_iter_bw_server(&ctx,&user_param)) {
			error = 17;
			goto free_mem;
		}

		print_report_bw(&user_param,&my_bw_rep);

		if (user_param.duplex && user_param.test_type != DURATION) {
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

		if (user_param.machine == CLIENT)
			ctx_set_send_wqes(&ctx,&user_param,rem_dest);

		else if (user_param.machine == SERVER) {

			if (ctx_set_recv_wqes(&ctx,&user_param)) {
				fprintf(stderr," Failed to post receive recv_wqes\n");
				goto free_mem;
			}
		}

		if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr,"Failed to exchange data between server and clients\n");
			goto free_mem;
		}

		if (user_param.machine == CLIENT) {

			if(run_iter_bw_infinitely(&ctx,&user_param)) {
				fprintf(stderr," Error occurred while running infinitely! aborting ...\n");
				goto free_mem;
			}

		} else if (user_param.machine == SERVER) {

			if(run_iter_bw_infinitely_server(&ctx,&user_param)) {
				fprintf(stderr," Error occurred while running infinitely on server! aborting ...\n");
				goto free_mem;
			}
		}
	}

	if (user_param.output == FULL_VERBOSITY) {
		if (user_param.report_per_port)
			printf(RESULT_LINE_PER_PORT);
		else
			printf(RESULT_LINE);
	}

	if (ctx_close_connection(&user_comm,&my_dest[0],&rem_dest[0])) {
		fprintf(stderr," Failed to close connection between server and client\n");
		fprintf(stderr," Trying to close this side resources\n");
	}
	if(again && user_param.connection_type == UD)
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
	    //printf("user_param.connectionless %d  \n",user_param.connectionless);
            
	   #if 0
	    for (i=0; i < user_param.num_of_qps; i++) {
		modify_qp_to_init(&ctx, &user_param, i, user_param.num_of_qps);
	    }

	    if (user_param.work_rdma_cm == OFF) {

	    	/* Prepare IB resources for rtr/rts. */
	    	if (ctx_connect(&ctx,rem_dest,&user_param,my_dest)) {
	    		fprintf(stderr," Unable to Connect the HCA's through the link\n");
	    		goto destroy_context;
	    	}
	    }
	printf(" send_set_up_connection \n");
	if (send_set_up_connection(&ctx,&user_param,my_dest,&mcg_params,&user_comm)) {
		fprintf(stderr," Unable to set up socket connection\n");
		goto destroy_context;
	}
          #endif

#if 0
	    /* shaking hands and gather the other side info. */
	    if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
	    	fprintf(stderr,"Failed to exchange data between server and clients\n");
	    	goto destroy_context;
	    }
	    if (user_param.connection_type == DC)
	    {
	    	/* Set up connection one more time to send qpn properly for DC */
	    	if (set_up_connection(&ctx,&user_param,my_dest)) {
	    		fprintf(stderr," Unable to set up socket connection\n");
	    		goto destroy_context;
	    	}
	    }
#endif
	    /* Print this machine QP information */
	    for (i=0; i < user_param.num_of_qps; i++)
	    	ctx_print_pingpong_data(&my_dest[i],&user_comm);

	    user_comm.rdma_params->side = REMOTE;

	if (ctx.send_rcredit)
		ctx_alloc_credit(&ctx,&user_param,my_dest);
	    usleep(50000);
	    fprintf(stdout," all qps hand shake \n");
	    for (i=0; i < user_param.num_of_qps; i++) {

#if 1
	    	if (ctx_hand_shake(&user_comm,&my_dest[i],&rem_dest[i])) {
	    		fprintf(stderr," Failed to exchange data between server and clients\n");
	    		goto destroy_context;
	    	}
#endif
	    }

	if (user_param.work_rdma_cm == OFF) {
		if (ctx_check_gid_compatibility(&my_dest[0], &rem_dest[0])) {
			fprintf(stderr,"\n Found Incompatibility issue with GID types.\n");
			fprintf(stderr," Please Try to use a different IP version.\n\n");
			goto destroy_context;
		}
	}
	  	/* shaking hands and gather the other side info. */
	  	if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
					fprintf(stderr,"Failed to exchange data between server and clients\n");
							goto destroy_context;
								}
	/* If credit for available recieve buffers is necessary,
	 * the credit sending is done via RDMA WRITE ops and the ctx_hand_shake above
	 * is used to exchange the rkeys and buf addresses for the RDMA WRITEs
	 */
	if (ctx.send_rcredit)
		ctx_set_credit_wqes(&ctx,&user_param,rem_dest);
	  // change dest gidx attr.ah_attr.grh.dgid 
	  if (ud_ctx_connect(&ctx,rem_dest,&user_param,my_dest)) {
	    		fprintf(stderr," Unable to Connect the HCA's through the link\n");
	 		goto destroy_context;
	  }
	    for (i=0; i < user_param.num_of_qps; i++) {

	    	if (ctx_hand_shake(&user_comm,&my_dest[i],&rem_dest[i])) {
	    		fprintf(stderr," Failed to exchange data between server and clients\n");
	    		goto destroy_context;
	    	}
	    	ctx_print_pingpong_data(&rem_dest[i],&user_comm);
	    }
#if 1
	    if (user_param.use_event) {
		    printf("*************** use_event \n");

	    	if (ibv_req_notify_cq(ctx.send_cq, 0)) {
	    		fprintf(stderr, " Couldn't request CQ notification\n");
	    		goto destroy_context;
	    	}

	    	if (ibv_req_notify_cq(ctx.recv_cq, 0)) {
	    		fprintf(stderr, " Couldn't request CQ notification\n");
	    		goto destroy_context;
	    	}
	    }
#endif
	printf(" ***** start iter ***** \n");
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
	if (user_param.test_method == RUN_ALL) {
	        printf(" go to restart, iter all size \n");
		if (user_param.connection_type == UD)
			size_max_pow = (int)MSG_SZ_2_EXP(MTU_SIZE(user_param.curr_mtu)) + 1;
		else if (user_param.connection_type == SRD)
			size_max_pow = (int)MSG_SZ_2_EXP(user_param.size) + 1;

		for (i = 1; i < size_max_pow ; ++i) {

			user_param.size = (uint64_t)1 << i;

			if (user_param.machine == CLIENT || user_param.duplex)
				ctx_set_send_wqes(&ctx,&user_param,rem_dest);

			if (user_param.machine == SERVER || user_param.duplex) {
				if (ctx_set_recv_wqes(&ctx,&user_param)) {
					fprintf(stderr," Failed to post receive recv_wqes\n");
					goto free_mem;
				}
			}

#if 1
			if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
				fprintf(stderr,"Failed to exchange data between server and clients\n");
				goto free_mem;
			}
#endif
			if (ctx.send_rcredit) {
				int j;
				for (j = 0; j < user_param.num_of_qps; j++)
					ctx.credit_buf[j] = 0;
			}

			if (user_param.duplex) {
				if(run_iter_bi(&ctx,&user_param)){
					error = 17;
					goto free_mem;
				}

			} else if (user_param.machine == CLIENT) {

				if(run_iter_bw(&ctx,&user_param)) {
					error = 17;
					goto free_mem;
				}

			} else	{

				if(run_iter_bw_server(&ctx,&user_param)) {
					error = 17;
					goto free_mem;
				}
			}

			print_report_bw(&user_param,&my_bw_rep);

			if (user_param.duplex && user_param.test_type != DURATION) {
				xchg_bw_reports(&user_comm, &my_bw_rep,&rem_bw_rep,atof(user_param.rem_version));
				print_full_bw_report(&user_param, &my_bw_rep, &rem_bw_rep);
			}
#if 1
			if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
				fprintf(stderr,"Failed to exchange data between server and clients\n");
				goto free_mem;
			}
#endif
			/* Check if last iteration ended well in UC/UD */
			if (user_param.check_alive_exited) {
				break;
			}
		}

	} else if (user_param.test_method == RUN_REGULAR) {

		if (user_param.machine == CLIENT || user_param.duplex)
			ctx_set_send_wqes(&ctx,&user_param,rem_dest);

		if (user_param.machine == SERVER || user_param.duplex) {
			if (ctx_set_recv_wqes(&ctx,&user_param)) {
				fprintf(stderr," Failed to post receive recv_wqes\n");
				goto free_mem;
			}
		}

#if 1
		if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr,"Failed to exchange data between server and clients\n");
			goto free_mem;
		}

#endif
		if (user_param.duplex) {

			if(run_iter_bi(&ctx,&user_param)){
				error = 17;
				goto free_mem;
			}
		} else if (user_param.machine == CLIENT) {

			if(run_iter_bw(&ctx,&user_param)) {
				error = 17;
				goto free_mem;
			}

		} else if(run_iter_bw_server(&ctx,&user_param)) {
			error = 17;
			goto free_mem;
		}

		print_report_bw(&user_param,&my_bw_rep);

		if (user_param.duplex && user_param.test_type != DURATION) {
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

		if (user_param.machine == CLIENT)
			ctx_set_send_wqes(&ctx,&user_param,rem_dest);

		else if (user_param.machine == SERVER) {

			if (ctx_set_recv_wqes(&ctx,&user_param)) {
				fprintf(stderr," Failed to post receive recv_wqes\n");
				goto free_mem;
			}
		}

#if 1
		if (ctx_hand_shake(&user_comm,&my_dest[0],&rem_dest[0])) {
			fprintf(stderr,"Failed to exchange data between server and clients\n");
			goto free_mem;
		}

#endif
		if (user_param.machine == CLIENT) {

			if(run_iter_bw_infinitely(&ctx,&user_param)) {
				fprintf(stderr," Error occurred while running infinitely! aborting ...\n");
				goto free_mem;
			}

		} else if (user_param.machine == SERVER) {

			if(run_iter_bw_infinitely_server(&ctx,&user_param)) {
				fprintf(stderr," Error occurred while running infinitely on server! aborting ...\n");
				goto free_mem;
			}
		}
	}

	     if (user_param.output == FULL_VERBOSITY) {
	     	if (user_param.report_per_port)
	     		printf(RESULT_LINE_PER_PORT);
	     	else
	     		printf(RESULT_LINE);
	     }
	     printf("start to close connection in restart  stage \n ");
	     //if (restart_stage_ctx_close_connection(&user_comm,&my_dest[0],&rem_dest[0])) {
	     if (ctx_close_connection(&user_comm,&my_dest[0],&rem_dest[0])) {
	     	fprintf(stderr," Failed to close connection between server and client\n");
	     	fprintf(stderr," Trying to close this side resources\n");
	     }
	}
	/* Destroy all test resources, including Mcast if exists */
	if (send_destroy_ctx(&ctx,&user_param,&mcg_params)) {
		fprintf(stderr,"Couldn't destroy all SEND resources\n");
		goto destroy_cm_context;
	}

	if (user_param.work_rdma_cm == ON) {
		user_comm.rdma_params->work_rdma_cm = OFF;

		if (destroy_ctx(user_comm.rdma_ctx,user_comm.rdma_params)) {
			fprintf(stderr,"Failed to destroy resources\n");
			goto free_mem;
		}
	}

	free(my_dest);
	free(rem_dest);

	if (!user_param.is_bw_limit_passed && (user_param.is_limit_bw == ON ) ) {
		fprintf(stderr,"Error: BW result is below bw limit\n");
		return FAILURE;
	}

	if (!user_param.is_msgrate_limit_passed && (user_param.is_limit_bw == ON )) {
		fprintf(stderr,"Error: Msg rate  is below msg_rate limit\n");
		return FAILURE;
	}

	return SUCCESS;

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
	return error;
}
