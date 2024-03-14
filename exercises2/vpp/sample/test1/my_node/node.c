/*
 * Copyright (c) 2015 Cisco and/or its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <vlib/vlib.h>

#include <vnet/vnet.h>
//#include <vnet/plugin/plugin.h>
#include <vnet/pg/pg.h>
#include <vnet/ethernet/ethernet.h>
#include <vppinfra/error.h>
#include <plugins/my_node/my_ip.h>

//#define NSTAGES 4
//#include <vnet/pipeline.h>
#if 0
VLIB_PLUGIN_REGISTER () = {
	  .version = MY_IP_BUILD_VER,
	  .description = "my_ip",
};
#endif
typedef struct
{
  u32 next_index;
  u32 sw_if_index;
  u8 new_src_mac[6];
  u8 new_dst_mac[6];
} my_ip_trace_t;


/* packet trace format function */
static u8 *
format_my_ip_trace (u8 * s, va_list * args)
{
  CLIB_UNUSED (vlib_main_t * vm) = va_arg (*args, vlib_main_t *);
  CLIB_UNUSED (vlib_node_t * node) = va_arg (*args, vlib_node_t *);
  my_ip_trace_t *t = va_arg (*args, my_ip_trace_t *);

  s = format (s, "SAMPLE: sw_if_index %d, next index %d\n",
	      t->sw_if_index, t->next_index);
  s = format (s, "  new src %U -> new dst %U",
	      format_mac_address, t->new_src_mac,
	      format_mac_address, t->new_dst_mac);

  return s;
}

extern vlib_node_registration_t my_ip_node;

#define foreach_my_ip_error \
_(SWAPPED, "Mac swap packets processed")

typedef enum
{
#define _(sym,str) MY_ERROR_##sym,
  foreach_my_ip_error
#undef _
    MY_N_ERROR,
} my_ip_error_t;

static char *my_ip_error_strings[] = {
#define _(sym,string) string,
  foreach_my_ip_error
#undef _
};

typedef enum
{
  MY_NEXT_INTERFACE_OUTPUT,
  MY_N_NEXT,
} my_ip_next_t;

typedef enum
{
  HANDOFFDEMO_NEXT_DROP,
  HANDOFFDEMO_N_NEXT,
} handoffdemo_next_t;

/*
 * Simple dual/single loop version, default version which will compile
 * everywhere.
 *
 * Node costs 30 clocks/pkt at a vector size of 51
 */


#if 1
static uword
my_ip_node_fn (vlib_main_t * vm, vlib_node_runtime_t * node, vlib_frame_t * frame)
{
#if 0
	   u32  *from;
	   u16 nexts[VLIB_FRAME_SIZE];// *next;
           from = vlib_frame_vector_args (frame);
	   vlib_buffer_enqueue_to_next (vm, node, from, (u16 *) nexts,
			    			       frame->n_vectors);
	   vlib_cli_output (vm, "my ip node run \n");
#else
	   //dispatch_pipeline (vm, node, frame);
  u32 n_left_from, *from;
  u16 nexts[VLIB_FRAME_SIZE], *next;
  u32 error0 = 0;
  //u32 n_enq;
  vlib_buffer_t *bufs[VLIB_FRAME_SIZE], **b;
      from = vlib_frame_vector_args (frame);
      n_left_from = frame->n_vectors;

      vlib_get_buffers (vm, from, bufs, n_left_from);
      next = nexts;
      b = bufs;

      while (n_left_from > 0)
	{
	  //f (is_trace && (b[0]->flags & VLIB_BUFFER_IS_TRACED))
	   // {
	   //   handoffdemo_trace_t *t = vlib_add_trace (vm, node, b[0],
	   //     				       sizeof (*t));
	   //   t->current_thread = vm->thread_index;
	   // }

	  next[0] = HANDOFFDEMO_NEXT_DROP;
	  b[0]->error = error0;
	  next++;
	  b++;
	  n_left_from--;
	}

      vlib_cli_output (vm, "my ip node run \n");
      vlib_buffer_enqueue_to_next (vm, node, from, (u16 *) nexts,
				   frame->n_vectors);
#endif
	return frame->n_vectors;
}
#else
VLIB_NODE_FN (my_ip_node) (vlib_main_t * vm, vlib_node_runtime_t * node,
			    vlib_frame_t * frame)
{
#if 0
	 vlib_cli_output (vm, "my node run \n");
#else
	   u32  *from;
	   //u32 n_left_from, *from;
	   //u32 pkts_processed = 0;
	   //vlib_buffer_t *bufs[VLIB_FRAME_SIZE], **b;
	   u16 nexts[VLIB_FRAME_SIZE];// *next;

           from = vlib_frame_vector_args (frame);
           //n_left_from = frame->n_vectors;

           //vlib_get_buffers (vm, from, bufs, n_left_from);
           //b = bufs;
	   //next = nexts;
#if 0
	     while (n_left_from > 0)
		         {
				       /*
					*       ip4_header_t *ip0 = 0;
					*             ip0 = vlib_buffer_get_current (b[0]);
					*                   */

				       next[0] = LEARN_NEXT_INTERFACE_OUTPUT;

				             b += 1;
					           next += 1;
						         n_left_from -= 1;
							       pkts_processed += 1;
							           }
#endif
	    vlib_buffer_enqueue_to_next (vm, node, from, (u16 *) nexts,
			    			       frame->n_vectors);

	    vlib_cli_output (vm, "my node run \n");
	    //vlib_node_increment_counter (vm, learn_node_internal.index, LEARN_ERROR_PROCESSED, pkts_processed);
#endif
	return frame->n_vectors;
}
#endif
/*
 * This version computes all of the buffer pointers in
 * one motion, uses a fully pipelined loop model, and
 * traces the entire frame in one motion.
 *
 * It's performance-competative with other coding paradigms,
 * and it's the simplest way to write performant vpp code
 */



#if 1
/* *INDENT-OFF* */
VLIB_REGISTER_NODE (my_ip_node) =
{
  .name = "my_ip",
  .function = my_ip_node_fn,
  .vector_size = sizeof (u32),
  .format_trace = format_my_ip_trace,
  .type = VLIB_NODE_TYPE_INTERNAL,
  //.type = VLIB_NODE_TYPE_INPUT,
  .n_errors = ARRAY_LEN(my_ip_error_strings),
  .error_strings = my_ip_error_strings,

  .n_next_nodes = MY_N_NEXT,

  /* edit / add dispositions here */
  .next_nodes = {
    //[MY_NEXT_INTERFACE_OUTPUT] = "interface-output",
    [MY_NEXT_INTERFACE_OUTPUT] = "ip4-lookup",
  },
};
#endif
#if 0
#define TEST_INTERFACE_INDEX 0
static clib_error_t *
my_ip_node_init (vlib_main_t * vm)
{
	//
    //my_ip_main_t * sm = &my_ip_main;
    //sm->frame_queue_index = vlib_frame_queue_main_init(my_ip_node.index, 16);
    //vnet_feature_enable_disable ("device-input", "my_ip", TEST_INTERFACE_INDEX, 1, NULL, 0);
    return 0;
}
VLIB_INIT_FUNCTION (my_ip_node_init);
#endif
#if 0
VLIB_PLUGIN_REGISTER () = {
	  .version = MY_IP_BUILD_VER,
	  .description = "my_ip",
};
#endif
#if 0
// lcp_arp_host_arp_feat
VNET_FEATURE_INIT (my_ip_node, static) =
{
   
  //.arc_name = "device-input",
  .arc_name = "arp",
  .node_name = "my_ip",
  .runs_before = VNET_FEATURES ("arp-reply"),
  //.runs_before = VNET_FEATURES ("ip4-lookup"),
  //.start_nodes = VNET_FEATURES ("ip4-input"),  
  //.node_name = "interface-output",
  //.runs_before = VNET_FEATURES ("ethernet-input"),
};
#elif 0
VNET_FEATURE_INIT (my_ip_node, static) =
{
   
  .arc_name = "device-input",
  .node_name = "my_ip",
  .runs_before = VNET_FEATURES ("ethernet-input"),
};
#else
//lcp_xc_ip4_ucast_node
VNET_FEATURE_INIT (my_ip_node, static) =
{
   
  //.arc_name = "device-input",
  .arc_name = "ip4-unicast",
  .node_name = "my_ip",
  .runs_before = VNET_FEATURES ("ip4-lookup"),
  //.start_nodes = VNET_FEATURES ("ip4-input"),  
  //.node_name = "interface-output",
  //.runs_before = VNET_FEATURES ("ethernet-input"),
};
#endif
/* *INDENT-ON* */

/*
 * fd.io coding-style-patch-verification: ON
 *
 * Local Variables:
 * eval: (c-set-style "gnu")
 * End:
 */
