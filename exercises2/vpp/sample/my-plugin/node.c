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
#include <vnet/pg/pg.h>
#include <vnet/ethernet/ethernet.h>
#include <vppinfra/error.h>
#include <plugins/my-plugin/my.h>

typedef struct
{
  u32 next_index;
  u32 sw_if_index;
  u8 new_src_mac[6];
  u8 new_dst_mac[6];
} my_trace_t;


/* packet trace format function */
static u8 *
format_my_trace (u8 * s, va_list * args)
{
  CLIB_UNUSED (vlib_main_t * vm) = va_arg (*args, vlib_main_t *);
  CLIB_UNUSED (vlib_node_t * node) = va_arg (*args, vlib_node_t *);
  my_trace_t *t = va_arg (*args, my_trace_t *);

  s = format (s, "SAMPLE: sw_if_index %d, next index %d\n",
	      t->sw_if_index, t->next_index);
  s = format (s, "  new src %U -> new dst %U",
	      format_mac_address, t->new_src_mac,
	      format_mac_address, t->new_dst_mac);

  return s;
}

extern vlib_node_registration_t my_node;

#define foreach_my_error \
_(SWAPPED, "Mac swap packets processed")

typedef enum
{
#define _(sym,str) MY_ERROR_##sym,
  foreach_my_error
#undef _
    MY_N_ERROR,
} my_error_t;

static char *my_error_strings[] = {
#define _(sym,string) string,
  foreach_my_error
#undef _
};

typedef enum
{
  MY_NEXT_INTERFACE_OUTPUT,
  MY_N_NEXT,
} my_next_t;

/*
 * Simple dual/single loop version, default version which will compile
 * everywhere.
 *
 * Node costs 30 clocks/pkt at a vector size of 51
 */

VLIB_NODE_FN (my_node) (vlib_main_t * vm, vlib_node_runtime_t * node,
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

	    //vlib_node_increment_counter (vm, learn_node_internal.index, LEARN_ERROR_PROCESSED, pkts_processed);
#endif
	return frame->n_vectors;
}

/*
 * This version computes all of the buffer pointers in
 * one motion, uses a fully pipelined loop model, and
 * traces the entire frame in one motion.
 *
 * It's performance-competative with other coding paradigms,
 * and it's the simplest way to write performant vpp code
 */



/* *INDENT-OFF* */
VLIB_REGISTER_NODE (my_node) =
{
  .name = "my",
  .vector_size = sizeof (u32),
  .format_trace = format_my_trace,
  .type = VLIB_NODE_TYPE_INTERNAL,
  //.type = VLIB_NODE_TYPE_INPUT,
  .n_errors = ARRAY_LEN(my_error_strings),
  .error_strings = my_error_strings,

  .n_next_nodes = MY_N_NEXT,

  /* edit / add dispositions here */
  .next_nodes = {
    [MY_NEXT_INTERFACE_OUTPUT] = "interface-output",
  },
};
/* *INDENT-ON* */

/*
 * fd.io coding-style-patch-verification: ON
 *
 * Local Variables:
 * eval: (c-set-style "gnu")
 * End:
 */
