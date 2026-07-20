#include <vlib/vlib.h>
#include <vnet/vnet.h>
#include <vnet/pg/pg.h>
#include <vnet/ethernet/ethernet.h>
#include <vppinfra/error.h>
#include <packet_dump/pkt_dump.h>

typedef enum
{
  CK_SAMPLE_NEXT_IP4,
  CK_SAMPLE_DROP,
  CK_SAMPLE_NEXT_N,
} ck_sample_next_t;

typedef struct
{
  u32 next_index;
  u32 sw_if_index;
  u8 new_src_mac[6];
  u8 new_dst_mac[6];
} ck_sample_trace_t;

#define foreach_ck_sample_error \
_(SHOWED, "show packets processed")

typedef enum
{
#define _(sym,str) SAMPLE_ERROR_##sym,
  foreach_ck_sample_error
#undef _
    SAMPLE_N_ERROR,
} ck_ssample_error_t;


static char *ck_sample_error_strings[] = {
#define _(sym, str) str,
        foreach_ck_sample_error
#undef _
};

extern vlib_node_registration_t ck_sample_node;

static u8 *
format_ck_sample_trace (u8 * s, va_list * args)
{
        s = format(s, "To Do!\n");
        return s;
}


/*
 *ping.c
 * print_ip46_icmp_reply (vlib_main_t * vm, u32 bi0, int is_ip6)
 * refer to ip46_get_icmp_id_and_seq
 * icmp46_header_t *icmp0, *icmp1;
 *  ip0 = vlib_buffer_get_current (p0);
 */

static void printf_ipv4_header(ip4_header_t *ip4)
{
     //int l4_offset;
     //l4_offset = ip4_header_bytes (ip4);
     //void *paddr;
     //void *format_addr_func;
     //u8 *src, *dts;
     u8 *src;
     if (IP_PROTOCOL_ICMP == ip4->protocol)
                        printf("proto icmp\t");
     src = (u8*) &ip4->src_address;
     printf(" src: %d.%d.%d.%d,\t", src[0],src[1],src[2],src[3]);
     //paddr = (void *) &ip4->src_address;
     //format_addr_func = (void *) format_ip4_address;
     //vlib_cli_output (vm, "Source address: %U ", format_addr_func, paddr);
     printf("\n");
}
static void set_ipv4_header(ip4_header_t *ip4)
{
     //int l4_offset;
     //l4_offset = ip4_header_bytes (ip4);
     //void *paddr;
     //void *format_addr_func;
     u8 *src, *dst;
     u8 temp[4];
     if (IP_PROTOCOL_ICMP == ip4->protocol)
     {
                        //printf("proto icmp\t");
     }
     src = (u8*) &ip4->src_address;
     dst = (u8*) &ip4->dst_address;
     clib_memcpy_fast(temp, dst, 4);
     clib_memcpy_fast(dst,src, 4);
     clib_memcpy_fast(src,temp, 4);
}
static void print_node(vlib_main_t *vm, u32 next_index)
{
#if 1
    vlib_node_t *node = vlib_get_node (vm, next_index);
    printf("\n next node name1 %s \t",node->name);
    node =  vlib_get_node_by_name (vm, (u8 *) "ck_sample_next");
    printf("next node name2 %s \n",node->name);
#endif
}
static uword ck_sample_next_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node,
        vlib_frame_t * frame)
{
        u32 n_left_from, *from, *to_next;
        u32 next_index;
        from        = vlib_frame_vector_args(frame);
        n_left_from = frame->n_vectors;
        next_index  = node->cached_next_index;

        while(n_left_from > 0){
                u32 n_left_to_next;
                vlib_get_next_frame(vm, node, next_index, to_next, n_left_to_next);

                while(n_left_from > 0 && n_left_to_next > 0){
                        vlib_buffer_t  *b0;
                        u32             bi0, next0 = 0;

                        bi0 = to_next[0] = from[0];
                        from           += 1;
                        to_next        += 1;
                        n_left_to_next -= 1;
                        n_left_from    -= 1;

                        b0 = vlib_get_buffer(vm, bi0);
			ip4_header_t *ip4  = vlib_buffer_get_current(b0);
			print_node(vm, node->cached_next_index);
			print_node(vm,node->node_index);
                        printf(" %s \t",__func__);
                        printf_ipv4_header(ip4); 
			set_ipv4_header(ip4);
                        vlib_validate_buffer_enqueue_x1(vm, node, next_index,
                                to_next, n_left_to_next, bi0, next0);
                }

                vlib_put_next_frame(vm, node, next_index, n_left_to_next);
        }

        return frame->n_vectors;
}
static uword ck_sample_skip_next_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node,
        vlib_frame_t * frame)
{
        u32 n_left_from, *from, *to_next;
        u32 next_index;
        from        = vlib_frame_vector_args(frame);
        n_left_from = frame->n_vectors;
        next_index  = node->cached_next_index;

        while(n_left_from > 0){
                u32 n_left_to_next;
                vlib_get_next_frame(vm, node, next_index, to_next, n_left_to_next);

                while(n_left_from > 0 && n_left_to_next > 0){
                        vlib_buffer_t  *b0;
                        u32             bi0, next0 = 0;

                        bi0 = to_next[0] = from[0];
                        from           += 1;
                        to_next        += 1;
                        n_left_to_next -= 1;
                        n_left_from    -= 1;

                        b0 = vlib_get_buffer(vm, bi0);
			ip4_header_t *ip4  = vlib_buffer_get_current(b0);
                        printf(" %s \t",__func__);
                        printf_ipv4_header(ip4); 
			set_ipv4_header(ip4);
                        vlib_validate_buffer_enqueue_x1(vm, node, next_index,
                                to_next, n_left_to_next, bi0, next0);
                }

                vlib_put_next_frame(vm, node, next_index, n_left_to_next);
        }

        return frame->n_vectors;
}
#if 0
static uword ck_sample_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node,
        vlib_frame_t * frame)
{
        u32 n_left_from, *from, *to_next;
        ck_sample_next_t     next_index;

        from        = vlib_frame_vector_args(frame);
        n_left_from = frame->n_vectors;
        next_index  = node->cached_next_index;

        while(n_left_from > 0){
                u32 n_left_to_next;
                vlib_get_next_frame(vm, node, next_index, to_next, n_left_to_next);

                while(n_left_from > 0 && n_left_to_next > 0){
                        vlib_buffer_t  *b0;
                        u32             bi0, next0 = 0;

                        bi0 = to_next[0] = from[0];
                        from           += 1;
                        to_next        += 1;
                        n_left_to_next -= 1;
                        n_left_from    -= 1;

                        b0 = vlib_get_buffer(vm, bi0);
                        
			void *en0 = vlib_buffer_get_current(b0);
                        int i = 0;
                        for (i = 0; i < 34; i++)
                        {
                                printf("%02x ", *(u8*)(en0+i));
                        }
                        printf("\n");
                        vlib_validate_buffer_enqueue_x1(vm, node, next_index,
                                to_next, n_left_to_next, bi0, next0);
                }

                vlib_put_next_frame(vm, node, next_index, n_left_to_next);
        }

        return frame->n_vectors;
}
#else
always_inline vlib_next_frame_t *
vlib_node_runtime_get_next_frame_test (vlib_main_t * vm, vlib_node_t *cur_node, u32 next_index)
{
  vlib_node_main_t *nm = &vm->node_main;
  vlib_next_frame_t *nf;
  vlib_node_runtime_t * run;
 #if 0
  vlib_node_t *cur_node;

  cur_node = vec_elt (nm->nodes,node_index);
  if(!cur_node){
        printf("in node main , next node is null \n ");
	return NULL;
  }
 #endif
  printf("\n in node main , cur node name %s \t", cur_node->name);
  run = vec_elt_at_index (nm->nodes_by_type[cur_node->type], cur_node->runtime_index);
  ASSERT (next_index < run->n_next_nodes);
  nf = vec_elt_at_index (nm->next_frames, run->next_frame_index + next_index);
  if (CLIB_DEBUG > 0)
  {
        vlib_node_t *node, *next;
        node = vec_elt (nm->nodes, run->node_index);
        if(!node){
              printf("in runtime ,  node is null \n ");
              return NULL;
        }
        next = vec_elt (nm->nodes, node->next_nodes[next_index]);
        if(!next){
              printf("in runtime ,  next node is null \n ");
              return NULL;
        }
        printf("in runtime , node name %s and next node name %s , next frame addr %p \n", node->name, next->name,nf);
        ASSERT (nf->node_runtime_index == next->runtime_index);
  }

  return nf;
}

static uword ck_sample_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node,
        vlib_frame_t * frame)
{
        u32 n_left_from, *from, *to_next;
        u32 next_index;
        u32 goto_index;
	vlib_frame_t * skip_frame;
        from        = vlib_frame_vector_args(frame);
        n_left_from = frame->n_vectors;
	next_index  = node->cached_next_index;
	vlib_node_t *next_node =  vlib_get_node_by_name (vm, (u8 *) "ck_sample_skip_next");
        goto_index  = next_node->index;
#if 1
       vlib_node_main_t *nm = &vm->node_main;
       vlib_node_t * cur_node = vec_elt (nm->nodes, node->node_index);
       vlib_node_runtime_get_next_frame_test (vm, cur_node, node->cached_next_index);
       vlib_node_runtime_get_next_frame_test (vm, next_node, node->cached_next_index);
       // will cause  ASSERT (next_index < run->n_next_nodes);
       //vlib_node_runtime_get_next_frame_test (vm,goto_index);
#endif
        while(n_left_from > 0){
                u32 n_left_to_next;
                vlib_get_next_frame(vm, node, next_index, to_next, n_left_to_next);
                while(n_left_from > 0 && n_left_to_next > 0){
                        vlib_buffer_t  *b0;
                        //u32             bi0, next0 = 0;
                        u32             bi0;

                        bi0 = to_next[0] = from[0];
                        from           += 1;
                        to_next        += 1;
                        n_left_to_next -= 1;
                        n_left_from    -= 1;

#if 0
                        b0 = vlib_get_buffer(vm, bi0);
			ip4_header_t *ip4  = vlib_buffer_get_current(b0);
                        printf(" %s \t",__func__);
			print_node(vm,next_index);
                        printf_ipv4_header(ip4); 
			set_ipv4_header(ip4);
                        vlib_validate_buffer_enqueue_x1(vm, node, next_index,
                                to_next, n_left_to_next, bi0, next0);
#else

                        u32  *go_to_next;
                        b0 = vlib_get_buffer(vm, bi0);
			ip4_header_t *ip4  = vlib_buffer_get_current(b0);
                        printf(" %s \t",__func__);
                        printf_ipv4_header(ip4); 
			set_ipv4_header(ip4);
			skip_frame = vlib_get_frame_to_node (vm, goto_index);
			go_to_next = vlib_frame_vector_args (skip_frame);
			go_to_next[0] = bi0;
			skip_frame->n_vectors = 1;
			vlib_put_frame_to_node (vm, goto_index, skip_frame);
#endif
                }

                //vlib_put_next_frame(vm, node, next_index, n_left_to_next);
        }
        return frame->n_vectors;
}
#endif
#if 1
VLIB_REGISTER_NODE (ck_sample_node) = {
        .name		= "ck_sample",
        .function       = ck_sample_node_fn,
        .vector_size    = sizeof(u32),
        .format_trace   = format_ck_sample_trace,
        .type           = VLIB_NODE_TYPE_INTERNAL,
        .n_errors       = ARRAY_LEN(ck_sample_error_strings),
        .error_strings  = ck_sample_error_strings,
        .n_next_nodes   = CK_SAMPLE_NEXT_N,
        .next_nodes     = {
                [CK_SAMPLE_NEXT_IP4]    = "ck_sample_next",
                //[CK_SAMPLE_NEXT_IP4]    = "ip4-lookup",
                [CK_SAMPLE_DROP]        = "error-drop",
        },
};
VLIB_REGISTER_NODE (ck_sample_next_node) = {
        .name		= "ck_sample_next",
        .function       = ck_sample_next_node_fn,
        .vector_size    = sizeof(u32),
        .format_trace   = format_ck_sample_trace,
        .type           = VLIB_NODE_TYPE_INTERNAL,
        .n_errors       = ARRAY_LEN(ck_sample_error_strings),
        .error_strings  = ck_sample_error_strings,
        .n_next_nodes   = CK_SAMPLE_NEXT_N,
        .next_nodes     = {
                //[CK_SAMPLE_NEXT_IP4]    = "ip4-lookup",
                [CK_SAMPLE_NEXT_IP4]    = "ck_sample_skip_next",
                [CK_SAMPLE_DROP]        = "error-drop",
        },
};
VLIB_REGISTER_NODE (ck_sample_skip_next_node) = {
        .name		= "ck_sample_skip_next",
        .function       = ck_sample_skip_next_node_fn,
        .vector_size    = sizeof(u32),
        .format_trace   = format_ck_sample_trace,
        .type           = VLIB_NODE_TYPE_INTERNAL,
        .n_errors       = ARRAY_LEN(ck_sample_error_strings),
        .error_strings  = ck_sample_error_strings,
        .n_next_nodes   = CK_SAMPLE_NEXT_N,
        .next_nodes     = {
                [CK_SAMPLE_NEXT_IP4]    = "ip4-lookup",
                [CK_SAMPLE_DROP]        = "error-drop",
        },
};
#else
VLIB_REGISTER_NODE (ck_sample_node) = {
        .name		= "ck_sample",
        .function       = ck_sample_node_fn,
        .vector_size    = sizeof(u32),
        .format_trace   = format_ck_sample_trace,
        .type           = VLIB_NODE_TYPE_INTERNAL,
        .n_errors       = ARRAY_LEN(ck_sample_error_strings),
        .error_strings  = ck_sample_error_strings,
        .n_next_nodes   = CK_SAMPLE_NEXT_N,
        .next_nodes     = {
                [CK_SAMPLE_NEXT_IP4]    = "ip4-icmp-echo-request",
                //[CK_SAMPLE_NEXT_IP4]    = "ip4-lookup",
                //[CK_SAMPLE_DROP]        = "error-drop",
		//[0] = "ip4-load-balance",
        },
};
static clib_error_t *
icmp_init (vlib_main_t * vm)
{
    ip4_icmp_register_type (vm, ICMP4_echo_request, ck_sample_node.index);
    return 0;
}
VLIB_INIT_FUNCTION (icmp_init);
#endif
