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
                        printf("proto icmp\t");
     src = (u8*) &ip4->src_address;
     dst = (u8*) &ip4->dst_address;
     clib_memcpy_fast(temp, dst, 4);
     clib_memcpy_fast(dst,src, 4);
     clib_memcpy_fast(src,temp, 4);
}
static void print_node(vlib_main_t *vm, u32 next_index)
{
    vlib_node_t *node = vlib_get_node (vm, next_index);
    printf("next node name1 %s \t",node->name);
    node =  vlib_get_node_by_name (vm, (u8 *) "ck_sample_next");
    printf("next node name2 %s \t",node->name);
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
                        printf(" %s \t",__func__);
			print_node(vm,next_index);
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
			ip4_header_t *ip4  = vlib_buffer_get_current(b0);
                        printf_ipv4_header(ip4); 
			set_ipv4_header(ip4);
                        vlib_validate_buffer_enqueue_x1(vm, node, next_index,
                                to_next, n_left_to_next, bi0, next0);
                }

                vlib_put_next_frame(vm, node, next_index, n_left_to_next);
        }

        return frame->n_vectors;
}
#else
// two packet

static uword ck_sample_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node,
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
		while (n_left_from >= 4 && n_left_to_next >= 2){
		printf("**********process two packets **********\n");
		 /* next0和next1指明包的下一个节点索引值 */
		u32 next0 = next_index;
		u32 next1 = next_index;
	        u32 bi0, bi1;
		vlib_buffer_t *b0, *b1;
	        to_next[0] = bi0 = from[0];
		to_next[1] = bi1 = from[1];
		from += 2;
		to_next += 2;
		n_left_to_next -= 2;
	 	n_left_from -= 2;
	
	        b0 = vlib_get_buffer (vm, bi0);
	        b1 = vlib_get_buffer (vm, bi1);
	       ip4_header_t *ip4  = vlib_buffer_get_current(b0);
               printf(" %s \t",__func__);
	       print_node(vm,next_index);
               printf_ipv4_header(ip4); 
	       ip4  = vlib_buffer_get_current(b1);
               printf_ipv4_header(ip4); 
	       /* verify speculative enqueues, maybe switch current next frame */
	       /* 
	        *         next_index:默认的下一结点的index
	        *         next0:实际的下一个结点的index
	        *         next0 == next_index则不需要做特别的处理，报文会自动进入下一个节点
	        *         next0 != next_index则需要对该数据包做调整，从之前next_index对应
	        *         的frame中删除，添加到next0对应的frame中
	        *
	        *         next1的判断和next0一样
	        **/
		vlib_validate_buffer_enqueue_x2 (vm, node, next_index, to_next, n_left_to_next, bi0, bi1, next0, next1);

		}
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
			print_node(vm,next_index);
                        printf_ipv4_header(ip4); 
			set_ipv4_header(ip4);
                        vlib_validate_buffer_enqueue_x1(vm, node, next_index,
                                to_next, n_left_to_next, bi0, next0);
                }

                vlib_put_next_frame(vm, node, next_index, n_left_to_next);
        }

        return frame->n_vectors;
}
#endif
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
