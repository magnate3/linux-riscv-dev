#include <vlib/vlib.h>
#include <vnet/vnet.h>
#include <vnet/pg/pg.h>
#include <vnet/ethernet/ethernet.h>
#include <vppinfra/error.h>
#include <packet_dump/pkt_dump.h>
#define TEST_FRAME 1
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


#if !TEST_FRAME
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
     src = (u8*) &ip4->src_address;
     if (IP_PROTOCOL_ICMP == ip4->protocol)
     {
          printf("proto icmp\t");
          printf(" src: %d.%d.%d.%d,\t", src[0],src[1],src[2],src[3]);
          printf("\n");
     }
     //paddr = (void *) &ip4->src_address;
     //format_addr_func = (void *) format_ip4_address;
     //vlib_cli_output (vm, "Source address: %U ", format_addr_func, paddr);
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
#endif
typedef struct
{
    int scalar;
    u32 vector[0];
} my_frame_t;
static u8 * format_my_node_frame (u8 * s, vlib_frame_t *f)
{
    //vlib_frame_t *f = va_arg (*va, vlib_frame_t *);
    u32 *scalar  = vlib_frame_scalar_args (f);
    int i;
    u32 * vectors = vlib_frame_vector_args (f);
    s = format (s, "scalar %u, vector { ", *scalar);
    for (i = 0; i < f->n_vectors; i++)
	      s = format (s, "%u, ", vectors[i]);
   s = format (s, " }");
   return s;
}
#if 0
static uword my_func (vlib_main_t * vm, vlib_node_runtime_t * rt, vlib_frame_t * f)
{
    vlib_node_t *node;
    my_frame_t *my;
    u32  n_left = 0;
    u32  *vectors;
    u32 *from        = vlib_frame_vector_args(f);
    //u32 i, n_left = 0;
    u32 n_left_from = f->n_vectors;
    //u32 *to_next;
    //u32 n_left_to_next;
    u32    next_index;
    vlib_frame_t *next;
    next_index  = rt->cached_next_index;
    node = vlib_get_node (vm, rt->node_index);
    vlib_cli_output (vm, "%v: call frame %p %U", node->name, f, format_my_node_frame, f);
    while(n_left_from > 0){
        //if (rt->n_next_nodes > 0) 
	do
	{
            //vlib_frame_t *next = vlib_get_next_frame (vm, rt, /* next index */ 0,to_next, n_left_to_next);
            next = vlib_get_next_frame_internal(vm, rt, /* next index */ 0,0);
            n_left = VLIB_FRAME_SIZE - next->n_vectors;
            my = vlib_frame_scalar_args (next) + (next->n_vectors) * sizeof ((vectors)[0]);
	    // scalar can not be changed
            //my->scalar = serial++;
        }while(0);    
	     vectors = my->vector;
             while(n_left_from > 0 && n_left > 0){
		   u32             bi0;
                   bi0 =  *from;
		   *vectors = *from;
                   ASSERT (n_left > 0);
		   ++ from;
		   ++ vectors;
                   n_left -= 1;
                   n_left_from    -= 1;
                   vlib_validate_buffer_enqueue_x1(vm, rt, /*next_index*/next_index, vectors, n_left, bi0,  /*next_index*/0);
                }

        vlib_put_next_frame (vm, rt, /* next index */ 0, n_left);
        //vlib_cli_output (vm, "%v: return frame %p", node->name, f);
    }
    return f->n_vectors;
}
#else
static uword my_func (vlib_main_t * vm, vlib_node_runtime_t * rt, vlib_frame_t * f)
{
    vlib_node_t *node;
    u32  n_left = 0;
    u32  *vectors;
    u32 *from        = vlib_frame_vector_args(f);
    //u32 i, n_left = 0;
    u32 n_left_from = f->n_vectors;
    //u32 *to_next;
    //u32 n_left_to_next;
    u32    next_index;
    vlib_frame_t *next;
    next_index  = rt->cached_next_index;
    node = vlib_get_node (vm, rt->node_index);
    vlib_cli_output (vm, "%v: call frame %p %U", node->name, f, format_my_node_frame, f);
    while(n_left_from > 0){
        //if (rt->n_next_nodes > 0) 
	do
	{
            //vlib_frame_t *next = vlib_get_next_frame (vm, rt, /* next index */ 0,to_next, n_left_to_next);
            next = vlib_get_next_frame_internal(vm, rt, /* next index */ 0,0);
            n_left = VLIB_FRAME_SIZE - next->n_vectors;
            vectors = vlib_frame_scalar_args (next) + (next->n_vectors) * sizeof ((vectors)[0]);
        }while(0);    
             while(n_left_from > 0 && n_left > 0){
		   u32             bi0;
                   bi0 =  *from;
		   *vectors = *from;
                   ASSERT (n_left > 0);
		   ++ from;
		   ++ vectors;
                   n_left -= 1;
                   n_left_from    -= 1;
                   vlib_validate_buffer_enqueue_x1(vm, rt, /*next_index*/next_index, vectors, n_left, bi0,  /*next_index*/0);
                }

        vlib_put_next_frame (vm, rt, /* next index */ 0, n_left);
        vlib_cli_output (vm, "%v: return frame %p", node->name, f);
    }
    return f->n_vectors;
}
#endif
#if 1
static uword ck_sample_node_fn(vlib_main_t *vm, vlib_node_runtime_t *node,
        vlib_frame_t * frame)
{
        u32 n_left_from, *from, *to_next;
        ck_sample_next_t     next_index;
	vlib_node_t *n;
        from        = vlib_frame_vector_args(frame);
        n_left_from = frame->n_vectors;
        next_index  = node->cached_next_index;

        n= vlib_get_node (vm, node->node_index);
        vlib_cli_output (vm, "%v: call frame %p %U", n->name, frame, format_my_node_frame, frame);
        while(n_left_from > 0){
                u32 n_left_to_next;
                vlib_get_next_frame(vm, node, next_index, to_next, n_left_to_next);

                while(n_left_from > 0 && n_left_to_next > 0){
                        //vlib_buffer_t  *b0;
                        u32             bi0, next0 = 0;

                        bi0 = to_next[0] = from[0];
                        from           += 1;
                        to_next        += 1;
                        n_left_to_next -= 1;
                        n_left_from    -= 1;
		#if 0
                        b0 = vlib_get_buffer(vm, bi0);
			ip4_header_t *ip4  = vlib_buffer_get_current(b0);
			set_ipv4_header(ip4);
		#endif
                        vlib_validate_buffer_enqueue_x1(vm, node, next_index,
                                to_next, n_left_to_next, bi0, next0);
                }

                vlib_put_next_frame(vm, node, next_index, n_left_to_next);
        }

        return frame->n_vectors;
}
#else
static void pending_frame_test(vlib_main_t * vm)
{
     int pending_frames = 0;
     vlib_node_main_t *nm = &vm->node_main;
     vlib_node_runtime_t *run;
     vlib_pending_frame_t *p;
     vlib_next_frame_t *nf;
     vlib_node_t *n;
     for (pending_frames = 0; pending_frames < vec_len (nm->pending_frames); pending_frames++)
     {
	 printf("\n");
	 p = vec_elt_at_index (nm->pending_frames, pending_frames);
	 run = vec_elt_at_index (nm->nodes_by_type[VLIB_NODE_TYPE_INTERNAL],
			                          p->node_runtime_index);
	 if (p->next_frame_index == VLIB_PENDING_FRAME_NO_NEXT_FRAME)
	  {
			         /* No next frame: so use placeholder on stack. */
				  nf->frame = NULL;
	  }
	  else
	  {
	      nf = vec_elt_at_index (nm->next_frames, p->next_frame_index); 
	      printf(" next frame addr %p",nf);
	  }
	 n = vlib_get_node (vm, run->node_index);
	 printf(" frame addr %p, node name %s \n",p->frame, n->name);
     }
     // refer to dispatch_pending_node 
     printf("    ********refer to dispatch_pending_node \n");
     for (pending_frames = 0; pending_frames < vec_len (nm->pending_frames); pending_frames++)
     {
	 p = nm->pending_frames +  pending_frames;
	 run = vec_elt_at_index (nm->nodes_by_type[VLIB_NODE_TYPE_INTERNAL],
			                          p->node_runtime_index);
	 if (p->next_frame_index == VLIB_PENDING_FRAME_NO_NEXT_FRAME)
	  {
			         /* No next frame: so use placeholder on stack. */
				  nf->frame = NULL;
	  }
	  else
	  {
	      nf = vec_elt_at_index (nm->next_frames, p->next_frame_index); 
	      printf(" next frame addr %p",nf);
	  }
	 n = vlib_get_node (vm, run->node_index);
	 printf(" frame addr %p, node name %s \n",p->frame, n->name);
     }
     printf("*********  traverse pending frame complete \n");
}
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
#if 0
       vlib_node_main_t *nm = &vm->node_main;
       vlib_node_t * cur_node = vec_elt (nm->nodes, node->node_index);
       vlib_node_runtime_get_next_frame_test (vm, cur_node, node->cached_next_index);
       vlib_node_runtime_get_next_frame_test (vm, next_node, node->cached_next_index);
       // will cause  ASSERT (next_index < run->n_next_nodes);
       //vlib_node_runtime_get_next_frame_test (vm,goto_index);
#else
       printf("********* before vlib_put_frame_to_node traverse pending frame \n");
       pending_frame_test(vm);
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
                        printf("*********  after vlib_put_frame_to_node traverse pending frame \n");
			pending_frame_test(vm);
#endif
                }

                //vlib_put_next_frame(vm, node, next_index, n_left_to_next);
        }
        return frame->n_vectors;
}
#endif
#if 0
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
                [CK_SAMPLE_NEXT_IP4]    = "my-node1",
                //[CK_SAMPLE_NEXT_IP4]    = "ip4-lookup",
                [CK_SAMPLE_DROP]        = "error-drop",
        },
};
VLIB_REGISTER_NODE (my_node1) = {
        .name		= "my-node1",
        .function       = my_func,
        .vector_size    = sizeof(u32),
        .format_trace   = format_ck_sample_trace,
        .type           = VLIB_NODE_TYPE_INTERNAL,
        .n_errors       = ARRAY_LEN(ck_sample_error_strings),
        .error_strings  = ck_sample_error_strings,
        .n_next_nodes   = CK_SAMPLE_NEXT_N,
        .next_nodes     = {
                [CK_SAMPLE_NEXT_IP4]    = "my-node2",
                //[CK_SAMPLE_NEXT_IP4]    = "ip4-lookup",
                [CK_SAMPLE_DROP]        = "error-drop",
        },
};
VLIB_REGISTER_NODE (my_node2) = {
        .name		= "my-node2",
        .function       = my_func,
        .vector_size    = sizeof(u32),
        .format_trace   = format_ck_sample_trace,
        .type           = VLIB_NODE_TYPE_INTERNAL,
        .n_errors       = ARRAY_LEN(ck_sample_error_strings),
        .error_strings  = ck_sample_error_strings,
        .n_next_nodes   = CK_SAMPLE_NEXT_N,
        .next_nodes     = {
                //[CK_SAMPLE_NEXT_IP4]    = "my-node1",
                [CK_SAMPLE_NEXT_IP4]    = "ip4-lookup",
                [CK_SAMPLE_DROP]        = "error-drop",
        },
};
#if 0
/* *INDENT-OFF* */
//VLIB_REGISTER_NODE (my_node1,static) = {
VLIB_REGISTER_NODE (my_node1) = {
	  .function = my_func,
	  .type = VLIB_NODE_TYPE_INPUT,
	  .name = "my-node1",
	  .scalar_size = sizeof (my_frame_t),
	  .vector_size = STRUCT_SIZE_OF (my_frame_t, vector[0]),
          .type           = VLIB_NODE_TYPE_INTERNAL,
	  .n_next_nodes = 1,
	  .next_nodes = {
	        [0] = "my-node2",
	   },
};
/* *INDENT-ON* */

/* *INDENT-OFF* */
//VLIB_REGISTER_NODE (my_node2,static) = {
VLIB_REGISTER_NODE (my_node2) = {
	  .function = my_func,
	  .name = "my-node2",
	  .scalar_size = sizeof (my_frame_t),
	  .vector_size = STRUCT_SIZE_OF (my_frame_t, vector[0]),
          .type           = VLIB_NODE_TYPE_INTERNAL,
          .n_next_nodes   = CK_SAMPLE_NEXT_N,
          .next_nodes     = {
                [CK_SAMPLE_NEXT_IP4]    = "ip4-lookup",
                [CK_SAMPLE_DROP]        = "error-drop",
           },
};
#endif
/* *INDENT-ON* */
#endif
#if 0
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
