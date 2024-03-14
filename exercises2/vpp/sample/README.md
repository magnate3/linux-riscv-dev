# vpp的每一个插件internal节点报文处理函数大致如下函数
```C
VLIB_NODE_FN (sample_node) (vlib_main_t * vm, vlib_node_runtime_t * node,
			    vlib_frame_t * frame)
{
  u32 n_left_from, *from, *to_next;
  sample_next_t next_index;
  u32 pkts_swapped = 0;

  /* 本节点收到的vector包的起始地址 */
  from = vlib_frame_vector_args (frame);
  /* 本节点收到的vector包数 */
  n_left_from = frame->n_vectors;
  /* cached_next_index记录着上一次经过该节点时的next_index
       next_index对应着VLIB_REGISTER_NODE (sample_node).next_nodes中下一节点的索引 */
  next_index = node->cached_next_index;

  while (n_left_from > 0)
    {
      u32 n_left_to_next;

      /* to_next: next_index所指下一个节点的收包缓存的空闲位置首地址 */
      /* n_left_to_next:下一个节点收包缓存的空闲位置总数 */
      vlib_get_next_frame (vm, node, next_index, to_next, n_left_to_next);

      /* 一次性处理两个包 */
      while (n_left_from >= 4 && n_left_to_next >= 2)
	{
      /* next0和next1指明包的下一个节点索引值 */
	  u32 next0 = SAMPLE_NEXT_INTERFACE_OUTPUT;
	  u32 next1 = SAMPLE_NEXT_INTERFACE_OUTPUT;
	  u32 sw_if_index0, sw_if_index1;
	  u8 tmp0[6], tmp1[6];
	  ethernet_header_t *en0, *en1;
	  u32 bi0, bi1;
	  vlib_buffer_t *b0, *b1;

	  /* Prefetch next iteration. */
      /* from[2]和from[3]是第2和第3个buf的索引，如果这里有第2和第3报文进来
         的话，就是放在from[2]和from[3]索引位置，所以这里对其做指令预取*/
	  {
	    vlib_buffer_t *p2, *p3;

	    p2 = vlib_get_buffer (vm, from[2]);
	    p3 = vlib_get_buffer (vm, from[3]);

	    vlib_prefetch_buffer_header (p2, LOAD);
	    vlib_prefetch_buffer_header (p3, LOAD);

	    CLIB_PREFETCH (p2->data, CLIB_CACHE_LINE_BYTES, STORE);
	    CLIB_PREFETCH (p3->data, CLIB_CACHE_LINE_BYTES, STORE);
	  }

	  /* speculatively enqueue b0 and b1 to the current next frame */
      /* from[0]和from[1]中保存的是本节点收到包的包索引值，
         这里直接把from[0]和from[1]放到to_next[0]和to_next[1]里面了,
         这里的意思是假设直接把报文放到next_index对应下一个节点的收包
         缓存里面了，后面vlib_validate_buffer_enqueue_x2宏会对其做调整
      */
	  to_next[0] = bi0 = from[0];
	  to_next[1] = bi1 = from[1];

      /* 偏移from和to_next指针的位置，并减少n_left_from和n_left_to_next
         这里n_left_from表示当前节点收到的报文总数
         而n_left_to_next表示下一个节点收包缓存队列的最大数量
      */
	  from += 2;
	  to_next += 2;
	  n_left_from -= 2;
	  n_left_to_next -= 2;

      /* 根据buf index从当前node里面拿到对应的vlib_buffer_t */
	  b0 = vlib_get_buffer (vm, bi0);
	  b1 = vlib_get_buffer (vm, bi1);

	  ASSERT (b0->current_data == 0);
	  ASSERT (b1->current_data == 0);

      /* 从vlib_buffer_t获取报文地址 */
	  en0 = vlib_buffer_get_current (b0);
	  en1 = vlib_buffer_get_current (b1);

      /* 下面这一段只是交换以太网报文的mac地址 */
	  /* This is not the fastest way to swap src + dst mac addresses */
#define _(a) tmp0[a] = en0->src_address[a];
	  foreach_mac_address_offset;
#undef _
#define _(a) en0->src_address[a] = en0->dst_address[a];
	  foreach_mac_address_offset;
#undef _
#define _(a) en0->dst_address[a] = tmp0[a];
	  foreach_mac_address_offset;
#undef _

#define _(a) tmp1[a] = en1->src_address[a];
	  foreach_mac_address_offset;
#undef _
#define _(a) en1->src_address[a] = en1->dst_address[a];
	  foreach_mac_address_offset;
#undef _
#define _(a) en1->dst_address[a] = tmp1[a];
	  foreach_mac_address_offset;
#undef _

      /* 获取rx的if index后设置到tx if index里面 */
	  sw_if_index0 = vnet_buffer (b0)->sw_if_index[VLIB_RX];
	  sw_if_index1 = vnet_buffer (b1)->sw_if_index[VLIB_RX];

	  /* Send pkt back out the RX interface */
	  vnet_buffer (b0)->sw_if_index[VLIB_TX] = sw_if_index0;
	  vnet_buffer (b1)->sw_if_index[VLIB_TX] = sw_if_index1;

	  pkts_swapped += 2;

	  if (PREDICT_FALSE ((node->flags & VLIB_NODE_FLAG_TRACE)))
	    {
	      if (b0->flags & VLIB_BUFFER_IS_TRACED)
		{
		  sample_trace_t *t =
		    vlib_add_trace (vm, node, b0, sizeof (*t));
		  t->sw_if_index = sw_if_index0;
		  t->next_index = next0;
		  clib_memcpy_fast (t->new_src_mac, en0->src_address,
				    sizeof (t->new_src_mac));
		  clib_memcpy_fast (t->new_dst_mac, en0->dst_address,
				    sizeof (t->new_dst_mac));

		}
	      if (b1->flags & VLIB_BUFFER_IS_TRACED)
		{
		  sample_trace_t *t =
		    vlib_add_trace (vm, node, b1, sizeof (*t));
		  t->sw_if_index = sw_if_index1;
		  t->next_index = next1;
		  clib_memcpy_fast (t->new_src_mac, en1->src_address,
				    sizeof (t->new_src_mac));
		  clib_memcpy_fast (t->new_dst_mac, en1->dst_address,
				    sizeof (t->new_dst_mac));
		}
	    }

	  /* verify speculative enqueues, maybe switch current next frame */
      /* 
        next_index:默认的下一结点的index
        next0:实际的下一个结点的index
        next0 == next_index则不需要做特别的处理，报文会自动进入下一个节点
        next0 != next_index则需要对该数据包做调整，从之前next_index对应
                           的frame中删除，添加到next0对应的frame中

        next1的判断和next0一样
      */
	  vlib_validate_buffer_enqueue_x2 (vm, node, next_index,
					   to_next, n_left_to_next,
					   bi0, bi1, next0, next1);
	}

      /* 一次性处理一个包, 处理逻辑和上面一致 */
      while (n_left_from > 0 && n_left_to_next > 0)
	{
	  u32 bi0;
	  vlib_buffer_t *b0;
	  u32 next0 = SAMPLE_NEXT_INTERFACE_OUTPUT;
	  u32 sw_if_index0;
	  u8 tmp0[6];
	  ethernet_header_t *en0;

	  /* speculatively enqueue b0 to the current next frame */
	  bi0 = from[0];
	  to_next[0] = bi0;
	  from += 1;
	  to_next += 1;
	  n_left_from -= 1;
	  n_left_to_next -= 1;

	  b0 = vlib_get_buffer (vm, bi0);
	  /*
	   * Direct from the driver, we should be at offset 0
	   * aka at &b0->data[0]
	   */
	  ASSERT (b0->current_data == 0);

	  en0 = vlib_buffer_get_current (b0);

	  /* This is not the fastest way to swap src + dst mac addresses */
#define _(a) tmp0[a] = en0->src_address[a];
	  foreach_mac_address_offset;
#undef _
#define _(a) en0->src_address[a] = en0->dst_address[a];
	  foreach_mac_address_offset;
#undef _
#define _(a) en0->dst_address[a] = tmp0[a];
	  foreach_mac_address_offset;
#undef _

	  sw_if_index0 = vnet_buffer (b0)->sw_if_index[VLIB_RX];

	  /* Send pkt back out the RX interface */
	  vnet_buffer (b0)->sw_if_index[VLIB_TX] = sw_if_index0;

	  if (PREDICT_FALSE ((node->flags & VLIB_NODE_FLAG_TRACE)
			     && (b0->flags & VLIB_BUFFER_IS_TRACED)))
	    {
	      sample_trace_t *t = vlib_add_trace (vm, node, b0, sizeof (*t));
	      t->sw_if_index = sw_if_index0;
	      t->next_index = next0;
	      clib_memcpy_fast (t->new_src_mac, en0->src_address,
				sizeof (t->new_src_mac));
	      clib_memcpy_fast (t->new_dst_mac, en0->dst_address,
				sizeof (t->new_dst_mac));
	    }

	  pkts_swapped += 1;

	  /* verify speculative enqueue, maybe switch current next frame */
	  vlib_validate_buffer_enqueue_x1 (vm, node, next_index,
					   to_next, n_left_to_next,
					   bi0, next0);
	}

      /* 所有流程都正确处理完毕后，下一结点的frame上已经有本结点处理过后的数据索引
         执行该函数，将相关信息登记到vlib_pending_frame_t中，准备开始调度处理 
      */
      vlib_put_next_frame (vm, node, next_index, n_left_to_next);
    }

  vlib_node_increment_counter (vm, sample_node.index,
			       SAMPLE_ERROR_SWAPPED, pkts_swapped);
  return frame->n_vectors;
}
```
# 初始化
显示的结果都是VPP在进入main函数loop循环之前的初始化函数。VPP初始化的实现代码是src/vlib/init.h，通过宏VLIB_INIT_FUNCTION来定义构造函数和析构函数。   
```
vpp# show init-function
[0]: vlib_buffer_funcs_init
[1]: vlib_cli_init
[2]: unix_main_init
[3]: linux_vmbus_init
[4]: unix_input_init
[5]: linux_pci_init
[6]: vlib_log_init
[7]: pci_bus_init
[8]: punt_init
[9]: punt_node_init
```


VPP图节点调度涉及如下结构体：  
vlib_node_main_t，图节点柱结构，记录图节点的全局信息。   
vlib_node_t，记录图节点的相关静态信息。  
vlib_node_runtime_t，图节点调度实际使用的结构体，由vlib_node_t结构体中的信息和私有信息组成。  
vlib_frame_t，保存图节点要处理的数据的内存地址信息。  
vlib_pending_frame_t，记录运行节点的索引、数据包索引和下一个数据包的索引。  
vlib_next_frame_t，记录图节点要处理的下一条的数据。  