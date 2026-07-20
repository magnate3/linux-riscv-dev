
# 1.xdp 获得的数据
以太网帧 数据结构体是 ethhdr，使用 xdp_buff 结构体来表示以太网帧的头部

 

#2.tc获得的数据
ip数据报数据结构体是 sk_buffer

tc BPF hook 的 BPF 程序可以读取 skb 的 mark、pkt_type、 protocol、priority、queue_mapping、napi_id、cb[]、hash、tc_classid 、tc_index、vlan 元数据、XDP 层传过来的自定义元数据以及其他信息。 tc BPF 的 BPF 上下文中使用了 struct __sk_buff，这个结构体中的所有成员字段都定 义在 linux/bpf.h 系统头文件。

 

# 3.sk_buff 和 xdp_buff 完全不同，二者各有优劣
+  sk_buff 修改与其关联的元数据非常方便，但它包含了大量协议相关的信息（例如 GSO 相关的状态），这使得无法仅仅通过重写包数据来切换协议。
这是因为协议栈是基于元数据处理包的，而不是每次都去读包的内容。因此，BPF 辅助函数需要额外的转换，并且还要正确处理 sk_buff 内部信息。   
+ xdp_buff 没有这些问题，因为它所处的阶段非常早，此时内核还没有分配 sk_buff，因此很容易实现各种类型的数据包重写。   
+ 但是，xdp_buff 的缺点是在它这个阶段进行 mangling 的时候，无法利用到 sk_buff 元数据。
解决这个问题的方式是从 XDP BPF 传递自定义的元数据到 tc BPF。这样，根据使用场景的不同，可以同时利用这两者 BPF 程序，以达到互补的效果。   
 
#  4.hook 触发点
tc BPF 程序在数据路径上的 ingress 和 egress 点都可以触发；而 XDP BPF 程序只能在 ingress 点触发。内核两个 hook 点：  
+ ingress hook sch_handle_ingress()：由 __netif_receive_skb_core() 触发   
+ egress hook sch_handle_egress()：由 __dev_queue_xmit() 触发  
__netif_receive_skb_core() 和 __dev_queue_xmit() 是 data path 的主要接收和发送函数，不考虑 XDP 的话（XDP 可能会拦截或修改，导致不经过这两个 hook 点）， 每个网络进入或离开系统的网络包都会经过这两个点，从而使得 tc BPF 程序具备完全可观测性。   



#  do_xdp_generic Generic XDP 处理（软件 XDP）
如果硬件网卡不支持 XDP 程序，那 XDP 程序会推迟到这里来执行。
XDP 的目的是将部分逻辑下放（offload）到网卡执行，通过硬件处理提高效率。 但是不是所有网卡都支持这个功能，所以内核引入了 Generic XDP 这样一个环境，如果网卡不支持 XDP， 那 XDP 程序就会推迟到这里来执行。它并不能提升效率，所以主要用来测试功能。
```
static int __netif_receive_skb_core(struct sk_buff **pskb, bool pfmemalloc, struct packet_type **ppt_prev) {
    struct sk_buff *skb = *pskb;
    net_timestamp_check(!netdev_tstamp_prequeue, skb); // 检查时间戳
 
    skb_reset_network_header(skb);
    if (!skb_transport_header_was_set(skb))
        skb_reset_transport_header(skb);
    skb_reset_mac_len(skb);
 
    struct packet_type *ptype    = NULL;
    struct net_device  *orig_dev = skb->dev;  // 记录 skb 原来所在的网络设备
 
another_round:
    skb->skb_iif = skb->dev->ifindex;         // 设置 skb 是从那个网络设备接收的
    __this_cpu_inc(softnet_data.processed);   // 更新 softnet_data.processed 计数
 
    if (static_branch_unlikely(&generic_xdp_needed_key)) { // Generic XDP（软件实现 XDP 功能）
        preempt_disable();
        ret2 = do_xdp_generic(skb->dev->xdp_prog, skb);
        preempt_enable();
 
        if (ret2 != XDP_PASS) {
            ret = NET_RX_DROP;
            goto out;
        }
        skb_reset_mac_len(skb);
    }

}
```

# mellanox  mlx5e_xdp_handle

```
if (unlikely((cqe->op_own >> 4) != MLX5_CQE_RESP_SEND)) {
    rq->stats.wqe_err++;
    mlx5e_page_release(rq, di, true);
    return NULL;
}


if (mlx5e_xdp_handle(rq, xdp_prog, di, data, cqe_bcnt))
    return NULL; /* page/packet was consumed by XDP */


skb = build_skb(va, RQ_PAGE_SIZE(rq));
```

