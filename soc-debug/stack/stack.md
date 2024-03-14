
# ping recv and send
```
(gdb) bt
#0  vp_notify (vq=0xffff888002cedcc0) at drivers/virtio/virtio_pci_common.c:45
#1  0xffffffff812fcfca in virtqueue_notify (_vq=0xffff888002cedcc0) at drivers/virtio/virtio_ring.c:1949
#2  0xffffffff813f0606 in start_xmit (skb=0xffff8880032dcc00, dev=<optimized out>) at ./include/linux/skbuff.h:2807
#3  0xffffffff814d9f8f in __netdev_start_xmit (more=<optimized out>, dev=0xffff888002cf6000, skb=0xffff8880032dcc00, ops=<optimized out>) at ./include/linux/netdevice.h:4995
#4  netdev_start_xmit (more=<optimized out>, txq=0xffff8880031e2600, dev=0xffff888002cf6000, skb=0xffff8880032dcc00) at ./include/linux/netdevice.h:5009
#5  xmit_one (more=<optimized out>, txq=0xffff8880031e2600, dev=0xffff888002cf6000, skb=0xffff8880032dcc00) at net/core/dev.c:3584
#6  dev_hard_start_xmit (first=first@entry=0xffff8880032dcc00, dev=dev@entry=0xffff888002cf6000, txq=txq@entry=0xffff8880031e2600, ret=ret@entry=0xffffc9000000373c)
    at net/core/dev.c:3600
#7  0xffffffff8150d7c4 in sch_direct_xmit (skb=skb@entry=0xffff8880032dcc00, q=q@entry=0xffff888003328800, dev=dev@entry=0xffff888002cf6000, txq=txq@entry=0xffff8880031e2600, 
    root_lock=root_lock@entry=0x0 <fixed_percpu_data>, validate=validate@entry=true) at net/sched/sch_generic.c:342
#8  0xffffffff814da541 in __dev_xmit_skb (txq=0xffff8880031e2600, dev=0xffff888002cf6000, q=0xffff888003328800, skb=0xffff8880032dcc00) at net/core/dev.c:3811
#9  __dev_queue_xmit (skb=skb@entry=0xffff8880032dcc00, sb_dev=sb_dev@entry=0x0 <fixed_percpu_data>) at net/core/dev.c:4179
#10 0xffffffff814da9db in dev_queue_xmit (skb=skb@entry=0xffff8880032dcc00) at net/core/dev.c:4247
#11 0xffffffff81530c58 in neigh_hh_output (skb=<optimized out>, hh=<optimized out>) at ./include/net/neighbour.h:500
#12 neigh_output (skip_cache=<optimized out>, skb=0xffff8880032dcc00, n=0xffff888003273800) at ./include/net/neighbour.h:514
#13 ip_finish_output2 (net=net@entry=0xffffffff81ed3ac0 <init_net>, sk=sk@entry=0xffff88800265e3c0, skb=skb@entry=0xffff8880032dcc00) at net/ipv4/ip_output.c:221
#14 0xffffffff81530f8b in __ip_finish_output (net=0xffffffff81ed3ac0 <init_net>, sk=0xffff88800265e3c0, skb=0xffff8880032dcc00) at net/ipv4/ip_output.c:299
#15 0xffffffff81531159 in ip_finish_output (skb=<optimized out>, sk=<optimized out>, net=<optimized out>) at net/ipv4/ip_output.c:309
#16 NF_HOOK_COND (cond=<optimized out>, okfn=<optimized out>, out=<optimized out>, in=<optimized out>, skb=<optimized out>, sk=<optimized out>, net=<optimized out>, hook=4, pf=2 '\002')
    at ./include/linux/netfilter.h:407
#17 ip_output (net=<optimized out>, sk=<optimized out>, skb=<optimized out>) at net/ipv4/ip_output.c:423
#18 0xffffffff815311cd in dst_output (skb=<optimized out>, sk=<optimized out>, net=0xffffffff81ed3ac0 <init_net>) at ./include/linux/skbuff.h:985
#19 ip_local_out (net=net@entry=0xffffffff81ed3ac0 <init_net>, sk=<optimized out>, skb=<optimized out>) at net/ipv4/ip_output.c:126
#20 0xffffffff81533184 in ip_send_skb (net=0xffffffff81ed3ac0 <init_net>, skb=<optimized out>) at net/ipv4/ip_output.c:1555
#21 0xffffffff815331de in ip_push_pending_frames (sk=sk@entry=0xffff88800265e3c0, fl4=fl4@entry=0xffffc90000003980) at ./include/net/net_namespace.h:327
#22 0xffffffff8156a150 in icmp_push_reply (icmp_param=icmp_param@entry=0xffffc900000039f8, fl4=fl4@entry=0xffffc90000003980, ipc=ipc@entry=0xffffc90000003950, 
    rt=rt@entry=0xffffc90000003948) at net/ipv4/icmp.c:393
#23 0xffffffff8156ad2c in icmp_reply (icmp_param=icmp_param@entry=0xffffc900000039f8, skb=skb@entry=0xffff8880025d5600) at net/ipv4/icmp.c:455
#24 0xffffffff8156b63e in icmp_echo (skb=0xffff8880025d5600) at net/ipv4/icmp.c:1015
#25 0xffffffff8156b694 in icmp_echo (skb=<optimized out>) at net/ipv4/icmp.c:1001
#26 0xffffffff8156b7fb in icmp_rcv (skb=0xffff8880025d5600) at net/ipv4/icmp.c:1261
#27 0xffffffff8152cf41 in ip_protocol_deliver_rcu (net=net@entry=0xffffffff81ed3ac0 <init_net>, skb=skb@entry=0xffff8880025d5600, protocol=1) at net/ipv4/ip_input.c:218
#28 0xffffffff8152cfe3 in ip_local_deliver_finish (sk=0x0 <fixed_percpu_data>, skb=0xffff8880025d5600, net=0xffffffff81ed3ac0 <init_net>) at ./include/linux/skbuff.h:2568
#29 NF_HOOK (okfn=<optimized out>, out=0x0 <fixed_percpu_data>, in=<optimized out>, skb=0xffff8880025d5600, sk=0x0 <fixed_percpu_data>, net=0xffffffff81ed3ac0 <init_net>, hook=1, 
    pf=2 '\002') at ./include/linux/netfilter.h:415
#30 ip_local_deliver (skb=0xffff8880025d5600) at net/ipv4/ip_input.c:252
#31 0xffffffff8152d064 in dst_input (skb=<optimized out>) at ./include/linux/skbuff.h:985
#32 ip_sublist_rcv_finish (head=head@entry=0xffffc90000003b50) at net/ipv4/ip_input.c:551
#33 0xffffffff8152d170 in ip_list_rcv_finish (net=net@entry=0xffffffff81ed3ac0 <init_net>, head=head@entry=0xffffc90000003bc0, sk=0x0 <fixed_percpu_data>) at net/ipv4/ip_input.c:601
#34 0xffffffff8152d31f in ip_sublist_rcv (dev=0x0 <fixed_percpu_data>, net=0xffffffff81ed3ac0 <init_net>, head=0xffffc90000003bc0) at net/ipv4/ip_input.c:609
#35 ip_list_rcv (head=0xffffc90000003c38, pt=<optimized out>, orig_dev=<optimized out>) at net/ipv4/ip_input.c:644
#36 0xffffffff814dc0b3 in __netif_receive_skb_list_ptype (orig_dev=0xffff888002cf6000, pt_prev=0xffffffff81ee4140 <ip_packet_type>, head=0xffffc90000003c38) at net/core/dev.c:5494
#37 __netif_receive_skb_list_core (head=head@entry=0xffff888002d06108, pfmemalloc=pfmemalloc@entry=false) at net/core/dev.c:5539
#38 0xffffffff814dc290 in __netif_receive_skb_list (head=0xffff888002d06108) at net/core/dev.c:5591
#39 netif_receive_skb_list_internal (head=head@entry=0xffff888002d06108) at net/core/dev.c:5682
#40 0xffffffff814dc3d9 in gro_normal_list (napi=0xffff888002d06008) at net/core/dev.c:5836
#41 0xffffffff814dc441 in gro_normal_list (napi=0xffff888002d06008) at net/core/dev.c:5849
#42 gro_normal_one (napi=napi@entry=0xffff888002d06008, skb=skb@entry=0xffff8880032dc900, segs=segs@entry=1) at net/core/dev.c:5849
#43 0xffffffff814dcd8b in napi_skb_finish (ret=GRO_NORMAL, skb=0xffff8880032dc900, napi=0xffff888002d06008) at net/core/dev.c:6186
#44 napi_gro_receive (napi=napi@entry=0xffff888002d06008, skb=skb@entry=0xffff8880032dc900) at net/core/dev.c:6216
#45 0xffffffff813f2c62 in receive_buf (vi=vi@entry=0xffff888002cf6800, rq=rq@entry=0xffff888002d06000, buf=<optimized out>, len=<optimized out>, ctx=<optimized out>, 
    xdp_xmit=xdp_xmit@entry=0xffffc90000003e58, stats=<optimized out>) at drivers/net/virtio_net.c:1163
#46 0xffffffff813f4142 in virtnet_receive (xdp_xmit=0xffffc90000003e58, budget=64, rq=0xffff888002d06000) at drivers/net/virtio_net.c:1427
#47 virtnet_poll (napi=0xffff888002d06008, budget=64) at drivers/net/virtio_net.c:1536
#48 0xffffffff814dd4da in __napi_poll (n=n@entry=0xffff888002d06008, repoll=repoll@entry=0xffffc90000003f27) at net/core/dev.c:6998
#49 0xffffffff814dd76d in napi_poll (repoll=0xffffc90000003f38, n=0xffff888002d06008) at net/core/dev.c:7065
#50 net_rx_action (h=<optimized out>) at net/core/dev.c:7152
#51 0xffffffff81a000bf in __do_softirq () at kernel/softirq.c:558
#52 0xffffffff81059f17 in invoke_softirq () at kernel/softirq.c:432
#53 __irq_exit_rcu () at kernel/softirq.c:636
--Type <RET> for more, q to quit, c to continue without paging--c
#54 __irq_exit_rcu () at kernel/softirq.c:626
#55 irq_exit_rcu () at kernel/softirq.c:648
#56 0xffffffff81619272 in common_interrupt (regs=0xffffffff81e03d88, error_code=<optimized out>) at arch/x86/kernel/irq.c:240
#57 0xffffffff81800b9e in asm_common_interrupt () at ./arch/x86/include/asm/idtentry.h:629
#58 0x0000000000000000 in ?? ()
```