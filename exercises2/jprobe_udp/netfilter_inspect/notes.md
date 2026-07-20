# The "ops" that gets registered

- net/ipv4/netfilter/ip_tables.c()
```
/* Returns one of the generic firewall policies, like NF_ACCEPT. */
unsigned int
ipt_do_table(struct sk_buff *skb,
             const struct nf_hook_state *state,
             struct xt_table *table)
{
    ...
}
```

- coming from
```
static unsigned int
iptable_filter_hook(void *priv, struct sk_buff *skb,
                    const struct nf_hook_state *state)
{
        return ipt_do_table(skb, state, priv);  <====
}
```

```
static struct nf_hook_ops *filter_ops __read_mostly;
```

```
static int __init iptable_filter_init(void)
{
        int ret;

        filter_ops = xt_hook_ops_alloc(&packet_filter, iptable_filter_hook);    <====
    ...
}
```

```
static int __net_init iptable_filter_table_init(struct net *net)
{
    ...
        err = ipt_register_table(net, &packet_filter, repl, filter_ops);    <===
    ...
}
```

```
int ipt_register_table(struct net *net, const struct xt_table *table,
                       const struct ipt_replace *repl,
                       const struct nf_hook_ops *template_ops)
{
    ...
        ret = nf_register_net_hooks(net, ops, num_ops);
    ...
}
```

# dump_stack within ipt_do_table()

```
Dec 24 18:56:22 leap kernel: ipt_do_table returned 0 with sk_buff at 0x00000000e17625ab - src=9af4, dst=1f90, retval=0
Dec 24 18:56:22 leap kernel: CPU: 2 PID: 0 Comm: swapper/2 Tainted: G           OE     N 5.14.21-150500.55.39-default #1 SLE15-SP5 534d850193522c220e62d98d0494b02a0dde2443
Dec 24 18:56:22 leap kernel: Hardware name: QEMU Standard PC (i440FX + PIIX, 1996), BIOS rel-1.16.0-0-gd239552c-rebuilt.opensuse.org 04/01/2014
Dec 24 18:56:22 leap kernel: Call Trace:
Dec 24 18:56:22 leap kernel:  <IRQ>
Dec 24 18:56:22 leap kernel:  dump_stack_lvl+0x45/0x5b
Dec 24 18:56:22 leap kernel:  ret_handler+0x81/0xb7 [kretprobe_nf_rules 84caed98b44e76c86965d753e9c55a9000b306ce]
Dec 24 18:56:22 leap kernel:  __kretprobe_trampoline_handler+0xbe/0x140
Dec 24 18:56:22 leap kernel:  trampoline_handler+0x43/0x60
Dec 24 18:56:22 leap kernel:  __kretprobe_trampoline+0x2a/0x50
Dec 24 18:56:22 leap kernel:  ? ipt_do_table+0x347/0x640 [ip_tables 72a54f0fa9d553ac7f37c1f9c9a6f4f151651b53]
Dec 24 18:56:22 leap kernel:  ? nf_hook_slow+0x40/0xc0
Dec 24 18:56:22 leap kernel:  elfcorehdr_read+0x40/0x40
Dec 24 18:56:22 leap kernel:  ip_local_deliver+0xdb/0x110
Dec 24 18:56:22 leap kernel:  ? ip_protocol_deliver_rcu+0x1a0/0x1a0
Dec 24 18:56:22 leap kernel:  ip_sublist_rcv_finish+0x69/0x80
Dec 24 18:56:22 leap kernel:  ip_sublist_rcv+0x16b/0x200
Dec 24 18:56:22 leap kernel:  ? inet_gro_receive+0x253/0x2e0
Dec 24 18:56:22 leap kernel:  ip_list_rcv+0x111/0x140
Dec 24 18:56:22 leap kernel:  __netif_receive_skb_list_core+0x25d/0x280
Dec 24 18:56:22 leap kernel:  netif_receive_skb_list_internal+0x18c/0x2b0
Dec 24 18:56:22 leap kernel:  napi_complete_done+0x104/0x190
Dec 24 18:56:22 leap kernel:  virtnet_poll+0x2f0/0x439 [virtio_net e173a78d5d30d7fd5c2a28bf5b85ba5ebf21f998]
Dec 24 18:56:22 leap kernel:  __napi_poll+0x2a/0x1b0
Dec 24 18:56:22 leap kernel:  net_rx_action+0x24c/0x2a0
Dec 24 18:56:22 leap kernel:  __do_softirq+0xd2/0x2c0
Dec 24 18:56:22 leap kernel:  irq_exit_rcu+0xa4/0xc0
Dec 24 18:56:22 leap kernel:  common_interrupt+0x5d/0xa0
Dec 24 18:56:22 leap kernel:  </IRQ>
Dec 24 18:56:22 leap kernel:  <TASK>
Dec 24 18:56:22 leap kernel:  asm_common_interrupt+0x59/0x80
Dec 24 18:56:22 leap kernel: RIP: 0010:native_safe_halt+0xb/0x10
Dec 24 18:56:22 leap kernel: Code: ff ff ff eb ba cc cc cc cc cc cc cc cc cc cc eb 07 0f 00 2d 09 1d 39 00 f4 c3 cc cc cc cc 90 eb 07 0f 00 2d f9 1c 39 00 fb f4 <c3> cc cc cc cc 0f 1f 44 00 00 65 8b 15 cc 04 5a 5f 0f 1f 44 00 00
Dec 24 18:56:22 leap kernel: RSP: 0018:ffffa6400009bec8 EFLAGS: 00000206
Dec 24 18:56:22 leap kernel: RAX: ffffffffa0a7b380 RBX: 0000000000000002 RCX: 0000000000000001
Dec 24 18:56:22 leap kernel: RDX: 00000000000249b6 RSI: 0000000000000087 RDI: ffff94a4402a2900
Dec 24 18:56:22 leap kernel: RBP: ffffffffa1b4b4a0 R08: 00000000000249b6 R09: 0000000000000000
Dec 24 18:56:22 leap kernel: R10: 0000000000000003 R11: 0000000000000149 R12: 0000000000000000
Dec 24 18:56:22 leap kernel: R13: 0000000000000000 R14: ffffffffffffffff R15: ffff94a4402a2900
Dec 24 18:56:22 leap kernel:  ? tdx_safe_halt+0x80/0x80
Dec 24 18:56:22 leap kernel:  default_idle+0xa/0x20
Dec 24 18:56:22 leap kernel:  default_idle_call+0x2d/0xf0
Dec 24 18:56:22 leap kernel:  do_idle+0x1f0/0x2d0
Dec 24 18:56:22 leap kernel:  cpu_startup_entry+0x19/0x20
Dec 24 18:56:22 leap kernel:  start_secondary+0x11c/0x160
Dec 24 18:56:22 leap kernel:  secondary_startup_64_no_verify+0xd2/0xdb
Dec 24 18:56:22 leap kernel:  </TASK>
```
