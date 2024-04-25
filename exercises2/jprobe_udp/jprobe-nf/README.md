

# skb_orphan   

dump_stack_skb会被skb_orphan调用
```
static inline void skb_orphan(struct sk_buff *skb)
{
if (skb->destructor)
skb->destructor(skb);
skb->destructor = NULL;
skb->sk = NULL;
}
```

# linux-6.0.2 arch/arm64/include/asm/ptrace.h


+ os

```
[root@centos7 nat64_udp_frag]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 nat64_udp_frag]# 
```

```
/*
 * Read a register given an architectural register index r.
 * This handles the common case where 31 means XZR, not SP.
 */
static inline unsigned long pt_regs_read_reg(const struct pt_regs *regs, int r)
{
        return (r == 31) ? 0 : regs->regs[r];
}
static inline unsigned long regs_get_kernel_argument(struct pt_regs *regs,
                                                     unsigned int n)
{
#define NR_REG_ARGUMENTS 8
        if (n < NR_REG_ARGUMENTS)
                return pt_regs_read_reg(regs, n);
        return 0;
}

```


# run


```
[ 9386.574964] kretprobe at 0xffff000007cf01a8 unregistered
[ 9386.580252] Missed probing 0 instances of ip6t_do_table
[ 9405.547331] Planted return probe at ip6t_do_table: 0xffff000007cf01a8
[ 9410.683984] udp Source: 2001:0db8:0000:0000:0000:0000:0a0a:6751 sport:5080 --->Dest:  2001:0db8:0000:0000:0000:0000:0a0a:6752 dport:8080    ---> 
[ 9410.696963] ip6t_do_table(raw) - devin=(null)/0, devout=nat64/11,  proto=0, verdict=0x1
[ 9410.704941] udp Source: 2001:0db8:0000:0000:0000:0000:0a0a:6751 sport:5080 --->Dest:  2001:0db8:0000:0000:0000:0000:0a0a:6752 dport:8080    ---> 
[ 9410.717919] ip6t_do_table(mangle) - devin=(null)/0, devout=nat64/11,  proto=0, verdict=0x1
[ 9410.726152] udp Source: 2001:0db8:0000:0000:0000:0000:0a0a:6751 sport:5080 --->Dest:  2001:0db8:0000:0000:0000:0000:0a0a:6752 dport:8080    ---> 
[ 9410.739128] ip6t_do_table(filter) - devin=(null)/0, devout=nat64/11,  proto=0, verdict=0x1
[ 9410.747358] udp Source: 2001:0db8:0000:0000:0000:0000:0a0a:6751 sport:5080 --->Dest:  2001:0db8:0000:0000:0000:0000:0a0a:6752 dport:8080    ---> 
[ 9410.760334] ip6t_do_table(security) - devin=(null)/0, devout=nat64/11,  proto=0, verdict=0x1
[ 9410.768739] udp Source: 2001:0db8:0000:0000:0000:0000:0a0a:6751 sport:5080 --->Dest:  2001:0db8:0000:0000:0000:0000:0a0a:6752 dport:8080    ---> 
[ 9410.781715] ip6t_do_table(mangle) - devin=(null)/0, devout=nat64/11,  proto=0, verdict=0x1
```