

# skb->skb_mstamp_ns 变迁

skb->skb_mstamp_ns 在4.9的内核中在每次tcp_transmit_skb发送数据的时候，通过 skb_mstamp_get(&skb->skb_mstamp)设置；而skb_mstamp_get本身就是获取当前的时间戳： 
``` 
static inline void skb_mstamp_get(struct skb_mstamp *cl)

{

u64 val = local_clock();

do_div(val, NSEC_PER_USEC);

cl->stamp_us = (u32)val;

cl->stamp_jiffies = (u32)jiffies;

}
```



在5.6中使用
```

prior_wstamp = tp->tcp_wstamp_ns;

tp->tcp_wstamp_ns = max(tp->tcp_wstamp_ns, tp->tcp_clock_cache);

skb->skb_mstamp_ns = tp->tcp_wstamp_ns;
```

设置值，为什么改为第二种呢？那就是更能体现TCP流的特性。
其实还是因为新的EDT特性，有了EDT计算更加准确。skb_mstamp_ns 有可能会跑到比当前的时钟更靠后。
但是EDT超级不准确.