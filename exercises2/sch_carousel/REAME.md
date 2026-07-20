

```
  union {
        ktime_t		tstamp;    // 时间戳
        u64		skb_mstamp_ns; /* earliest departure time */
    };
```

#    不再基于排队（queue），而是基于时间戳（EDT）
[使用 Cilium 给 K8s 数据平面提供强大的带宽管理功能](https://jishu.proginn.com/doc/625764781f33a4b93)    

结合 流量控制（TC）五十年：从基于缓冲队列（Queue）到基于时间戳（EDT）的演进（Google, 2018）[2]， 这里只做几点说明：   

TCP 的发送模型是尽可能快（As Fast As Possible, AFAP）   
网络流量主要是靠网络设备上的出向队列（device output queue）做整形（shaping）  
队列长度（queue length）和接收窗口（receive window）决定了传输中的数据速率（in-flight rate）  
“多快”（how fast）取决于队列的 drain rate   

 