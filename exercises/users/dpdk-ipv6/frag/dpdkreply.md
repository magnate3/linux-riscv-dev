
# python

```
[root@bogon data]# python -V
Python 2.7.5
```

# use

```
python dpdkreply_test1.py --pcap  dpdk-ip62.pcap 
```

# ipv6.nh

如果要进行分片，在创建IPv6报文头时，如果设置ipv6.nh=58，会导致对端在采用tcpdump时，报错误。不要设置nh=ipv6.nh
'''
    set nh=ipv6.nh will cause error
    ipv6_new=IPv6(src=src_ip6, dst=dst_ip6, hlim=ipv6.hlim, fl=ipv6.fl,tc=ipv6.tc,plen=ipv6.plen,nh=ipv6.nh) 
'''
 

# cksum
随便设置cksum会导致对端直接drop,接收不到reply    
'''
    #csum=in6_chksum(58, pkt, str(icmpv6))
    #icmpv6.cksum = csum
    #icmpv6.cksum = 0xffee
'''
    
   