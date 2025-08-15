

```
from scapy.all import *
pkt = Ether()/IPv6()/ICMPv6ND_RA()/ ICMPv6NDOptPrefixInfo(prefix="2001:db8:cafe:deca::", prefixlen=64)/ ICMPv6NDOptSrcLLAddr(lladdr="00:b0:de:ad:be:ef")
pkt.show2()
```


```
[root@centos7 tcpreplay]# python3 test_ipv62.py 
###[ Ethernet ]### 
  dst       = 33:33:00:00:00:01
  src       = b0:08:75:5f:b8:5b
  type      = IPv6
###[ IPv6 ]### 
     version   = 6
     tc        = 0
     fl        = 0
     plen      = 56
     nh        = ICMPv6
     hlim      = 255
     src       = fe80::a82e:8486:712:201a
     dst       = ff02::1
###[ ICMPv6 Neighbor Discovery - Router Advertisement ]### 
        type      = Router Advertisement
        code      = 0
        cksum     = 0x6681
        chlim     = 0
        M         = 0
        O         = 0
        H         = 0
        prf       = High
        P         = 0
        res       = 0
        routerlifetime= 1800
        reachabletime= 0
        retranstimer= 0
###[ ICMPv6 Neighbor Discovery Option - Prefix Information ]### 
           type      = 3
           len       = 4
           prefixlen = 64
           L         = 1
           A         = 1
           R         = 0
           res1      = 0
           validlifetime= 0xffffffff
           preferredlifetime= 0xffffffff
           res2      = 0x0
           prefix    = 2001:db8:cafe:deca::
###[ ICMPv6 Neighbor Discovery Option - Source Link-Layer Address ]### 
              type      = 1
              len       = 1
              lladdr    = 00:b0:de:ad:be:ef
```