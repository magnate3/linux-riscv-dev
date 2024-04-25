# 配置

```
[root@centos7 reasm]# iptables -F
[root@centos7 reasm]# sysctl -w net.ipv6.conf.all.disable_ipv6=0
net.ipv6.conf.all.disable_ipv6 = 0
[root@centos7 reasm]# sysctl -w net.ipv6.conf.default.disable_ipv6=0
net.ipv6.conf.default.disable_ipv6 = 0
[root@centos7 reasm]# sysctl -w net.ipv6.conf.enp5s0.disable_ipv6=0
net.ipv6.conf.enp5s0.disable_ipv6 = 0
[root@centos7 reasm]# 
```

# python3

```
[root@centos7 reasm]# python3 -V
Python 3.6.8
[root@centos7 reasm]# 
```
# UDP checksum

``` Text   
The following example explains how to use the checksum() function to compute and UDP checksum manually. The following steps must be performed:

compute the UDP pseudo header as described in RFC768

build a UDP packet with Scapy with p[UDP].chksum=0

call checksum() with the pseudo header and the UDP packet

from scapy.all import *

# Get the UDP checksum computed by Scapy
packet = IP(dst="10.11.12.13", src="10.11.12.14")/UDP()/DNS()
packet = IP(raw(packet))  # Build packet (automatically done when sending)
checksum_scapy = packet[UDP].chksum

# Set the UDP checksum to 0 and compute the checksum 'manually'
packet = IP(dst="10.11.12.13", src="10.11.12.14")/UDP(chksum=0)/DNS()
packet_raw = raw(packet)
udp_raw = packet_raw[20:]
# in4_chksum is used to automatically build a pseudo-header
chksum = in4_chksum(socket.IPPROTO_UDP, packet[IP], udp_raw)  # For more infos, call "help(in4_chksum)"

assert(checksum_scapy == chksum)
```
> ## python3 dpdkreply_test1.py

```
[root@centos7 reasm]# python3 dpdkreply_test1.py --pcap  udp.pcap 
udp.pcap
###[ IPv6 ]### 
  version   = 6
  tc        = 0
  fl        = 0
  plen      = 1516
  nh        = UDP
  hlim      = 63
  src       = 2001:db8::a0a:6752
  dst       = 2001:db8::a0a:6751
###[ UDP ]### 
     sport     = webcache
     dport     = 33130
     len       = 1516
     chksum    = 0x4421
###[ Raw ]### 
        load      = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'

csum: 0x4421,origin_cksum : 0x4421
[root@centos7 reasm]# 
```

# 构造udp分片

```
def send_udp_frag3():
    eth=Ether(src=node81_mac, dst=ndoe251_mac,type =ETH_P_IPV6)
    ip6 = IPv6(src = src_ipv6, dst = dst_ipv6, nh = 44)
    udp_hdr_len = 8
    udp_data_len = 2048
    #udp_data_len = 1488
    total = udp_hdr_len + udp_data_len
    sec_of  = (total/2)&~7
    len1 = sec_of - udp_hdr_len
    len2 =  total - sec_of
    fisrt_of = 0
    payload1=packet.Raw("a"*len1) 
    payload2=packet.Raw("a"*len2) 
    udp=UDP(sport=8080, dport=5080, len=udp_data_len + udp_hdr_len, chksum=0x0)
    total_payload = packet.Raw("a"*udp_data_len)
    packet_raw = raw(ip6/udp/total_payload)
    udp_raw = packet_raw[40:]
    udp.chksum = 0
    csum = in6_chksum(socket.IPPROTO_UDP, ip6,udp_raw)
    frag_1 = IPv6ExtHdrFragment(nh = 17,id= 0x0000c911, offset = 0, m = 1)
    frag_2 = IPv6ExtHdrFragment(nh = 17,id= 0x0000c911, offset = sec_of/8, m = 0)
    udp.chksum = csum
    pkt1 = ip6/frag_1/udp/payload1
    sendp(eth/pkt1,verbose=0,iface='enahisic2i3')
    pkt2 = ip6/frag_2/payload2
    sendp(eth/pkt2,verbose=0,iface='enahisic2i3')
```


#   tcpdump bad udp cksum

```
tcpdump -i  enp5s0 ip6 src 2001:db8::a0a:6752 and udp -env

23:18:32.499715 48:57:02:64:ea:1e > 44:a1:91:a4:9c:0b, ethertype IPv6 (0x86dd), length 1462: (hlim 64, next-header UDP (17) payload length: 1408) 2001:db8::a0a:6752.webcache > 2001:db8::a0a:6751.onscreen: [bad udp cksum 0xb1b3 -> 0x3d16!] UDP, length 1400
23:28:36.399777 48:57:02:64:ea:1e > 44:a1:91:a4:9c:0b, ethertype IPv6 (0x86dd), length 1462: (hlim 64, next-header UDP (17) payload length: 1408) 2001:db8::a0a:6752.webcache > 2001:db8::a0a:6751.onscreen: [udp sum ok] UDP, length 1400
```

```
tcpdump -i  enp5s0 ip6 src 2001:db8::a0a:6752 or  ip6 src 2001:db8::a0a:6751  and udp -env
```