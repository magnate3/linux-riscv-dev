

# ipv6 side

+ 加载v6_side   
```
root@ubuntux86:# ./xdp_loader --progsec v6_side --filename xdp_prog_kern.o --dev veth6
```

```
ip l set dev veth6 xdp off
```

+ ping6   
```
root@ubuntux86:# ip netns exec nat64 ping6 64:ff9b::0a00:0102
PING 64:ff9b::0a00:0102(64:ff9b::a00:102) 56 data bytes

```

+ tcpdump     
10.0.1.1 > 10.0.1.2   
```
root@ubuntux86:# tcpdump -i veth6   -env
tcpdump: listening on veth6, link-type EN10MB (Ethernet), capture size 262144 bytes
15:30:01.909596 aa:80:e3:5e:3d:0b > aa:70:e3:5e:3d:0a, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 0, offset 0, flags [DF], proto ICMP (1), length 84)
    10.0.1.1 > 10.0.1.2: ICMP echo request, id 7464, seq 90, length 64 (wrong icmp cksum da7d (->9975)!)
15:30:02.933653 aa:80:e3:5e:3d:0b > aa:70:e3:5e:3d:0a, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 0, offset 0, flags [DF], proto ICMP (1), length 84)
    10.0.1.1 > 10.0.1.2: ICMP echo request, id 7464, seq 91, length 64 (wrong icmp cksum da7c (->a016)!)
15:30:03.957607 aa:80:e3:5e:3d:0b > aa:70:e3:5e:3d:0a, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 0, offset 0, flags [DF], proto ICMP (1), length 84)
    10.0.1.1 > 10.0.1.2: ICMP echo request, id 7464, seq 92, length 64 (wrong icmp cksum da7b (->db8)!)
15:30:04.981610 aa:80:e3:5e:3d:0b > aa:70:e3:5e:3d:0a, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 0, offset 0, flags [DF], proto ICMP (1), length 84)
```

# reference

[nat-and-nat64](https://github.com/steps-to-reproduce/nat-and-nat64)