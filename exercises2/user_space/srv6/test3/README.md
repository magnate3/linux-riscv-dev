
[p4srv6](https://github.com/ebiken/p4srv6/blob/c5049a80ba366f0cacf20b8bfb88b21540150383/archive/p4-14/demo/DEMO.md)    

# compile   
```
root@ubuntux86:# pwd
/work/ovs_p4/p4c_demo/p4-srv6-demo/p4srv6/archive/p4-14/p4src
root@ubuntux86:# 
 p4c -x p4-14 p4srv6.p4
```

# create topo


```
root@ubuntux86:# pwd
/work/ovs_p4/p4c_demo/p4-srv6-demo/p4srv6
root@ubuntux86:# ./archive/tools/namespace-hosts.sh -c
create_network
ip netns add host0
ip netns add host1
ip link add veth0 type veth peer name vtap0
ip link add veth1 type veth peer name vtap1
ip link add vtap102 type veth peer name vtap103
ip link set dev vtap102 up
ip link set dev vtap103 up
ip link set veth0 netns host0
ip link set veth1 netns host1
ip link set dev vtap0 up
ip link set dev vtap1 up
ip netns exec host0 ip link set veth0 up
ip netns exec host0 ifconfig lo up
ip netns exec host1 ip link set veth1 up
ip netns exec host1 ifconfig lo up
ip netns exec host0 ip addr add 172.20.0.1/24 dev veth0
ip netns exec host1 ip addr add 172.20.0.2/24 dev veth1
ip netns exec host0 ip -6 addr add db8::1/64 dev veth0
ip netns exec host1 ip -6 addr add db8::2/64 dev veth1
ip link set dev vtap0 up
ip link set dev vtap1 up
root@ubuntux86:# 
```

# run switch


```
root@ubuntux86:# pwd
/work/ovs_p4/p4c_demo/p4-srv6-demo/p4srv6/archive/p4-14/p4src
root@ubuntux86:# ls
include  p4srv6.json  p4srv6.p4  p4srv6.p4i
root@ubuntux86:# simple_switch p4srv6.json -i 0@vtap0 -i 1@vtap1 -i 2@vtap102 -i 3@vtap103 \
> --log-console -L debug -- nanolog ipc:///tmp/bm-0-log.ipc --notifications-addr \
> ipc:///tmp/bmv2-0-notifications.ipc
Calling target program-options parser
unrecognised option '--notifications-addr'
Target parser returned an error
[10:05:30.495] [bmv2] [D] [thread 3205] Set default default entry for table 'fwd': NoAction - 
[10:05:30.495] [bmv2] [D] [thread 3205] Set default default entry for table 'gtpu_v6': NoAction - 
[10:05:30.495] [bmv2] [D] [thread 3205] Set default default entry for table 'srv6_localsid': NoAction - 
Adding interface vtap0 as port 0
[10:05:30.498] [bmv2] [D] [thread 3205] Adding interface vtap0 as port 0
Adding interface vtap1 as port 1
[10:05:30.527] [bmv2] [D] [thread 3205] Adding interface vtap1 as port 1
Adding interface vtap102 as port 2
[10:05:30.550] [bmv2] [D] [thread 3205] Adding interface vtap102 as port 2
Adding interface vtap103 as port 3
[10:05:30.583] [bmv2] [D] [thread 3205] Adding interface vtap103 as port 3
[10:05:30.615] [bmv2] [I] [thread 3205] Starting Thrift server on port 9090
[10:05:30.615] [bmv2] [I] [thread 3205] Thrift server was started
```

# 流表


```
root@ubuntux86:# cat srv6.txt 
>> RuntimeCmd: help table_add
>> Add entry to a match table:
>>   table_add <table name> <action name> <match fields> => <action parameters> [priority]
RuntimeCmd:
//table_add fwd forward 0 => 1
//table_add fwd forward 1 => 0
table_add fwd forward 0 => 2
table_add fwd forward 2 => 0
table_add fwd forward 1 => 3
table_add fwd forward 3 => 1

table_add srv6_localsid srv6_T_Insert1 db8::2 => db8::11
table_add srv6_localsid srv6_T_Insert2 db8::2 => db8::21 db8::22
table_add srv6_localsid srv6_T_Insert3 db8::2 => db8::31 db8::32 db8::33
>> srcAddr=db8::1:11, sid0=db8::11
table_add srv6_localsid srv6_T_Encaps1 db8::2 => db8::1:11 db8::11
table_add srv6_localsid srv6_T_Encaps2 db8::2 => db8::1:11 db8::21 db8::22
table_add srv6_localsid srv6_T_Encaps3 db8::2 => db8::1:11 db8::31 db8::32 db8::33

>> srcAddr=db8::1:11, sid0=db8::11
table_add srv6_localsid srv6_End_M_GTP6_D3 db8::2 => db8::1:11 db8::31 db8::32 db8::33
table_add srv6_localsid srv6_End_M_GTP6_D3 db8::2:2 => db8::1:11 db8::31 db8::32 db8::33
root@ubuntux86:# 
```

```
simple_switch_CLI  < srv6.txt 
```

# 测试2

##  Network Topology

```Text
 
                    +--------------+          +-------------+
                    |   p4 switch  |          |   routerB   |
   host1 veth1 -- vethA1        vethAB  --  vethBA       vethB2 -- veth2 host2
                    |              |          |             |
                    +--------------+          +-------------+
```
+ 交换机启动    

```
simple_switch p4srv6.json -i 0@vethA1 -i 1@vethAB --log-console -L debug -- nanolog ipc:///tmp/bm-0-log.ipc --notifications-addr ipc:///tmp/bmv2-0-notifications.ipc
```

+ 邻居表    

lladdr 26:bb:7e:03:c0:62 是vethA1的mac     
```
root@ubuntux86:# ip netns exec host1 ip n del fc00:a::a dev veth1 
root@ubuntux86:# ip netns exec host1 ip -6 neigh add  fc00:a::a  lladdr 26:bb:7e:03:c0:62 nud permanent dev veth1
```



+ vethAB配置ip    

```
root@ubuntux86:# ip a add fc00:00ab::a/64 dev vethAB
```

+ rule    
32:90:78:99:c1:32是vethBA（routerB）的mac    
1e:43:58:05:4d:e5是veth1（host1）的mac     
```
root@ubuntux86:# cat srv6.txt5
table_add fwd forward 0 => 1  0x32907899c132
table_add fwd forward 1 => 0  0x1e4358054de5
table_add srv6_localsid  srv6_T_Encaps1 fc00:000b::10 => fc00:000a::a  fc00:00ab::b
table_add srv6_localsid  srv6_End_DT6  fc00:00ab::a => 
root@ubuntux86:# 
```

## ping
+ 1
```
root@ubuntux86:# ./p4-srv6.sh  -c
ip netns add host1
ip netns add routerA
ip netns add routerB
ip netns add host2
ip link add name veth1 type veth peer name vethA1
ip link set veth1 netns host1
ip link add name vethAB type veth peer name vethBA
ip link set vethBA netns routerB
ip link add name veth2 type veth peer name vethB2
ip link set veth2 netns host2
ip link set vethB2 netns routerB
ip netns exec host1 ip link set lo up
ip netns exec host1 ip ad add fc00:000a::10/64 dev veth1
ip netns exec host1 ifconfig veth1 hw ether 1e:43:58:05:4d:e5
ip netns exec host1 ip -6 neigh add fc00:a::a lladdr 26:bb:7e:03:c0:62 nud permanent dev veth1
ip netns exec host1 ip link set veth1 up
ip netns exec host1 ip -6 route add fc00::/16 via fc00:000a::a
ip netns exec routerA ip link set lo up
ip netns exec routerA sysctl net.ipv6.conf.all.forwarding=1
net.ipv6.conf.all.forwarding = 1
ip netns exec routerA sysctl net.ipv6.conf.all.seg6_enabled=1
net.ipv6.conf.all.seg6_enabled = 1
sysctl net.ipv6.conf.vethA1.seg6_enabled=1
net.ipv6.conf.vethA1.seg6_enabled = 1
ifconfig vethA1 hw ether 26:bb:7e:03:c0:62
ip link set vethA1 up
sysctl net.ipv6.conf.vethAB.seg6_enabled=1
net.ipv6.conf.vethAB.seg6_enabled = 1
ip a add fc00:00ab::a/64 dev vethAB
ip link set vethAB up
ip netns exec routerB ip link set lo up
ip netns exec routerB sysctl net.ipv6.conf.all.forwarding=1
net.ipv6.conf.all.forwarding = 1
ip netns exec routerB sysctl net.ipv6.conf.all.seg6_enabled=1
net.ipv6.conf.all.seg6_enabled = 1
ip netns exec routerB sysctl net.ipv6.conf.vethB2.seg6_enabled=1
net.ipv6.conf.vethB2.seg6_enabled = 1
ip netns exec routerB ip ad add fc00:000b::b/64 dev vethB2
ip netns exec routerB ip link set vethB2 up
ip netns exec routerB sysctl net.ipv6.conf.vethBA.seg6_enabled=1
net.ipv6.conf.vethBA.seg6_enabled = 1
ip netns exec routerB ip ad add fc00:00ab::b/64 dev vethBA
ip netns exec routerB ifconfig vethBA hw ether 32:90:78:99:c1:32
ip netns exec routerB ip link set vethBA up
ip netns exec routerB ip -6 route add fc00:000a::/64 encap seg6 mode encap segs fc00:00ab::a dev vethB2
ip netns exec host2 ip link set lo up
ip netns exec host2 ip ad add fc00:000b::10/64 dev veth2
ip netns exec host2 ifconfig veth2 hw ether 16:b5:08:6b:96:25
ip netns exec host2 ip link set veth2 up
ip netns exec host2 ip -6 route add fc00::/16 via fc00:000b::b
root@ubuntux86:# 
```
+ 2
```
root@ubuntux86:# simple_switch p4srv6.json -i 0@vethA1 -i 1@vethAB --log-console -L debug -- nanolog ipc:///tmp/bm-0-log.ipc --notifications-addr ipc:///tmp/bmv2-0-notifications.ipc
Calling target program-options parser
```
+ 3
```
root@ubuntux86:# simple_switch_CLI  < srv6.txt5
Obtaining JSON from switch...
Done
Control utility for runtime P4 table manipulation
RuntimeCmd: Adding entry to exact match table fwd
match key:           EXACT-00:00
action:              forward
runtime data:        00:01      32:90:78:99:c1:32
Entry has been added with handle 0
RuntimeCmd: Adding entry to exact match table fwd
match key:           EXACT-00:01
action:              forward
runtime data:        00:00      1e:43:58:05:4d:e5
Entry has been added with handle 1
RuntimeCmd: Adding entry to exact match table srv6_localsid
match key:           EXACT-fc:00:00:0b:00:00:00:00:00:00:00:00:00:00:00:10
action:              srv6_T_Encaps1
runtime data:        fc:00:00:0a:00:00:00:00:00:00:00:00:00:00:00:0a    fc:00:00:ab:00:00:00:00:00:00:00:00:00:00:00:0b
Entry has been added with handle 0
RuntimeCmd: Adding entry to exact match table srv6_localsid
match key:           EXACT-fc:00:00:ab:00:00:00:00:00:00:00:00:00:00:00:0a
action:              srv6_End_DT6
runtime data:        
Entry has been added with handle 1
RuntimeCmd: 
root@ubuntux86:#
```
 
```
root@ubuntux86:# ip netns exec host1 ping fc00:000b::10
PING fc00:000b::10(fc00:b::10) 56 data bytes
64 bytes from fc00:b::10: icmp_seq=1 ttl=64 time=1.58 ms
64 bytes from fc00:b::10: icmp_seq=2 ttl=64 time=2.82 ms
64 bytes from fc00:b::10: icmp_seq=3 ttl=64 time=2.62 ms
64 bytes from fc00:b::10: icmp_seq=4 ttl=64 time=2.44 ms
^C
--- fc00:000b::10 ping statistics ---
5 packets transmitted, 5 received, 0% packet loss, time 4008ms
rtt min/avg/max/mdev = 1.583/2.443/2.820/0.448 ms
root@ubuntux86:# 
```

```
root@ubuntux86:# ip netns exec host2 ping fc00:000a::10
PING fc00:000a::10(fc00:a::10) 56 data bytes
64 bytes from fc00:a::10: icmp_seq=1 ttl=63 time=1.12 ms
64 bytes from fc00:a::10: icmp_seq=2 ttl=63 time=2.01 ms
64 bytes from fc00:a::10: icmp_seq=3 ttl=63 time=2.03 ms
^C
--- fc00:000a::10 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 1.121/1.722/2.034/0.425 ms
root@ubuntux86:# 
```

# 代码

IP_PROTOCOLS_SRV6 : parse_srh  
