


```
root@ubuntux86:# ./xdp_stats -d   enp0s31f6
WARN: Failed to open bpf map file:/sys/fs/bpf/enp0s31f6/xdp_stats_map err(2):No such file or directory
root@ubuntux86:# 
```


```
xdp_loader.c:27:static const char *default_filename = "xdp_prog_kern.o";

```

# /work/ovs_p4/xdp-tutorial/basic04-pinning-maps


```
root@ubuntux86:# ./xdp_loader  load  -S     --dev enp0s31f6 --progname xdp_pass_func
libbpf: elf: skipping unrecognized data section(7) xdp_metadata
libbpf: prog 'xdp_pass': BPF program load failed: Invalid argument
libbpf: prog 'xdp_pass': failed to load: -22
libbpf: failed to load object 'xdp-dispatcher.o'
libbpf: elf: skipping unrecognized data section(7) xdp_metadata
libbpf: elf: skipping unrecognized data section(7) xdp_metadata
libbpf: elf: skipping unrecognized data section(7) xdp_metadata
Success: Loaded BPF-object(xdp_prog_kern.o) and used program(xdp_pass_func)
 - XDP prog attached on device:enp0s31f6(ifindex:2)
 - Unpinning (remove) prev maps in /sys/fs/bpf/enp0s31f6/
 - Pinning maps in /sys/fs/bpf/enp0s31f6/
root@ubuntux86:# ls /sys/fs/bpf/enp0s31f6/
xdp_stats_map
```
 bpf_object__pin_maps  /sys/fs/bpf/enp0s31f6/     
 
 ```
 root@ubuntux86:# cat /sys/fs/bpf/enp0s31f6/xdp_stats_map 
# WARNING!! The output is for debug purpose only
# WARNING!! The output format will change
0: {
        cpu0: {0,0,}
        cpu1: {0,0,}
        cpu2: {0,0,}
        cpu3: {0,0,}
        cpu4: {0,0,}
        cpu5: {0,0,}
        cpu6: {0,0,}
        cpu7: {0,0,}
        cpu8: {0,0,}
        cpu9: {0,0,}
        cpu10: {0,0,}
        cpu11: {0,0,}
        cpu12: {0,0,}
        cpu13: {0,0,}
        cpu14: {0,0,}
        cpu15: {0,0,}
        cpu16: {0,0,}
        cpu17: {0,0,}
        cpu18: {0,0,}
        cpu19: {0,0,}
}
 ```

```
root@ubuntux86:# ./xdp_stats -d   enp0s31f6

Collecting stats from BPF map
 - BPF map (bpf_map_type:6) id:71 name:xdp_stats_map key_size:4 value_size:16 max_entries:5
XDP-action  
XDP_ABORTED            0 pkts (         0 pps)           0 Kbytes (     0 Mbits/s) period:0.250252
XDP_DROP               0 pkts (         0 pps)           0 Kbytes (     0 Mbits/s) period:0.250179
XDP_PASS               0 pkts (         0 pps)           0 Kbytes (     0 Mbits/s) period:0.250179
XDP_TX                 0 pkts (         0 pps)           0 Kbytes (     0 Mbits/s) period:0.250178
XDP_REDIRECT           0 pkts (         0 pps)           0 Kbytes (     0 Mbits/s) period:0.250178

^C
root@ubuntux86:# 
```

```
2: enp0s31f6: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 xdpgeneric/id:137 qdisc fq_codel state UP group default qlen 1000
    link/ether 30:d0:42:fa:ae:11 brd ff:ff:ff:ff:ff:ff
    inet 192.168.5.82/24 brd 192.168.5.255 scope global noprefixroute enp0s31f6
       valid_lft forever preferred_lft forever
    inet 10.11.12.82/24 brd 10.11.12.255 scope global noprefixroute enp0s31f6
       valid_lft forever preferred_lft forever
    inet6 fe80::a222:d618:3432:22df/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
```
卸载    
```
ip l set  enp0s31f6 xdp off
```

# packet03-redirecting
```

root@ubuntux86:# pwd
/work/ovs_p4/xdp-tutorial/packet03-redirecting
root@ubuntux86:# 
root@ubuntux86:# ./xdp-loader  load  -m skb --section  xdp  enp0s31f6 xdp_prog_kern.o
root@ubuntux86:# ./xdp-loader  unload    enp0s31f6  --all
root@ubuntux86:# 
```
id 卸载    

```
2: enp0s31f6: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 xdpgeneric/id:99 qdisc fq_codel state UP group default qlen 1000
    link/ether 30:d0:42:fa:ae:11 brd ff:ff:ff:ff:ff:ff
    inet 192.168.5.82/24 brd 192.168.5.255 scope global noprefixroute enp0s31f6
       valid_lft forever preferred_lft forever
    inet 10.11.12.82/24 brd 10.11.12.255 scope global noprefixroute enp0s31f6
       valid_lft forever preferred_lft forever
    inet6 fe80::a222:d618:3432:22df/64 scope link noprefixroute 
       valid_lft forever preferred_lft foreve
```

```
root@ubuntux86:# ./xdp-loader  unload    enp0s31f6  -i 99
Program with ID 99 not loaded on enp0s31f6
root@ubuntux86:# 
```