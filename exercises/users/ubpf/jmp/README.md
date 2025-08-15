 
# Setup

[原始项目](https://github.com/adhamkhalil/eBPF/tree/447406aa5fbbf43ab02543450d5ff26610aab5ea)

After cloning the repository:
```
git submodule update --init
cd libbpf/src
(possibly apt install libelf-dev and gcc-multilib)
make

```

# make and run
更改网卡   
```
#define DEFAULT_IFACE "enx00e04c3662aa"
DEV=enx00e04c3662aa
```

+ 1 make   
```
root@ubuntux86:# make
clang -target bpf -g -O3 -I../libbpf/src  -c -o xdp_network_tracker.o xdp_network_tracker.c
clang -static -g -O3 -I../libbpf/src  -o xdp_network_tracker_user xdp_network_tracker_user.c -L../libbpf/src -lbpf -lelf -lz
```
+ 2 make load   
```
root@ubuntux86:# make load
mount -t bpf none /sys/fs/bpf
bpftool map create /sys/fs/bpf/map_illegal_domains type hash \
        name map_illegal_domains key 4 value 4 entries 2 flags 1
bpftool map create /sys/fs/bpf/query type hash \
        name query key 4 value 56 entries 100 flags 1
./xdp_network_tracker_user
map_xdp_progs entry key -> name -> fd
: 0 -> xdp-packet-preprocess -> 12
map_xdp_progs entry key -> name -> fd
: 1 -> xdp-packet-validation -> 13
Program attached and running.
Press Ctrl-C to stop followed by make unload
^Cmake: *** [Makefile:46: load] Interrupt
```
+ 3 make clean
```
root@ubuntux86:# make clean
umount /sys/fs/bpf/ 2>/dev/null || true
rm -f xdp_network_tracker.o xdp_network_tracker_user
root@ubuntux86:# 
```
