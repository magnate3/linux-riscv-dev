 # make
 
 ```
 [root@centos7 udp]# make
make -C /lib/modules/4.14.0-115.el7a.0.1.aarch64/build \
M=/root/programming/kernel/udp modules
make[1]: Entering directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
  CC [M]  /root/programming/kernel/udp/sendUDPWithKernelModule.o
/root/programming/kernel/udp/sendUDPWithKernelModule.c: In function ‘send_by_skb’:
/root/programming/kernel/udp/sendUDPWithKernelModule.c:105:46: warning: format ‘%d’ expects argument of type ‘int’, but argument 2 has type ‘long unsigned int’ [-Wformat=]
         printk("iphdr : %d\n", sizeof(struct iphdr));
                                              ^
/root/programming/kernel/udp/sendUDPWithKernelModule.c:106:47: warning: format ‘%d’ expects argument of type ‘int’, but argument 2 has type ‘long unsigned int’ [-Wformat=]
         printk("udphdr : %d\n", sizeof(struct udphdr));
                                               ^
/root/programming/kernel/udp/sendUDPWithKernelModule.c:173:20: warning: assignment from incompatible pointer type [enabled by default]
         eth_header = (struct ehthdr *)skb_push(skb, ETH_HLEN);
                    ^
  Building modules, stage 2.
  MODPOST 3 modules
  CC      /root/programming/kernel/udp/sendUDPWithKernelModule.mod.o
  LD [M]  /root/programming/kernel/udp/sendUDPWithKernelModule.ko
make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
 ```
 
 # run
 
 ```
 make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
[root@centos7 udp]# insmod  sendUDPWithKernelModule.ko
[root@centos7 udp]# rmmod  sendUDPWithKernelModule.ko
[root@centos7 udp]# dmesg | tail -n 20
[  775.822435] skb_len  : 60
[  775.824963] payload:         hello world
[  775.828880] send packet by skb success.
[  935.001499] testmod kernel module removed!
[  967.306317] testmod kernel module load!
[  967.310143] iphdr    : 20
[  967.312504] udphdr   : 8
[  967.314852] data_len: 16
[  967.317374] skb_len  : 60
[  967.319898] payload:         hello world
[  967.323820] send packet by skb success.
[ 1068.216004] testmod kernel module removed!
[ 1101.872484] testmod kernel module load!
[ 1101.876310] iphdr    : 20
[ 1101.878668] udphdr   : 8
[ 1101.881018] data_len: 16
[ 1101.883540] skb_len  : 60
[ 1101.886063] payload:         hello world
[ 1101.889979] send packet by skb success.
[ 1109.856519] testmod kernel module removed!
[root@centos7 udp]# 
 ```
 
 # tcpdump in 10.10.16.82
 
 ```
 root@ubuntu:~# tcpdump -i enahisic2i0 udp and host 10.10.16.251 -eennvv 
tcpdump: listening on enahisic2i0, link-type EN10MB (Ethernet), capture size 262144 bytes
17:22:15.144926 b0:08:75:5f:b8:5b > 48:57:02:64:e7:ab, ethertype IPv4 (0x0800), length 60: (tos 0x0, ttl 64, id 0, offset 0, flags [none], proto UDP (17), length 28)
    10.10.16.251.31900 > 10.10.16.82.31900: truncated-udplength 0

 ```