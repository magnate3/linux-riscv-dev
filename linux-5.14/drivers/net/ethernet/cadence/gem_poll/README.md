 # purpose
  disable  recv and send interrupt to achieve poll recv
  
 # make
 ```
 make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi-  UIMAGE_LOADADDR=0x8000 uImage -j20
 ```
 # run
 
 ```
root@x86:/home/ubuntu# tunctl  -t qtap -u $(whoami) 
Set 'qtap' persistent and owned by uid 0
root@x86:/home/ubuntu# ip link set dev qtap up
root@x86:/home/ubuntu# ip addr add 169.254.1.1/16 dev qtap
 ```
 
 ```
 qemu-system-arm -M xilinx-zynq-a9 -serial /dev/null -serial mon:stdio -display none -kernel my.uImage.p -dtb Prebuilt_functional/my_devicetree.dtb --initrd Prebuilt_functional/ramdisk.img.gz  -net tap,ifname=qtap,script=no,downscript=no   -net nic,model=cadence_gem,macaddr=0e:b0:ba:5e:ba:12 
 ```
 
 ```
 zedboard-zynq7 login: root
root@zedboard-zynq7:~# ls
root@zedboard-zynq7:~# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
[   12.150342] random: fast init done
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether 0e:b0:ba:5e:ba:12 brd ff:ff:ff:ff:ff:ff
3: tunl0@NONE: <NOARP> mtu 1480 qdisc noop state DOWN group default qlen 1000
    link/ipip 0.0.0.0 brd 0.0.0.0
root@zedboard-zynq7:~# ip a add 169.254.1.2/16 dev eth0
root@zedboard-zynq7:~# ping  169.254.1.1
PING 169.254.1.1 (169.254.1.1): 56 data bytes
[   28.229665] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=0 ttl=64 time=9.054 ms
[   28.231379] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=1 ttl=64 time=18.616 ms
[   29.249623] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=2 ttl=64 time=6.358 ms
[   30.239755] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=3 ttl=64 time=25.371 ms
[   31.260026] macb: ************  netif_rx **********
[   31.349965] macb: ************  netif_rx **********
[   31.350600] NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!
[   31.351604] NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!
64 bytes from 169.254.1.1: seq=4 ttl=64 time=12.956 ms
[   32.249933] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=5 ttl=64 time=2.461 ms
[   33.239920] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=6 ttl=64 time=29.248 ms
[   34.269949] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=7 ttl=64 time=18.135 ms
[   35.260064] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=8 ttl=64 time=5.699 ms
[   36.249758] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=9 ttl=64 time=24.362 ms
[   37.269935] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=10 ttl=64 time=13.238 ms
[   38.259921] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=11 ttl=64 time=1.862 ms
QEMU: Terminated
root@x86:/home/ubuntu/QEMU_CPUFreq_Zynq# 

 ```
   ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/gem_poll/pic/poll.jpg)
  