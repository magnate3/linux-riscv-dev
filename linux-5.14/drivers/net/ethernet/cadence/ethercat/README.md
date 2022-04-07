# qemu
```
qemu-system-arm -M xilinx-zynq-a9 -serial /dev/null -serial mon:stdio -display none -kernel my.uImage.p -dtb Prebuilt_functional/my_devicetree.dtb --initrd Prebuilt_functional/ramdisk.img.gz  -net tap,ifname=qtap,script=no,downscript=no   -net nic,model=cadence_gem,macaddr=0e:b0:ba:5e:ba:12
```

# insmod  ec_master main_devices=0e:b0:ba:5e:ba:12
```
root@zedboard-zynq7:~# ls
ec_master.ko  macb_main.ko
root@zedboard-zynq7:~# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: tunl0@NONE: <NOARP> mtu 1480 qdisc noop state DOWN group default qlen 1000
    link/ipip 0.0.0.0 brd 0.0.0.0
3: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether 0e:b0:ba:5e:ba:12 brd ff:ff:ff:ff:ff:ff
    inet 169.254.1.2/16 scope global eth0
       valid_lft forever preferred_lft forever
root@zedboard-zynq7:~#  modprobe ec_master main_devices=0e:b0:ba:5e:ba:12
modprobe: can't change directory to '5.15.0-axiom': No such file or directory
root@zedboard-zynq7:~# ls
ec_master.ko  macb_main.ko
root@zedboard-zynq7:~# insmod  ec_master.ko main_devices=0e:b0:ba:5e:ba:12
[   84.248442] ec_master: loading out-of-tree module taints kernel.
[   84.261179] EtherCAT: Master driver 1.5.2 unknown
[   84.262642] EtherCAT: 1 master waiting for devices.
```

#  insmod  macb_main.ko   
```
root@zedboard-zynq7:~# lsmod | grep macb  
macb 49152 0 - Live 0xbf007000
phylink 24576 1 macb, Live 0xbf000000
root@zedboard-zynq7:~# rmmod macb
[   98.105302] macb e000b000.ethernet eth0: Link is Down
root@zedboard-zynq7:~# insmod  macb_main.ko                               
[  103.837861] macb e000b000.ethernet: not need to register interrupt 
[  103.841812] libphy: MACB_mii_bus: probed
[  103.853059] macb e000b000.ethernet eth0: Cadence GEM rev 0x00020118 at 0xe000b000 irq 36 (0e:b0:ba:5e:ba:12)
[  103.853254] EtherCAT: Accepting 0E:B0:BA:5E:BA:12 as main device for master 0.
root@zedboard-zynq7:~# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: tunl0@NONE: <NOARP> mtu 1480 qdisc noop state DOWN group default qlen 1000
    link/ipip 0.0.0.0 brd 0.0.0.0
4: ecm0: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether 0e:b0:ba:5e:ba:12 brd ff:ff:ff:ff:ff:ff
root@zedboard-zynq7:~# ip link set ecmo up
Cannot find device "ecmo"
root@zedboard-zynq7:~# ip link set ecmo up
Cannot find device "ecmo"
root@zedboard-zynq7:~# ip link set ecm0 up
Cannot find device "ecm0"
root@zedboard-zynq7:~# ip link set ecm0 up
Cannot find device "ecm0"
root@zedboard-zynq7:~# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: tunl0@NONE: <NOARP> mtu 1480 qdisc noop state DOWN group default qlen 1000
    link/ipip 0.0.0.0 brd 0.0.0.0
4: ecm0: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether 0e:b0:ba:5e:ba:12 brd ff:ff:ff:ff:ff:ff
root@zedboard-zynq7:~# 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/ethercat/pic/old.png)

# resotre /lib/modules/4.6.0-xilinx/macb.ko
##  rmmod macb_main.ko   and  insmod /lib/modules/4.6.0-xilinx/macb.ko 
```
root@zedboard-zynq7:~# rmmod macb_main.ko 
root@zedboard-zynq7:~# insmod /lib/modules/4.6.0-xilinx/macb.ko 
[  284.654838] macb e000b000.ethernet: not need to register interrupt 
[  284.658665] libphy: MACB_mii_bus: probed
[  284.684762] macb e000b000.ethernet eth0: Cadence GEM rev 0x00020118 at 0xe000b000 irq 36 (0e:b0:ba:5e:ba:12)
root@zedboard-zynq7:~# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: tunl0@NONE: <NOARP> mtu 1480 qdisc noop state DOWN group default qlen 1000
    link/ipip 0.0.0.0 brd 0.0.0.0
5: eth0: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether 0e:b0:ba:5e:ba:12 brd ff:ff:ff:ff:ff:ff
root@zedboard-zynq7:~# ip link set eth0 up
[  293.480697] macb e000b000.ethernet eth0: PHY [e000b000.ethernet-ffffffff:00] driver [Marvell 88E1111] (irq=POLL)
[  293.481818] macb e000b000.ethernet eth0: configuring for phy/rgmii-id link mode
[  293.483822] Marvell 88E1111 e000b000.ethernet-ffffffff:00: Downshift occurred from negotiated speed 1Gbps to actual speed 100Mbps, check cabling!
[  293.484511] macb e000b000.ethernet eth0: unable to generate target frequency: 25000000 Hz
[  293.486898] macb e000b000.ethernet eth0: Link is Up - 100Mbps/Full - flow control tx
root@zedboard-zynq7:~# ip addr add 169.254.1.2/16 dev eth0
root@zedboard-zynq7:~# ping 169.254.1.1
PING 169.254.1.1 (169.254.1.1): 56 data bytes
[  297.627705] macb: *********** recv complete call gem_rx ************
[  297.627767] macb: *********** napi_schedul raise rx softirq  ************
[  297.629453] macb: htons(ETH_P_ARP) 1544 =? skb->protocol  1544 
[  297.629530] macb: before skb_reset_network_header, the  skb->network_header 0 
[  297.629536] macb: after skb_reset_network_header, the  skb->network_header 80 
[  297.629586] macb: htons(ARPOP_REQUEST) 256 =? arp_hdr(skb)->ar_op  512 
[  297.631200] macb: htons(ETH_P_ARP) 1544 =? skb->protocol  8 
64 bytes from 169.254.1.1: seq=0 ttl=64 time=14.847 ms
^C
--- 169.254.1.1 ping statistics ---
1 packets transmitted, 1 packets received, 0% packet loss
round-trip min/avg/max = 14.847/14.847/14.847 ms
```
# solve  Cannot find device "ecm0"

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/ethercat/pic/new.png)