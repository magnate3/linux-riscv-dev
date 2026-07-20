# qemu
```
root@ubuntux86:# qemu-system-x86_64      --version
QEMU emulator version 6.1.0
Copyright (c) 2003-2021 Fabrice Bellard and the QEMU Project developers
root@ubuntux86:# 
```

## network

Setting up Qemu with a tap interface. There are two parts to networking within QEMU:   

The virtual network device that is provided to the guest (e.g. a PCI network card).
The network backend that interacts with the emulated NIC (e.g. puts packets onto the host's network).  
+ 1) Example: User mode network
User mode networking allows the guest to connect back to the outside world through TCP, UDP etc. ICMP Ping is not allowed. Also connections from host to guest are not allowed unless using port forwarding.
```
$ qemu-system-i386 -cdrom Core-current.iso -boot d -netdev user,id=mynet0,hostfwd=tcp::8080-:80 -device e1000,netdev=mynet0
```
`-netdev user,id=mynet0,hostfwd=tcp::8080-:80`   

Create the user mode network backend having id mynet0. Redirect incoming tcp connections on host port 8080 to guest port 80. The syntax is hostfwd=[tcp|udp]:[hostaddr]:hostport-[guestaddr]:guestport   

`-device e1000,netdev=mynet0`   
Create a NIC (model e1000) and connect to mynet0 backend created by the previous parameter

+ 2) Example: Tap network
TAP network overcomes all of the limitations of user mode networking, but requires a tap to be setup before running qemu.     
```
$ sudo qemu-system-i386 -cdrom Core-current.iso -boot d -netdev tap,id=mynet0,ifname=tap0,script=no,downscript=no -device e1000,netdev=mynet0,mac=52:55:00:d1:55:01
-netdev tap,id=mynet0,ifname=tap0,script=no,downscript=no
```
Create a tap network backend with id mynet0. This will connect to a tap interface tap0 which must be already setup. Do not use any network configuration scripts.

`-device e1000,netdev=mynet0,mac=52:55:00:d1:55:01`
Create a NIC (model e1000) and connect to mynet0 backend created by the previous parameter. Also specify a mac address for the NIC.   

## e1000

```
  /* Possible combinations:
     *
     *  1. Old way:   -net nic,model=e1000,vlan=1 -net tap,vlan=1
     *  2. Semi-new:  -device e1000,vlan=1        -net tap,vlan=1
     *  3. Best way:  -netdev type=tap,id=netdev1 -device e1000,id=netdev1
     *
     * NB, no support for -netdev without use of -device
     */
```

# host
```
root@ubuntux86:# ip tuntap show
tap1: tap one_queue vnet_hdr persist user 0
tap0: tap one_queue vnet_hdr persist user 0
root@ubuntux86:# 
```

```
qemu-system-x86_64 -enable-kvm -smp 4 -m 8G -cpu IvyBridge -hda ubuntu-14.04-server-cloudimg-amd64-disk1.img -netdev tap,id=tap1,ifname=tap1,script=no,downscript=no,vhost=off -device e1000,id=e0,netdev=tap1,mac=52:55:00:d1:55:02 -nographic
```
网桥   
```
root@ubuntux86:# brctl show br1
bridge name     bridge id               STP enabled     interfaces
br1             8000.36299cfdb449       no              tap0
                                                        tap1
root@ubuntux86:# ip a sh br1 
8: br1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether 36:29:9c:fd:b4:49 brd ff:ff:ff:ff:ff:ff
    inet 192.168.11.33/24 scope global br1
       valid_lft forever preferred_lft forever
    inet6 fe80::3429:9cff:fefd:b449/64 scope link 
       valid_lft forever preferred_lft forever
root@ubuntux86:# 
```

# vm

```
root@ubuntu:~# ethtool -i  eth0
driver: e1000
version: 7.3.21-k8-NAPI
firmware-version: 
bus-info: 0000:00:03.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: yes
supports-register-dump: yes
supports-priv-flags: no
root@ubuntu:~#
```

ping   

```
root@ubuntu:~# ping 192.168.11.33
PING 192.168.11.33 (192.168.11.33) 56(84) bytes of data.
64 bytes from 192.168.11.33: icmp_seq=1 ttl=64 time=2.31 ms
64 bytes from 192.168.11.33: icmp_seq=2 ttl=64 time=0.564 ms
^C
--- 192.168.11.33 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1000ms
rtt min/avg/max/mdev = 0.564/1.441/2.318/0.877 ms
root@ubuntu:~# 
```