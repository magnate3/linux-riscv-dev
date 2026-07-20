
#   hw-tc-offload
```
root@dirk:/home/dirk# ethtool -i  ens14f0
driver: mlx5_core
version: 4.9-5.1.0
firmware-version: 14.27.1016 (MT_2420110004)
expansion-rom-version: 
bus-info: 0000:98:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
root@dirk:/home/dirk#  ethtool -k ens14f0 | grep hw-tc-offload
hw-tc-offload: off
root@dirk:/home/dirk# 
```

# ens4f1开启sriov

```
root@dirk:/home/dirk# ethtool -i  ens4f1 
driver: mlx5_core
version: 4.9-5.1.0
firmware-version: 14.27.1016 (MT_2420110004)
expansion-rom-version: 
bus-info: 0000:31:00.1
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
```

```
echo 2 > /sys/class/net/ens4f1/device/sriov_numvfs
```
ens4f1的两个vf是enp49s1f2和enp49s1f3：   
```
root@dirk:/home/dirk# ip link show ens4f1
3: ens4f1: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 08:c0:eb:40:8d:f3 brd ff:ff:ff:ff:ff:ff
    vf 0 MAC 00:00:00:00:00:00, spoof checking off, link-state auto, trust off, query_rss off
    vf 1 MAC 00:00:00:00:00:00, spoof checking off, link-state auto, trust off, query_rss off
10: enp49s1f2: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether e6:7d:9f:01:4b:d3 brd ff:ff:ff:ff:ff:ff
11: enp49s1f3: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether be:57:a9:c0:48:13 brd ff:ff:ff:ff:ff:ff
```

```
root@dirk:/home/dirk# ethtool -k ens4f1 | grep hw-tc-offload
hw-tc-offload: off
root@dirk:/home/dirk# ethtool -k enp49s1f2 | grep hw-tc-offload
hw-tc-offload: off
root@dirk:/home/dirk# ethtool -k enp49s1f3 | grep hw-tc-offload
hw-tc-offload: off
root@dirk:/home/dirk# 

```

vf也可以支持hw-tc-offload   
```
root@dirk:/home/dirk# ethtool -K ens4f1 hw-tc-offload on
root@dirk:/home/dirk# ethtool -K enp49s1f2 hw-tc-offload on
root@dirk:/home/dirk# ethtool -k enp49s1f2 | grep hw-tc-offload
hw-tc-offload: on
root@dirk:/home/dirk# ethtool -k ens4f1 | grep hw-tc-offload
hw-tc-offload: on
root@dirk:/home/dirk# 
```

## enp49s1f3的驱动
```
root@dirk:/home/dirk# ethtool -i  enp49s1f3
driver: mlx5_core
version: 4.9-5.1.0
firmware-version: 14.27.1016 (MT_2420110004)
expansion-rom-version: 
bus-info: 0000:31:01.3
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
root@dirk:/home/dirk# 
```
***01 pk 00***   
+ bus-info: 0000:31:01.3   
+ bus-info: 0000:31:00.1   
#  /sys/bus/pci/drivers/mlx5_core/unbind
```
root@dirk:/home/dirk# ethtool -i enp49s1f2
driver: mlx5_core
version: 4.9-5.1.0
firmware-version: 14.27.1016 (MT_2420110004)
expansion-rom-version: 
bus-info: 0000:31:01.2
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
root@dirk:/home/dirk# ethtool -i enp49s1f3
driver: mlx5_core
version: 4.9-5.1.0
firmware-version: 14.27.1016 (MT_2420110004)
expansion-rom-version: 
bus-info: 0000:31:01.3
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
root@dirk:/home/dirk# 
```

```
root@dirk:/home/dirk# echo '0000:31:01.2' >  /sys/bus/pci/drivers/mlx5_core/unbind
root@dirk:/home/dirk# echo '0000:31:01.3' >  /sys/bus/pci/drivers/mlx5_core/unbind
root@dirk:/home/dirk# 
```


#  devlink dev eswitch set pci/0000:31:00.1 mode switchdev

执行后多了ens4f1_0和ens4f1_1两个网卡
```
root@dirk:/home/dirk# devlink dev eswitch set pci/0000:31:00.1 mode switchdev
root@dirk:/home/dirk# ip a
6: ens14f1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
    link/ether 08:c0:eb:3b:35:3d brd ff:ff:ff:ff:ff:ff

12: ens4f1: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether 08:c0:eb:40:8d:f3 brd ff:ff:ff:ff:ff:ff
13: ens4f1_0: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether d2:6d:4b:d4:94:06 brd ff:ff:ff:ff:ff:ff
14: ens4f1_1: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether ca:ae:18:be:3f:6b brd ff:ff:ff:ff:ff:ff
```

#   mlx5e_rep driver
```
root@dirk:/home/dirk# ethtool -i  ens4f1_0
driver: mlx5e_rep
version: 4.15.0-55-generic
firmware-version: 14.27.1016 (MT_2420110004)
expansion-rom-version: 
bus-info: 
supports-statistics: yes
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
root@dirk:/home/dirk# ethtool -i  ens4f1_1
driver: mlx5e_rep
version: 4.15.0-55-generic
firmware-version: 14.27.1016 (MT_2420110004)
expansion-rom-version: 
bus-info: 
supports-statistics: yes
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
root@dirk:/home/dirk# 
```
rep支持hw-tc-offload   
```
root@dirk:/home/dirk# ethtool -k ens4f1_0 | grep hw-tc-offload
hw-tc-offload: on
root@dirk:/home/dirk# ethtool -k ens4f1_1 | grep hw-tc-offload
hw-tc-offload: on
root@dirk:/home/dirk# 
```
ens4f1_0、ens4f1_1不属于vf   
```
root@dirk:/home/dirk# ethtool -k ens4f1_0 | grep hw-tc-offload
hw-tc-offload: on
root@dirk:/home/dirk# ethtool -k ens4f1_1 | grep hw-tc-offload
hw-tc-offload: on
root@dirk:/home/dirk# ip link show ens4f1
12: ens4f1: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 08:c0:eb:40:8d:f3 brd ff:ff:ff:ff:ff:ff
    vf 0 MAC 00:00:00:00:00:00, spoof checking off, link-state disable, trust off, query_rss off
    vf 1 MAC 00:00:00:00:00:00, spoof checking off, link-state disable, trust off, query_rss off
root@dirk:/home/dirk# 
```

#  echo '0000:31:01.2' >  /sys/bus/pci/drivers/mlx5_core/bind

```
root@dirk:/home/dirk# echo '0000:31:01.3' >  /sys/bus/pci/drivers/mlx5_core/bind
root@dirk:/home/dirk# echo '0000:31:01.2' >  /sys/bus/pci/drivers/mlx5_core/bind
root@dirk:/home/dirk# ip a
```

```
12: ens4f1: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether 08:c0:eb:40:8d:f3 brd ff:ff:ff:ff:ff:ff
13: ens4f1_0: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether d2:6d:4b:d4:94:06 brd ff:ff:ff:ff:ff:ff
14: ens4f1_1: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether ca:ae:18:be:3f:6b brd ff:ff:ff:ff:ff:ff
15: enp49s1f3: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether be:57:a9:c0:48:13 brd ff:ff:ff:ff:ff:ff
16: enp49s1f2: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether e6:7d:9f:01:4b:d3 brd ff:ff:ff:ff:ff:ff
```
#  ens4f1网卡reset
```
root@dirk:/home/dirk# echo '0000:31:00.1' >  /sys/bus/pci/drivers/mlx5_core/unbind
root@dirk:/home/dirk# echo '0000:31:00.1' >  /sys/bus/pci/drivers/mlx5_core/bind
root@dirk:/home/dirk# ip a
```