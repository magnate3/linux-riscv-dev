
# tcp-segmentation-offload
```Text
使用 ethtool 工具打开网卡和驱动对 TSO（TCP Segmentation Offload）的支持。如下命令中的参数“$eth”为待调整配置的网卡设备名称，如 eth0，eth1 等。
# ethtool -K $eth tso on
说明：要使用 TSO 功能，物理网卡需同时支持 TCP 校验计算和分散-聚集 (Scatter Gather) 功能。
查看网卡是否支持 TSO：
# ethtool -K $eth
rx-checksumming: on
tx-checksumming: on
scatter-gather: on
tcp-segmentation-offload: on
```

```
root@ubuntu:/home/ubuntu# ethtool -k ens4f1 | grep -i scatter-gather
scatter-gather: on
        tx-scatter-gather: on
        tx-scatter-gather-fraglist: off [fixed]
root@ubuntu:/home/ubuntu# ethtool -k ens4f1 | grep -i tcp-segmentation
tcp-segmentation-offload: on
        tx-tcp-segmentation: on
root@ubuntu:/home/ubuntu# 
```
# generic-segmentation-offload
```
root@ubuntu:/home/ubuntu# ethtool -k ens4f1 | grep -i generic-segmentation-offload
generic-segmentation-offload: on
root@ubuntu:/home/ubuntu# ethtool -k ens4f1 | grep -i generic-receive-offload
generic-receive-offload: on
root@ubuntu:/home/ubuntu# ethtool -k ens4f1 | grep -i udp-fragmentation-offload
udp-fragmentation-offload: off
root@ubuntu:/home/ubuntu# 
```


# tx-udp_tnl-segmentation

```
root@ubuntu:/home/ubuntu# ethtool -i ens4f1
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
root@ubuntu:/home/ubuntu# ethtool -k ens4f1 | grep tx-udp_tnl-segmentation
tx-udp_tnl-segmentation: on
root@ubuntu:/home/ubuntu# 
```


## i40e
```
[dirk@v5-65 ~]$ ethtool -k eno1 | grep tx-udp_tnl-segmentation
tx-udp_tnl-segmentation: on
[dirk@v5-65 ~]$ ethtool -i eno1
driver: i40e
version: 2.8.10-k
firmware-version: 3.33 0x80000f09 255.65535.255
expansion-rom-version: 
bus-info: 0000:1a:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: yes
supports-register-dump: yes
supports-priv-flags: yes
[dirk@v5-65 ~]$ 
```
## Intel X54
VXLAN设备在发送数据时，会设置SKB_GSO_UDP_TUNNEL:  
```C
static int handle_offloads(struct sk_buff *skb)
{
	if (skb_is_gso(skb)) {
		int err = skb_unclone(skb, GFP_ATOMIC);
		if (unlikely(err))
			return err;

		skb_shinfo(skb)->gso_type |= SKB_GSO_UDP_TUNNEL;
	} else if (skb->ip_summed != CHECKSUM_PARTIAL)
		skb->ip_summed = CHECKSUM_NONE;

	return 0;
}
```
值得注意的是，该特性***只有当内层的packet为TCP协议时***，才有意义。前面已经讨论ixgbe不支持UFO，所以对UDP packet，最终会在推送给物理网卡时(dev_hard_start_xmit)进行软件GSO。   

# Discussion

* (1) Is it necessary to change MTU of flannel.1 to 1450 ?

对于TCP，veth/flannel.1都开启了TSO和GSO。对于inner packet > flannel.1 MTU，如果物理网卡支持VXLAN offload，最终由物理网卡完成分片；如果物理网卡不支持vxlan offload，走内核的GSO完成分片。但是，对于inner packet < flannel.1 MTU，inner packet + outer header >  物理网卡MTU，就会导致outer packet在ip_fragment中进行第二次分片，影响性能。

对于UDP，veth默认没有UFO，flannel.1开启了UFO，如果不减小flannel.1的MTU，可能会导致inner packet + outer header > 物理网卡的MTU，这可能会导致outer packet在ip_fragment中进行第二次分片，影响性能。

所以，不管怎样，减少flannel.1的MTU都是必要的。

* (2) Should enable UDP RSS for vxlan ?