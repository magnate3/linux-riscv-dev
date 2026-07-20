

# driver

```
root@SIG-0101:~# brctl show 
bridge name     bridge id               STP enabled     interfaces
switch          8000.001df312527c       no              swp0
                                                        swp1
                                                        swp2
                                                        swp3
root@SIG-0101:~# 

root@SIG-0101:~# ethtool -i swp2
driver: dsa
version: 5.4.3-rt1
firmware-version: N/A
expansion-rom-version: 
bus-info: platform
supports-statistics: yes
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: no
root@SIG-0101:~# ethtool -i  switch
driver: bridge
version: 2.3
firmware-version: N/A
expansion-rom-version: 
bus-info: N/A
supports-statistics: no
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: no
root@SIG-0101:~#
```

#  kernel module “bridge”.

```
static int __init br_init(void)
{
...
	err = stp_proto_register(&br_stp_proto);
...
	err = br_fdb_init();
...
	err = register_pernet_subsys(&br_net_ops);
...
	err = br_nf_core_init();
...
	err = br_netlink_init();
...
	brioctl_set(br_ioctl_deviceless_stub);
...
}
module_init(br_init)
...
MODULE_ALIAS_RTNL_LINK("bridge");
static const struct stp_proto br_stp_proto = {
	.rcv	= br_stp_rcv,
};
```

#  br_flood

```

```

# references

![Transmitting on a Bridge Device](https://kernelnewbies.org/Bridging_and_Forwarding)

![Understanding how Linux ethernet bridge is setup and works](https://medium.com/@ravi.eticala/understanding-how-linux-ethernet-bridge-is-setup-and-works-771ee75bdf67)