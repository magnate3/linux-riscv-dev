

```
[root@centos7 ICTCP]# insmod  ictcp.ko param_dev=enp5s0
[root@centos7 ICTCP]# dmesg | tail -n 10
[21100.156987] dwrr mq ecn marking 
[21100.180594] dwrr mq ecn marking 
[21102.410556] dwrr mq ecn marking 
[21102.413965] dwrr mq ecn marking 
[416653.291323] Ebtables v2.0 unregistered
[416661.403488] ip_tables: (C) 2000-2006 Netfilter Core Team
[603368.060414] ictcp: not specify network interface.
[603368.357784] Start ICTCP kernel module on eth1
[603442.446820] Stop ICTCP kernel module
[603500.595487] Start ICTCP kernel module on enp5s0
[root@centos7 ICTCP]# 
```