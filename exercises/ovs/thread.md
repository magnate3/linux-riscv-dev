
```
ps -ef | grep ovs
top -H -p 12519
```

# handler和revalidator线程个数
```
[root@centos7 openvswitch-dbg-2.13.0]# ovs-vsctl --no-wait set Open_vSwitch . other_config:n-handler-threads=1
[root@centos7 openvswitch-dbg-2.13.0]# ovs-vsctl --no-wait set Open_vSwitch . other_config:n-revalidator-threads=1
[root@centos7 openvswitch-dbg-2.13.0]#
```

revalidate_ukey__    xlate_ukey  xlate_key   xlate_actions
 rule_dpif_lookup_from_table 
 
  b ofproto/ofproto-dpif-xlate.c:7632
  
  
  
  
  
  ```
  ```