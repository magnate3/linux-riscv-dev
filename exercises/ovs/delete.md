

# del-flows 
不需要action，output关键字  
```
[root@centos7 openvswitch-dbg-2.13.0]# ./utilities/ovs-ofctl  del-flows br0 "in_port=1,actions=internet:172.16.1.111,output=2"
ovs-ofctl: unknown keyword actions
[root@centos7 openvswitch-dbg-2.13.0]# ./utilities/ovs-ofctl  del-flows br0 "in_port=1,output=2"
ovs-ofctl: unknown keyword output
[root@centos7 openvswitch-dbg-2.13.0]# ./utilities/ovs-ofctl  del-flows br0 "in_port=1"
[root@centos7 openvswitch-dbg-2.13.0]# 
```