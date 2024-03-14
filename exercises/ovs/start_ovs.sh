#!/bin/bash
#启动ovsdb server
./ovsdb/ovsdb-server /etc/openvswitch/conf.db \
-vconsole:emer -vsyslog:err -vfile:info \
--remote=punix:/var/run/openvswitch/db.sock \
--private-key=db:Open_vSwitch,SSL,private_key \
--certificate=db:Open_vSwitch,SSL,certificate \
--bootstrap-ca-cert=db:Open_vSwitch,SSL,ca_cert --no-chdir \
--log-file=/var/log/openvswitch/ovsdb-server.log \
--pidfile=/var/run/openvswitch/ovsdb-server.pid \
--detach --monitor

sleep 5
#export PATH=$PATH:/usr/local/share/openvswitch/scripts
#export PATH=$PATH:./utilities
#ovs-ctl start
#第一次启动ovs需要初始化
ovs-vsctl --no-wait init (低版本 < ovs-v2.7.0)
#从ovs-v2.7.0开始，开启dpdk功能已不是vswitchd进程启动时指定–dpdk等参数了，而是通过设置ovsdb来开启dpdk功能
sleep 5
ovs-vsctl --no-wait set Open_vSwitch . other_config:dpdk-init=true (高版本 > ovs-v2.7.0)
sleep 5
#启动vswitchd进程
ovs-vswitchd unix:/var/run/openvswitch/db.sock \
-vconsole:emer -vsyslog:err -vfile:info --mlockall --no-chdir \
--log-file=/var/log/openvswitch/ovs-vswitchd.log \
--pidfile=/var/run/openvswitch/ovs-vswitchd.pid \
--detach --monitor
