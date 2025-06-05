#!/bin/sh
#
# +-----------+ 30.30.0.2/24  +-----------+               +--------+
# | CL        |---------------| UPF       |---------------| SV     |
# +-----------+            .1 +-----------+  70.70.0.2/24 +--------+
#
# enp0s16              enp0s17
# 0000:00:10.0         0000:00:11.0
# 08:00:27:a0:a9:3e    08:00:27:af:62:0b
#
#                          enp0s9                enp0s10
#                          0000:00:09.0          0000:00:0a.0
#                          08:00:27:15:8c:70     08:00:27:dc:7d:2a
#

ip link add enp0s10 type veth peer  enp0s16

ip netns add SV
ip link set enp0s10 up netns SV
ip netns exec SV ip addr add 70.70.0.2/24 dev enp0s10
ip netns exec SV ip addr add 127.0.0.1/8 dev lo
ip netns exec SV ip link set lo up
ip netns exec CL ip addr add 80.60.0.2/16 dev lo
ip netns exec SV ip route add default via 70.70.0.1
ip netns exec SV ip link set enp0s10 mtu 1456

ip netns add CL
ip link set enp0s16 up netns CL
ip netns exec CL ip addr add 30.30.0.2/24 dev enp0s16
ip netns exec CL ip addr add 127.0.0.1/8 dev lo
ip netns exec CL ip addr add 60.60.0.2/16 dev lo
ip netns exec CL ip link set lo up

echo "===> Spawn gtp5g-link"
#ip netns exec CL ./gtp5g-link add gtp5gtest --ran &
ip netns exec CL ./gtp5g-link add gtp5gtest &
echo $! > gtp5g-link.pid
sleep 0.1
ip netns exec CL ./gtp5g-tunnel add qer gtp5gtest 1:123 --qfi 9
ip netns exec CL ./gtp5g-tunnel add qer gtp5gtest 1:321 --qfi 9
ip netns exec CL ./gtp5g-tunnel add far gtp5gtest 1:1 --action 2 \
	--hdr-creation 0 78 70.70.0.1 2152
ip netns exec CL ./gtp5g-tunnel add far gtp5gtest 1:2 --action 2 \
	--hdr-creation 0 78 30.30.0.1 2152
ip netns exec CL ./gtp5g-tunnel add pdr gtp5gtest 1:1 --pcd 1 --hdr-rm 0 \
	--ue-ipv4 80.60.0.2 --f-teid 87 70.70.0.2  --far-id 1 --qer-id 321
ip netns exec CL ./gtp5g-tunnel add pdr gtp5gtest 1:2 --pcd 2 \
	--ue-ipv4 60.60.0.2 --f-teid 87 30.30.0.2 --far-id 2 --qer-id 123
ip netns exec CL ip route add 70.70.0.0/24 dev gtp5gtest
ip netns exec CL ip link set gtp5gtest mtu 1456

