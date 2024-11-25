#!/bin/bash

cat << EOF >/etc/quagga/srv6-1_ospf6d.conf
hostname srv6-1
password zebra
interface veth-sr1-h1
 ipv6 ospf6 instance-id 0
interface veth-sr1-sr2
 ipv6 ospf6 instance-id 0
interface veth-sr1-ip1
 ipv6 ospf6 instance-id 0
router ospf6
 router-id 1.1.1.1
 area 0.0.0.0 range 2001:db1::/64
 area 0.0.0.0 range 2001:db2::/64
 area 0.0.0.0 range 2001:db3::/64
 area 0.0.0.0 range 2001:db4::/64
 area 0.0.0.0 range 2001:db5::/64
 area 0.0.0.0 range 2001:db6::/64
 area 0.0.0.0 range 2001:db7::/64
 area 0.0.0.0 range 2001:db8::/64
 area 0.0.0.0 range 2001:db9::/64
 area 0.0.0.0 range 2001:db10::/64
 area 0.0.0.0 range 2001:db11::/64
 interface veth-sr1-h1 area 0.0.0.0
 interface veth-sr1-sr2 area 0.0.0.0
 interface veth-sr1-ip1 area 0.0.0.0
EOF

cat << EOF >/etc/quagga/srv6-2_ospf6d.conf
hostname srv6-2
password zebra
interface veth-sr2-sr1
 ipv6 ospf6 instance-id 0
interface veth-sr2-sr3
 ipv6 ospf6 instance-id 0
interface veth-sr2-ip1
 ipv6 ospf6 instance-id 0
router ospf6
 router-id 2.2.2.2
 area 0.0.0.0 range 2001:db1::/64
 area 0.0.0.0 range 2001:db2::/64
 area 0.0.0.0 range 2001:db3::/64
 area 0.0.0.0 range 2001:db4::/64
 area 0.0.0.0 range 2001:db5::/64
 area 0.0.0.0 range 2001:db6::/64
 area 0.0.0.0 range 2001:db7::/64
 area 0.0.0.0 range 2001:db8::/64
 area 0.0.0.0 range 2001:db9::/64
 area 0.0.0.0 range 2001:db10::/64
 area 0.0.0.0 range 2001:db11::/64
 interface veth-sr2-sr1 area 0.0.0.0
 interface veth-sr2-sr3 area 0.0.0.0
 interface veth-sr2-ip1 area 0.0.0.0
EOF

cat << EOF >/etc/quagga/srv6-3_ospf6d.conf
hostname srv6-3
password zebra
interface veth-sr3-sr4
 ipv6 ospf6 instance-id 0
interface veth-sr3-sr2
 ipv6 ospf6 instance-id 0
interface veth-sr3-ip2
 ipv6 ospf6 instance-id 0
router ospf6
 router-id 3.3.3.3
 area 0.0.0.0 range 2001:db1::/64
 area 0.0.0.0 range 2001:db2::/64
 area 0.0.0.0 range 2001:db3::/64
 area 0.0.0.0 range 2001:db4::/64
 area 0.0.0.0 range 2001:db5::/64
 area 0.0.0.0 range 2001:db6::/64
 area 0.0.0.0 range 2001:db7::/64
 area 0.0.0.0 range 2001:db8::/64
 area 0.0.0.0 range 2001:db9::/64
 area 0.0.0.0 range 2001:db10::/64
 area 0.0.0.0 range 2001:db11::/64
 interface veth-sr3-sr4 area 0.0.0.0
 interface veth-sr3-sr2 area 0.0.0.0
 interface veth-sr3-ip2 area 0.0.0.0
EOF

cat << EOF >/etc/quagga/srv6-4_ospf6d.conf
hostname srv6-4
password zebra
interface veth-sr4-sr3
 ipv6 ospf6 instance-id 0
interface veth-sr4-h2
 ipv6 ospf6 instance-id 0
interface veth-sr4-ip2
 ipv6 ospf6 instance-id 0
router ospf6
 router-id 4.4.4.4
 area 0.0.0.0 range 2001:db1::/64
 area 0.0.0.0 range 2001:db2::/64
 area 0.0.0.0 range 2001:db3::/64
 area 0.0.0.0 range 2001:db4::/64
 area 0.0.0.0 range 2001:db5::/64
 area 0.0.0.0 range 2001:db6::/64
 area 0.0.0.0 range 2001:db7::/64
 area 0.0.0.0 range 2001:db8::/64
 area 0.0.0.0 range 2001:db9::/64
 area 0.0.0.0 range 2001:db10::/64
 area 0.0.0.0 range 2001:db11::/64
 interface veth-sr4-sr3 area 0.0.0.0
 interface veth-sr4-h2 area 0.0.0.0
 interface veth-sr4-ip2 area 0.0.0.0
EOF

cat << EOF >/etc/quagga/srv6-5_ospf6d.conf
hostname srv6-5
password zebra
interface veth-sr5-sr3
 ipv6 ospf6 instance-id 0
router ospf6
 router-id 5.5.5.5
 area 0.0.0.0 range 2001:db1::/64
 area 0.0.0.0 range 2001:db2::/64
 area 0.0.0.0 range 2001:db3::/64
 area 0.0.0.0 range 2001:db4::/64
 area 0.0.0.0 range 2001:db5::/64
 area 0.0.0.0 range 2001:db6::/64
 area 0.0.0.0 range 2001:db7::/64
 area 0.0.0.0 range 2001:db8::/64
 area 0.0.0.0 range 2001:db9::/64
 area 0.0.0.0 range 2001:db10::/64
 area 0.0.0.0 range 2001:db11::/64
 interface veth-sr5-sr3 area 0.0.0.0
EOF

cat << EOF >/etc/quagga/srv6-6_ospf6d.conf
hostname srv6-6
password zebra
interface veth-sr6-sr3
 ipv6 ospf6 instance-id 0
router ospf6
 router-id 6.6.6.6
 area 0.0.0.0 range 2001:db1::/64
 area 0.0.0.0 range 2001:db2::/64
 area 0.0.0.0 range 2001:db3::/64
 area 0.0.0.0 range 2001:db4::/64
 area 0.0.0.0 range 2001:db5::/64
 area 0.0.0.0 range 2001:db6::/64
 area 0.0.0.0 range 2001:db7::/64
 area 0.0.0.0 range 2001:db8::/64
 area 0.0.0.0 range 2001:db9::/64
 area 0.0.0.0 range 2001:db10::/64
 area 0.0.0.0 range 2001:db11::/64
 interface veth-sr6-sr3 area 0.0.0.0
EOF

cat << EOF >/etc/quagga/ipv6-1_ospf6d.conf
hostname ipv6-1 
password zebra
interface veth-ip1-sr1
 ipv6 ospf6 instance-id 0
interface veth-ip1-sr2
 ipv6 ospf6 instance-id 0
interface veth-ip1-ip2
 ipv6 ospf6 instance-id 0
router ospf6
 router-id 7.7.7.7
 area 0.0.0.0 range 2001:db1::/64
 area 0.0.0.0 range 2001:db2::/64
 area 0.0.0.0 range 2001:db3::/64
 area 0.0.0.0 range 2001:db4::/64
 area 0.0.0.0 range 2001:db5::/64
 area 0.0.0.0 range 2001:db6::/64
 area 0.0.0.0 range 2001:db7::/64
 area 0.0.0.0 range 2001:db8::/64
 area 0.0.0.0 range 2001:db9::/64
 area 0.0.0.0 range 2001:db10::/64
 area 0.0.0.0 range 2001:db11::/64
 interface veth-ip1-sr1 area 0.0.0.0
 interface veth-ip1-sr2 area 0.0.0.0
 interface veth-ip1-ip2 area 0.0.0.0
EOF

cat << EOF >/etc/quagga/ipv6-2_ospf6d.conf
hostname ipv6-2
password zebra
interface veth-ip2-ip1
 ipv6 ospf6 instance-id 0
interface veth-ip2-sr3
 ipv6 ospf6 instance-id 0
interface veth-ip2-sr4
 ipv6 ospf6 instance-id 0
router ospf6
 router-id 8.8.8.8
 area 0.0.0.0 range 2001:db1::/64
 area 0.0.0.0 range 2001:db2::/64
 area 0.0.0.0 range 2001:db3::/64
 area 0.0.0.0 range 2001:db4::/64
 area 0.0.0.0 range 2001:db5::/64
 area 0.0.0.0 range 2001:db6::/64
 area 0.0.0.0 range 2001:db7::/64
 area 0.0.0.0 range 2001:db8::/64
 area 0.0.0.0 range 2001:db9::/64
 area 0.0.0.0 range 2001:db10::/64
 area 0.0.0.0 range 2001:db11::/64
 interface veth-ip2-ip1 area 0.0.0.0
 interface veth-ip2-sr3 area 0.0.0.0
 interface veth-ip2-sr4 area 0.0.0.0
EOF

chown quagga.quagga /etc/quagga/*conf


touch /etc/quagga/zebra.conf
touch /etc/quagga/ospf6d.conf

ip netns exec srv6-1 /usr/sbin/zebra -d -f /etc/quagga/srv6-1_zebra.conf -i /var/run/quagga/srv6-1_zebra.pid -A 127.0.0.1 -z /var/run/quagga/srv6-1_zebra.vty
ip netns exec srv6-1 /usr/sbin/ospf6d -d -f /etc/quagga/srv6-1_ospf6d.conf -i /var/run/quagga/srv6-1_ospf6d.pid  -A 127.0.0.1 -z /var/run/quagga/srv6-1_zebra.vty -P 2601

ip netns exec srv6-2 /usr/sbin/zebra -d -f /etc/quagga/srv6-2_zebra.conf -i /var/run/quagga/srv6-2_zebra.pid -A 127.0.0.1 -z /var/run/quagga/srv6-2_zebra.vty
ip netns exec srv6-2 /usr/sbin/ospf6d -d -f /etc/quagga/srv6-2_ospf6d.conf -i /var/run/quagga/srv6-2_ospf6d.pid -A 127.0.0.1 -z /var/run/quagga/srv6-2_zebra.vty -P 2601

ip netns exec srv6-3 /usr/sbin/zebra -d -f /etc/quagga/srv6-3_zebra.conf -i /var/run/quagga/srv6-3_zebra.pid -A 127.0.0.1 -z /var/run/quagga/srv6-3_zebra.vty
ip netns exec srv6-3 /usr/sbin/ospf6d -d -f /etc/quagga/srv6-3_ospf6d.conf -i /var/run/quagga/srv6-3_ospf6d.pid -A 127.0.0.1 -z /var/run/quagga/srv6-3_zebra.vty -P 2601

ip netns exec srv6-4 /usr/sbin/zebra -d -f /etc/quagga/srv6-4_zebra.conf -i /var/run/quagga/srv6-4_zebra.pid -A 127.0.0.1 -z /var/run/quagga/srv6-4_zebra.vty
ip netns exec srv6-4 /usr/sbin/ospf6d -d -f /etc/quagga/srv6-4_ospf6d.conf -i /var/run/quagga/srv6-4_ospf6d.pid -A 127.0.0.1 -z /var/run/quagga/srv6-4_zebra.vty -P 2601

ip netns exec srv6-5 /usr/sbin/zebra -d -f /etc/quagga/srv6-5_zebra.conf -i /var/run/quagga/srv6-5_zebra.pid -A 127.0.0.1 -z /var/run/quagga/srv6-5_zebra.vty
ip netns exec srv6-5 /usr/sbin/ospf6d -d -f /etc/quagga/srv6-5_ospf6d.conf -i /var/run/quagga/srv6-5_ospf6d.pid -A 127.0.0.1 -z /var/run/quagga/srv6-5_zebra.vty -P 2601

ip netns exec srv6-6 /usr/sbin/zebra -d -f /etc/quagga/srv6-6_zebra.conf -i /var/run/quagga/srv6-6_zebra.pid -A 127.0.0.1 -z /var/run/quagga/srv6-6_zebra.vty
ip netns exec srv6-6 /usr/sbin/ospf6d -d -f /etc/quagga/srv6-6_ospf6d.conf -i /var/run/quagga/srv6-6_ospf6d.pid -A 127.0.0.1 -z /var/run/quagga/srv6-6_zebra.vty -P 2601

ip netns exec ipv6-1 /usr/sbin/zebra -d -f /etc/quagga/ipv6-1_zebra.conf -i /var/run/quagga/ipv6-1_zebra.pid -A 127.0.0.1 -z /var/run/quagga/ipv6-1_zebra.vty
ip netns exec ipv6-1 /usr/sbin/ospf6d -d -f /etc/quagga/ipv6-1_ospf6d.conf -i /var/run/quagga/ipv6-1_ospf6d.pid -A 127.0.0.1 -z /var/run/quagga/ipv6-1_zebra.vty -P 2601

ip netns exec ipv6-2 /usr/sbin/zebra -d -f /etc/quagga/ipv6-2_zebra.conf -i /var/run/quagga/ipv6-2_zebra.pid -A 127.0.0.1 -z /var/run/quagga/ipv6-2_zebra.vty
ip netns exec ipv6-2 /usr/sbin/ospf6d -d -f /etc/quagga/ipv6-2_ospf6d.conf -i /var/run/quagga/ipv6-2_ospf6d.pid -A 127.0.0.1 -z /var/run/quagga/ipv6-2_zebra.vty -P 2601
