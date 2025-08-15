#!/bin/bash

#      arp proxy <------+                                            +------> arp proxy
#                       |                                            |
#                     +------------------+          +------------------+          +------------------+
#                     | |      r0        |          |        r1      | |          |        cer       |
#  +-------+          | |                |          |                | |          |                  |           +-------+
#  |   h0  |          | |                |          |                | |          |                  |           |  h1   |
#  |       +----------+ veth1      veth2 +----------+ veth3      veth4 +----------+ veth5      veth6 +-----------+       |
#  | veth0 |          |                  |          |                  |          |                  |           | veth5 |
#  +-------+      10.0.0.254/24     fdff::1/64   fdff::2/64  172.16.0.254/24  172.16.0.253/24    10.0.0.254/24   +-------+
#                     |               ^  |          |  ^               |          |                  |
# 10.0.0.1/24         +------------------+          +------------------+          +------------------+          10.0.0.2/24
#                                     |                |
#                                     |                |
#                                     |                |
#                                     |                |
#                                     +                +
#     10.0.0.2 encap segs fc00::2 veth2                fc00::2 action End.DX4 nh4 172.16.0.253 veth4



IPP=ip

# Clean up previous nents
$IPP -all netns delete

$IPP netns add h0
$IPP netns add h1

$IPP netns add r0
$IPP netns add r1
$IPP netns add cer

$IPP link add veth0 type veth peer name veth1
$IPP link add veth2 type veth peer name veth3
$IPP link add veth4 type veth peer name veth5
$IPP link add veth6 type veth peer name veth7

$IPP link set veth0 netns h0

$IPP link set veth1 netns r0
$IPP link set veth2 netns r0

$IPP link set veth3 netns r1
$IPP link set veth4 netns r1

$IPP link set veth5 netns cer
$IPP link set veth6 netns cer

$IPP link set veth7 netns h1

###################
#### Node: h0 #####
###################

$IPP netns exec h0 $IPP link set dev lo up
$IPP netns exec h0 $IPP link set dev veth0 up
$IPP netns exec h0 $IPP addr add 10.0.0.1/24 dev veth0
$IPP netns exec h0 $IPP -4 route add default via 10.0.0.254 dev veth0


###################
#### Node: r0 #####
###################

$IPP netns exec r0 sysctl -w net.ipv4.ip_forward=1
$IPP netns exec r0 sysctl -w net.ipv4.conf.all.forwarding=1
$IPP netns exec r0 sysctl -w net.ipv6.conf.all.forwarding=1
$IPP netns exec r0 sysctl -w net.ipv6.conf.all.seg6_enabled=1
# disable also rp_filter on the receiving decap interface that will forward the
# packet to the right destination (through the nexthop)
$IPP netns exec r0 sysctl -w net.ipv4.conf.all.rp_filter=0
$IPP netns exec r0 sysctl -w net.ipv4.conf.veth1.rp_filter=0
$IPP netns exec r0 sysctl -w net.ipv4.conf.veth2.rp_filter=0
# Using proxy_arp we can simplify the configuration of clients
$IPP netns exec r0 sysctl -w net.ipv4.conf.all.proxy_arp=1
$IPP netns exec r0 sysctl -w net.ipv4.conf.veth1.proxy_arp=1

$IPP netns exec r0 $IPP link set dev lo up
$IPP netns exec r0 $IPP link set dev veth1 up
$IPP netns exec r0 $IPP link set dev veth2 up
$IPP netns exec r0 $IPP addr add fdff::1/64 dev veth2
$IPP netns exec r0 $IPP addr add 10.0.0.254/24 dev veth1
# Decap DX4
$IPP netns exec r0 $IPP -6 route add fc00::1/128 \
	encap seg6local action End.DX4 nh4 0.0.0.0 dev veth1
# Encap IPv4-in-IPv6
$IPP netns exec r0 $IPP -6 route add fc00::2/128 via fdff::2 dev veth2
$IPP netns exec r0 $IPP -4 route add 10.0.0.2/32 \
	encap seg6 mode encap segs fc00::2 dev veth2


###################
#### Node: r1 #####
###################

$IPP netns exec r1 sysctl -w net.ipv4.ip_forward=1
$IPP netns exec r1 sysctl -w net.ipv4.conf.all.forwarding=1
$IPP netns exec r1 sysctl -w net.ipv6.conf.all.forwarding=1
$IPP netns exec r1 sysctl -w net.ipv6.conf.all.seg6_enabled=1
# disable also rp_filter on the receiving decap interface that will forward the
# packet to the right destination (through the nexthop)
$IPP netns exec r1 sysctl -w net.ipv4.conf.all.rp_filter=0
$IPP netns exec r1 sysctl -w net.ipv4.conf.veth3.rp_filter=0
$IPP netns exec r1 sysctl -w net.ipv4.conf.veth4.rp_filter=0

$IPP netns exec r1 $IPP link set dev lo up
$IPP netns exec r1 $IPP link set dev veth3 up
$IPP netns exec r1 $IPP link set dev veth4 up
$IPP netns exec r1 $IPP addr add fdff::2/64 dev veth3
$IPP netns exec r1 $IPP addr add 172.16.0.254/24 dev veth4
# Decap DX4
$IPP netns exec r1 $IPP -6 route add fc00::2/128 \
	encap seg6local action End.DX4 nh4 172.16.0.253 dev veth4
# Encap IPv4-in-IPv6
$IPP netns exec r1 $IPP -6 route add fc00::1/128 via fdff::1 dev veth3
$IPP netns exec r1 $IPP -4 route add 10.0.0.1/32 \
	encap seg6 mode encap segs fc00::1 dev veth3


###################
#### Node: cer ####
###################

$IPP netns exec cer sysctl -w net.ipv4.ip_forward=1
$IPP netns exec cer sysctl -w net.ipv4.conf.all.forwarding=1
# disable also rp_filter on the receiving decap interface that will forward the
# packet to the right destination (through the nexthop)
$IPP netns exec cer sysctl -w net.ipv4.conf.all.rp_filter=0
$IPP netns exec cer sysctl -w net.ipv4.conf.veth5.rp_filter=0
$IPP netns exec cer sysctl -w net.ipv4.conf.veth6.rp_filter=0
# Using proxy_arp we can simplify the configuration of clients
$IPP netns exec cer sysctl -w net.ipv4.conf.all.proxy_arp=1
$IPP netns exec cer sysctl -w net.ipv4.conf.veth6.proxy_arp=1

$IPP netns exec cer $IPP link set dev lo up
$IPP netns exec cer $IPP link set dev veth5 up
$IPP netns exec cer $IPP link set dev veth6 up

$IPP netns exec cer $IPP addr add 172.16.0.253/24 dev veth5
$IPP netns exec cer $IPP addr add 10.0.0.254/24 dev veth6
# Route for reaching h0
$IPP netns exec cer $IPP route add 10.0.0.1/32 via 172.16.0.254 dev veth5


###################
#### Node: h1 #####
###################

$IPP netns exec h1 $IPP link set dev lo up
$IPP netns exec h1 $IPP link set dev veth7 up
$IPP netns exec h1 $IPP addr add 10.0.0.2/24 dev veth7
$IPP netns exec h1 $IPP -4 route add default via 10.0.0.254 dev veth7

