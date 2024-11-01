#!/bin/bash

#      arp proxy <------+                                        +------> arp proxy
#                       |                                        |
#                     +------------------+      +------------------+
#                     | |      r0        |      |        r1      | |
#  +-------+          | |                |      |                | |        +-------+
#  |   h0  |          | +                |      |                + |        |  h1   |
#  |       +----------+ veth1      veth2 +------+ veth3      veth4 +--------+       |
#  | veth0 |          |                  |      |                  |        | veth5 |
#  +-------+      10.0.0.254/24    fdff::1/64  fdff::2/64   10.0.0.254/24   +-------+
#                     |                 ^|      |v                 |
# 10.0.0.1/24         +------------------+      +------------------+       10.0.0.2/24
#                                       |        |
#                                       |        |
#                                       |        |
#                                       |        |
#                                       +        +
#             10.0.0.2 encap segs fc00::2        fc00::2 action End.DX4 nh4 10.0.0.2

IPP=ip

# Clean up previous network namespaces
$IPP -all netns delete

$IPP netns add h0
$IPP netns add h1

$IPP netns add r0
$IPP netns add r1

$IPP link add veth0 type veth peer name veth1
$IPP link add veth2 type veth peer name veth3
$IPP link add veth4 type veth peer name veth5

$IPP link set veth0 netns h0

$IPP link set veth1 netns r0
$IPP link set veth2 netns r0

$IPP link set veth3 netns r1
$IPP link set veth4 netns r1

$IPP link set veth5 netns h1

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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# NB: it is enough, for this example, to disable rp_filter only for     #
# interfaces that handle IPv4. veth1 is IPv4 configured but not veth2.  #
# In IPv6 we do not have any rp_filter feature implemented.             #
# $IPP netns exec r0 sysctl -w net.ipv4.conf.veth2.rp_filter=0          #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
	encap seg6local action End.DX4 nh4 10.0.0.1 dev veth1
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
#$IPP netns exec r1 sysctl -w net.ipv4.conf.veth3.rp_filter=0
$IPP netns exec r1 sysctl -w net.ipv4.conf.veth4.rp_filter=0
# Using proxy_arp we can simplify the configuration of clients
$IPP netns exec r1 sysctl -w net.ipv4.conf.all.proxy_arp=1
$IPP netns exec r1 sysctl -w net.ipv4.conf.veth4.proxy_arp=1

$IPP netns exec r1 $IPP link set dev lo up
$IPP netns exec r1 $IPP link set dev veth3 up
$IPP netns exec r1 $IPP link set dev veth4 up
$IPP netns exec r1 $IPP addr add fdff::2/64 dev veth3
$IPP netns exec r1 $IPP addr add 10.0.0.254/24 dev veth4
# Decap DX4
$IPP netns exec r1 $IPP -6 route add fc00::2/128 \
	encap seg6local action End.DX4 nh4 10.0.0.2 dev veth4
# Encap IPv4-in-IPv6
$IPP netns exec r1 $IPP -6 route add fc00::1/128 via fdff::1 dev veth3
$IPP netns exec r1 $IPP -4 route add 10.0.0.1/32 \
		encap seg6 mode encap segs fc00::1 dev veth3


###################
#### Node: h1 #####
###################

$IPP netns exec h1 $IPP link set dev lo up
$IPP netns exec h1 $IPP link set dev veth5 up
$IPP netns exec h1 $IPP addr add 10.0.0.2/24 dev veth5
$IPP netns exec h1 $IPP -4 route add default via 10.0.0.254 dev veth5
