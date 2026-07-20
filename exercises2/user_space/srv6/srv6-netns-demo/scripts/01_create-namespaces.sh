#!/bin/bash

if [[ $(id -u) -ne 0 ]] ; then
    echo "Please run with sudo"
    exit 1
fi

run () {
    echo "$@"
    "$@" || exit 1
}


create_router_SRv6-1 () {
    # setup namespaces
    run ip netns add host1
    run ip netns add srv6-1

    # setup veth peer srv6-1 <-> host1
    run ip link add veth-h1-sr1 type veth peer name veth-sr1-h1
    run ip link set veth-h1-sr1 netns host1
    run ip link set veth-sr1-h1 netns srv6-1

    # host1 configuraiton
    run ip netns exec host1 ip link set lo up
    run ip netns exec host1 ip addr add 2001:db1::2/64 dev veth-h1-sr1
    run ip netns exec host1 ip link set veth-h1-sr1 up
    run ip netns exec host1 ip -6 route add 2001:db2::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db3::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db4::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db5::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db6::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db7::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db8::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db9::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db10::/64 via 2001:db1::1
    run ip netns exec host1 ip -6 route add 2001:db11::/64 via 2001:db1::1

    # srv6-1 configuration
    run ip netns exec srv6-1 ip link set lo up
    run ip netns exec srv6-1 ip link set veth-sr1-h1 up
    run ip netns exec srv6-1 ip addr add 2001:db1::1/64 dev veth-sr1-h1

    # sysctl for router srv6-1
    ip netns exec srv6-1 sysctl net.ipv6.conf.all.forwarding=1
}


create_router_SRv6-2 () {
    # setup namespaces
    run ip netns add srv6-2

    # srv6-2 configuration
    run ip netns exec srv6-2 ip link set lo up

    # sysctl for router srv6-2
    ip netns exec srv6-2 sysctl net.ipv6.conf.all.forwarding=1
}


create_router_SRv6-3 () {
    # setup namespaces
    run ip netns add srv6-5
    run ip netns add srv6-6
    run ip netns add srv6-3

    # setup veth peer
    run ip link add veth-sr5-sr3 type veth peer name veth-sr3-sr5
    run ip link set veth-sr5-sr3 netns srv6-5
    run ip link set veth-sr3-sr5 netns srv6-3

    run ip link add veth-sr6-sr3 type veth peer name veth-sr3-sr6
    run ip link set veth-sr6-sr3 netns srv6-6
    run ip link set veth-sr3-sr6 netns srv6-3

    # srv6-5 configuraiton
    run ip netns exec srv6-5 ip link set lo up
    run ip netns exec srv6-5 ip addr add 2001:db11::8/64 dev veth-sr5-sr3
    run ip netns exec srv6-5 ip link set veth-sr5-sr3 up
    run ip netns exec srv6-5 ip -6 route add 2001:db1::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db2::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db3::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db4::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db5::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db6::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db7::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db8::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db9::/64 via 2001:db11::1
    run ip netns exec srv6-5 ip -6 route add 2001:db10::/64 via 2001:db11::1

    # srv6-6 configuraiton
    run ip netns exec srv6-6 ip link set lo up
    run ip netns exec srv6-6 ip addr add 2001:db11::9/64 dev veth-sr6-sr3
    run ip netns exec srv6-6 ip link set veth-sr6-sr3 up
    run ip netns exec srv6-6 ip -6 route add 2001:db1::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db2::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db3::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db4::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db5::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db6::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db7::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db8::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db9::/64 via 2001:db11::1
    run ip netns exec srv6-6 ip -6 route add 2001:db10::/64 via 2001:db11::1


    # srv6-3 configuration
    run ip netns exec srv6-3 ip link set lo up
    run ip netns exec srv6-3 ip link set veth-sr3-sr5 up
    run ip netns exec srv6-3 ip link set veth-sr3-sr6 up
    run ip netns exec srv6-3 ip link add hostbr0 type bridge
    run ip netns exec srv6-3 ip link set hostbr0 up
    run ip netns exec srv6-3 ip link set dev veth-sr3-sr5 master hostbr0
    run ip netns exec srv6-3 ip link set dev veth-sr3-sr6 master hostbr0
    run ip netns exec srv6-3 ip addr add 2001:db11::1/64 dev hostbr0

    # sysctl for srv6-3
    ip netns exec srv6-3 sysctl net.ipv6.conf.all.forwarding=1

    # seg6_enable for srv6-5 and srv6-6
    ip netns exec srv6-5 sysctl net.ipv6.conf.all.forwarding=1
    ip netns exec srv6-6 sysctl net.ipv6.conf.all.forwarding=1
}


create_router_SRv6-4 () {
    # setup namespaces
    run ip netns add host2
    run ip netns add srv6-4

    # setup veth peer
    run ip link add veth-h2-sr4 type veth peer name veth-sr4-h2
    run ip link set veth-h2-sr4 netns host2
    run ip link set veth-sr4-h2 netns srv6-4

    # host2 configuraiton
    run ip netns exec host2 ip link set lo up
    run ip netns exec host2 ip addr add 2001:db10::2/64 dev veth-h2-sr4
    run ip netns exec host2 ip link set veth-h2-sr4 up
    run ip netns exec host2 ip -6 route add 2001:db1::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db2::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db3::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db4::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db5::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db6::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db7::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db8::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db9::/64 via 2001:db10::1
    run ip netns exec host2 ip -6 route add 2001:db11::/64 via 2001:db10::1

    # srv6-4 configuration
    run ip netns exec srv6-4 ip link set lo up
    run ip netns exec srv6-4 ip link set veth-sr4-h2 up
    run ip netns exec srv6-4 ip addr add 2001:db10::1/64 dev veth-sr4-h2


    # sysctl for srv6-4
    ip netns exec srv6-4 sysctl net.ipv6.conf.all.forwarding=1
}


create_router_IPv6-1 () {
    # setup namespaces
    run ip netns add ipv6-1

    # ipv6-1 configuration
    run ip netns exec ipv6-1 ip link set lo up



    # sysctl for router ipv6-1
    ip netns exec ipv6-1 sysctl net.ipv6.conf.all.forwarding=1
    # ip netns exec ipv6-1 sysctl net.ipv6.conf.all.seg6_enabled=1
}


create_router_IPv6-2 () {
    # setup namespaces
    run ip netns add ipv6-2

    # ipv6-2 configuration
    run ip netns exec ipv6-2 ip link set lo up

    # sysctl for router ipv6-2
    ip netns exec ipv6-2 sysctl net.ipv6.conf.all.forwarding=1
    # ip netns exec ipv6-2 sysctl net.ipv6.conf.all.seg6_enabled=1
}


## peer ##
connect_srv6-1_srv6-2 () {
    # create veth peer
    run ip link add veth-sr1-sr2 type veth peer name veth-sr2-sr1
    run ip link set veth-sr1-sr2 netns srv6-1
    run ip link set veth-sr2-sr1 netns srv6-2

    # configure srv6-1
    run ip netns exec srv6-1 ip link set veth-sr1-sr2 up
    run ip netns exec srv6-1 ip addr add 2001:db2::1/64 dev veth-sr1-sr2


    # configure srv6-2
    run ip netns exec srv6-2 ip link set veth-sr2-sr1 up
    run ip netns exec srv6-2 ip addr add 2001:db2::2/64 dev veth-sr2-sr1
}


connect_srv6-1_ipv6-1 () {
    # create veth peer
    run ip link add veth-sr1-ip1 type veth peer name veth-ip1-sr1
    run ip link set veth-sr1-ip1 netns srv6-1
    run ip link set veth-ip1-sr1 netns ipv6-1

    # configure srv6-1
    run ip netns exec srv6-1 ip link set veth-sr1-ip1 up
    run ip netns exec srv6-1 ip addr add 2001:db3::1/64 dev veth-sr1-ip1

    # configure ipv6-1
    run ip netns exec ipv6-1 ip link set veth-ip1-sr1 up
    run ip netns exec ipv6-1 ip addr add 2001:db3::2/64 dev veth-ip1-sr1

}


connect_srv6-2_ipv6-1 () {
    # create veth peer
    run ip link add veth-sr2-ip1 type veth peer name veth-ip1-sr2
    run ip link set veth-sr2-ip1 netns srv6-2
    run ip link set veth-ip1-sr2 netns ipv6-1

    # configure srv6-2
    run ip netns exec srv6-2 ip link set veth-sr2-ip1 up
    run ip netns exec srv6-2 ip addr add 2001:db4::1/64 dev veth-sr2-ip1

    # configure ipv6-1
    run ip netns exec ipv6-1 ip link set veth-ip1-sr2 up
    run ip netns exec ipv6-1 ip addr add 2001:db4::2/64 dev veth-ip1-sr2
}


connect_srv6-2_srv6-3 () {
    # create veth peer
    run ip link add veth-sr2-sr3 type veth peer name veth-sr3-sr2
    run ip link set veth-sr2-sr3 netns srv6-2
    run ip link set veth-sr3-sr2 netns srv6-3

    # configure srv6-2
    run ip netns exec srv6-2 ip link set veth-sr2-sr3 up
    run ip netns exec srv6-2 ip addr add 2001:db5::1/64 dev veth-sr2-sr3

    # configure srv6-3
    run ip netns exec srv6-3 ip link set veth-sr3-sr2 up
    run ip netns exec srv6-3 ip addr add 2001:db5::2/64 dev veth-sr3-sr2
}

connect_ipv6-1_ipv6-2 () {
    # create veth peer
    run ip link add veth-ip1-ip2 type veth peer name veth-ip2-ip1
    run ip link set veth-ip1-ip2 netns ipv6-1
    run ip link set veth-ip2-ip1 netns ipv6-2

    # configure ipv6-1
    run ip netns exec ipv6-1 ip link set veth-ip1-ip2 up
    run ip netns exec ipv6-1 ip addr add 2001:db6::1/64 dev veth-ip1-ip2

    # configure ipv6-2
    run ip netns exec ipv6-2 ip link set veth-ip2-ip1 up
    run ip netns exec ipv6-2 ip addr add 2001:db6::2/64 dev veth-ip2-ip1
}


connect_srv6-3_ipv6-2 () {
    # create veth peer
    run ip link add veth-sr3-ip2 type veth peer name veth-ip2-sr3
    run ip link set veth-sr3-ip2 netns srv6-3
    run ip link set veth-ip2-sr3 netns ipv6-2

    # configure srv6-3
    run ip netns exec srv6-3 ip link set veth-sr3-ip2 up
    run ip netns exec srv6-3 ip addr add 2001:db7::1/64 dev veth-sr3-ip2

    # configure ipv6-2
    run ip netns exec ipv6-2 ip link set veth-ip2-sr3 up
    run ip netns exec ipv6-2 ip addr add 2001:db7::2/64 dev veth-ip2-sr3
}


connect_srv6-3_srv6-4 () {
    # create veth peer
    run ip link add veth-sr3-sr4 type veth peer name veth-sr4-sr3
    run ip link set veth-sr3-sr4 netns srv6-3
    run ip link set veth-sr4-sr3 netns srv6-4

    # configure srv6-3
    run ip netns exec srv6-3 ip link set veth-sr3-sr4 up
    run ip netns exec srv6-3 ip addr add 2001:db8::1/64 dev veth-sr3-sr4

    # configure srv6-4
    run ip netns exec srv6-4 ip link set veth-sr4-sr3 up
    run ip netns exec srv6-4 ip addr add 2001:db8::2/64 dev veth-sr4-sr3
}


connect_ipv6-2_srv6-4 () {
    # create veth peer
    run ip link add veth-ip2-sr4 type veth peer name veth-sr4-ip2
    run ip link set veth-ip2-sr4 netns ipv6-2
    run ip link set veth-sr4-ip2 netns srv6-4

    # configure ipv6-2
    run ip netns exec ipv6-2 ip link set veth-ip2-sr4 up
    run ip netns exec ipv6-2 ip addr add 2001:db9::1/64 dev veth-ip2-sr4

    # configure srv6-4
    run ip netns exec srv6-4 ip link set veth-sr4-ip2 up
    run ip netns exec srv6-4 ip addr add 2001:db9::2/64 dev veth-sr4-ip2
}


route () {

    run ip netns exec srv6-1 ip -6 route add 2001:db10::/64 encap seg6 mode encap segs 2001:db5::/64 2001:db8::/64 dev veth-sr1-h1

    # srv6-1
    run ip netns exec srv6-1 ip -6 route add 2001:db4::/64 via 2001:db2::2
    run ip netns exec srv6-1 ip -6 route add 2001:db5::/64 via 2001:db2::2
    run ip netns exec srv6-1 ip -6 route add 2001:db6::/64 via 2001:db2::2
    run ip netns exec srv6-1 ip -6 route add 2001:db7::/64 via 2001:db2::2
    run ip netns exec srv6-1 ip -6 route add 2001:db8::/64 via 2001:db2::2
    run ip netns exec srv6-1 ip -6 route add 2001:db9::/64 via 2001:db2::2
    run ip netns exec srv6-1 ip -6 route add 2001:db10::/64 via 2001:db2::2
    run ip netns exec srv6-1 ip -6 route add 2001:db11::/64 via 2001:db2::2
    run ip netns exec srv6-1 ip -6 route add 2001:db4::/64 via 2001:db3::2
    run ip netns exec srv6-1 ip -6 route add 2001:db5::/64 via 2001:db3::2
    run ip netns exec srv6-1 ip -6 route add 2001:db6::/64 via 2001:db3::2
    run ip netns exec srv6-1 ip -6 route add 2001:db7::/64 via 2001:db3::2
    run ip netns exec srv6-1 ip -6 route add 2001:db8::/64 via 2001:db3::2
    run ip netns exec srv6-1 ip -6 route add 2001:db9::/64 via 2001:db3::2
    run ip netns exec srv6-1 ip -6 route add 2001:db10::/64 via 2001:db3::2
    run ip netns exec srv6-1 ip -6 route add 2001:db11::/64 via 2001:db3::2
    
    # srv6-2
    run ip netns exec srv6-2 ip -6 route add 2001:db1::/64 via 2001:db2::1
    run ip netns exec srv6-2 ip -6 route add 2001:db3::/64 via 2001:db2::1
    run ip netns exec srv6-2 ip -6 route add 2001:db6::/64 via 2001:db2::1
    run ip netns exec srv6-2 ip -6 route add 2001:db7::/64 via 2001:db2::1
    run ip netns exec srv6-2 ip -6 route add 2001:db8::/64 via 2001:db2::1
    run ip netns exec srv6-2 ip -6 route add 2001:db9::/64 via 2001:db2::1
    run ip netns exec srv6-2 ip -6 route add 2001:db10::/64 via 2001:db2::1
    run ip netns exec srv6-2 ip -6 route add 2001:db11::/64 via 2001:db2::1
    run ip netns exec srv6-2 ip -6 route add 2001:db1::/64 via 2001:db4::2
    run ip netns exec srv6-2 ip -6 route add 2001:db3::/64 via 2001:db4::2
    run ip netns exec srv6-2 ip -6 route add 2001:db6::/64 via 2001:db4::2
    run ip netns exec srv6-2 ip -6 route add 2001:db7::/64 via 2001:db4::2
    run ip netns exec srv6-2 ip -6 route add 2001:db8::/64 via 2001:db4::2
    run ip netns exec srv6-2 ip -6 route add 2001:db9::/64 via 2001:db4::2
    run ip netns exec srv6-2 ip -6 route add 2001:db10::/64 via 2001:db4::2
    run ip netns exec srv6-2 ip -6 route add 2001:db11::/64 via 2001:db4::2
    run ip netns exec srv6-2 ip -6 route add 2001:db1::/64 via 2001:db5::2
    run ip netns exec srv6-2 ip -6 route add 2001:db3::/64 via 2001:db5::2
    run ip netns exec srv6-2 ip -6 route add 2001:db6::/64 via 2001:db5::2
    run ip netns exec srv6-2 ip -6 route add 2001:db7::/64 via 2001:db5::2
    run ip netns exec srv6-2 ip -6 route add 2001:db8::/64 via 2001:db5::2
    run ip netns exec srv6-2 ip -6 route add 2001:db9::/64 via 2001:db5::2
    run ip netns exec srv6-2 ip -6 route add 2001:db10::/64 via 2001:db5::2
    run ip netns exec srv6-2 ip -6 route add 2001:db11::/64 via 2001:db5::2
    
    # srv6-3
    run ip netns exec srv6-3 ip -6 route add 2001:db1::/64 via 2001:db5::1
    run ip netns exec srv6-3 ip -6 route add 2001:db2::/64 via 2001:db5::1
    run ip netns exec srv6-3 ip -6 route add 2001:db3::/64 via 2001:db5::1
    run ip netns exec srv6-3 ip -6 route add 2001:db4::/64 via 2001:db5::1
    run ip netns exec srv6-3 ip -6 route add 2001:db6::/64 via 2001:db5::1
    run ip netns exec srv6-3 ip -6 route add 2001:db9::/64 via 2001:db5::1
    run ip netns exec srv6-3 ip -6 route add 2001:db10::/64 via 2001:db5::1
    run ip netns exec srv6-3 ip -6 route add 2001:db1::/64 via 2001:db7::2
    run ip netns exec srv6-3 ip -6 route add 2001:db2::/64 via 2001:db7::2
    run ip netns exec srv6-3 ip -6 route add 2001:db3::/64 via 2001:db7::2
    run ip netns exec srv6-3 ip -6 route add 2001:db4::/64 via 2001:db7::2
    run ip netns exec srv6-3 ip -6 route add 2001:db6::/64 via 2001:db7::2
    run ip netns exec srv6-3 ip -6 route add 2001:db9::/64 via 2001:db7::2
    run ip netns exec srv6-3 ip -6 route add 2001:db10::/64 via 2001:db7::2
    run ip netns exec srv6-3 ip -6 route add 2001:db1::/64 via 2001:db8::2
    run ip netns exec srv6-3 ip -6 route add 2001:db2::/64 via 2001:db8::2
    run ip netns exec srv6-3 ip -6 route add 2001:db3::/64 via 2001:db8::2
    run ip netns exec srv6-3 ip -6 route add 2001:db4::/64 via 2001:db8::2
    run ip netns exec srv6-3 ip -6 route add 2001:db6::/64 via 2001:db8::2
    run ip netns exec srv6-3 ip -6 route add 2001:db9::/64 via 2001:db8::2
    run ip netns exec srv6-3 ip -6 route add 2001:db10::/64 via 2001:db8::2
    
    # srv6-4
    run ip netns exec srv6-4 ip -6 route add 2001:db1::/64 via 2001:db8::1
    run ip netns exec srv6-4 ip -6 route add 2001:db2::/64 via 2001:db8::1
    run ip netns exec srv6-4 ip -6 route add 2001:db3::/64 via 2001:db8::1
    run ip netns exec srv6-4 ip -6 route add 2001:db4::/64 via 2001:db8::1
    run ip netns exec srv6-4 ip -6 route add 2001:db5::/64 via 2001:db8::1
    run ip netns exec srv6-4 ip -6 route add 2001:db6::/64 via 2001:db8::1
    run ip netns exec srv6-4 ip -6 route add 2001:db7::/64 via 2001:db8::1
    run ip netns exec srv6-4 ip -6 route add 2001:db1::/64 via 2001:db9::1
    run ip netns exec srv6-4 ip -6 route add 2001:db2::/64 via 2001:db9::1
    run ip netns exec srv6-4 ip -6 route add 2001:db3::/64 via 2001:db9::1
    run ip netns exec srv6-4 ip -6 route add 2001:db4::/64 via 2001:db9::1
    run ip netns exec srv6-4 ip -6 route add 2001:db5::/64 via 2001:db9::1
    run ip netns exec srv6-4 ip -6 route add 2001:db6::/64 via 2001:db9::1
    run ip netns exec srv6-4 ip -6 route add 2001:db7::/64 via 2001:db9::1
    
    
    # ipv6-1
    run ip netns exec ipv6-1 ip -6 route add 2001:db1::/64 via 2001:db3::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db2::/64 via 2001:db3::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db5::/64 via 2001:db3::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db7::/64 via 2001:db3::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db8::/64 via 2001:db3::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db9::/64 via 2001:db3::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db10::/64 via 2001:db3::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db11::/64 via 2001:db3::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db1::/64 via 2001:db4::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db2::/64 via 2001:db4::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db5::/64 via 2001:db4::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db7::/64 via 2001:db4::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db8::/64 via 2001:db4::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db9::/64 via 2001:db4::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db10::/64 via 2001:db4::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db11::/64 via 2001:db4::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db1::/64 via 2001:db6::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db2::/64 via 2001:db6::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db5::/64 via 2001:db6::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db7::/64 via 2001:db6::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db8::/64 via 2001:db6::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db9::/64 via 2001:db6::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db10::/64 via 2001:db6::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db11::/64 via 2001:db6::2
    
    # ipv6-2
    run ip netns exec ipv6-1 ip -6 route add 2001:db1::/64 via 2001:db6::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db2::/64 via 2001:db6::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db3::/64 via 2001:db6::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db4::/64 via 2001:db6::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db5::/64 via 2001:db6::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db8::/64 via 2001:db6::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db10::/64 via 2001:db6::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db11::/64 via 2001:db6::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db1::/64 via 2001:db7::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db2::/64 via 2001:db7::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db3::/64 via 2001:db7::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db4::/64 via 2001:db7::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db5::/64 via 2001:db7::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db8::/64 via 2001:db7::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db10::/64 via 2001:db7::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db11::/64 via 2001:db7::1
    run ip netns exec ipv6-1 ip -6 route add 2001:db1::/64 via 2001:db9::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db2::/64 via 2001:db9::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db3::/64 via 2001:db9::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db4::/64 via 2001:db9::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db5::/64 via 2001:db9::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db8::/64 via 2001:db9::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db10::/64 via 2001:db9::2
    run ip netns exec ipv6-1 ip -6 route add 2001:db11::/64 via 2001:db9::2
}






# exec router functions
create_router_SRv6-1
create_router_SRv6-2
create_router_SRv6-3
create_router_SRv6-4
create_router_IPv6-1
create_router_IPv6-2


# exec connect functions
connect_srv6-1_srv6-2
connect_srv6-1_ipv6-1
connect_srv6-2_ipv6-1
connect_srv6-2_srv6-3
connect_ipv6-1_ipv6-2
connect_srv6-3_ipv6-2
connect_srv6-3_srv6-4
connect_ipv6-2_srv6-4

# route

status=0; $SHELL || status=$?
exit $status
