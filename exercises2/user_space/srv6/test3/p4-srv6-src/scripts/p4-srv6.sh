#! /bin/bash
#

NTOPOLOGY=$(cat <<EOF
#========================================================================================
#  Network Topology
#
#                   +--------------+          +-------------+
#                   |   routerA    |          |   routerB   |
#  host1 veth1 -- vethA1        vethAB  --  vethBA       vethB2 -- veth2 host2
#                   |              |          |             |
#                   +--------------+          +-------------+
#
#
# Hosts:
#     host1:
#        veth1: fc00:000a::10/64
#     host2:
#        veth2: fc00:000b::10/64
# Routers:
#     routerA:
#        vethA1: fc00:000a::a/64
#        vethAB: fc00:00ab::a/64
#     routerB:
#        vethBA: fc00:00ab::b/64
#        vethB2: fc00:000b::b/64
#
# Example:
#     ip netns exec host1 ping fc00:000b::10
#     ip netns exec host2 ping fc00:000a::10
#=======================================================
** Exit to this shell to kill ** 
EOF
)

VERSION="0.0.4"

if [[ $(id -u) -ne 0 ]] ; then echo "Please run with sudo" ; exit 1 ; fi

set -e

run () {
    echo "$@"
    "$@" || exit 1
}

silent () {
    "$@" 2> /dev/null || true
}

create_network () {
    run ip netns add host1
    run ip netns add routerA
    run ip netns add routerB
    run ip netns add host2

    run ip link add name veth1 type veth peer name vethA1
    run ip link set veth1 netns host1
    #run ip link set vethA1 netns routerA

    run ip link add name vethAB type veth peer name vethBA
    #run ip link set vethAB netns routerA
    run ip link set vethBA netns routerB

    run ip link add name veth2 type veth peer name vethB2
    run ip link set veth2 netns host2
    run ip link set vethB2 netns routerB

    # host1 configuration
    run ip netns exec host1 ip link set lo up
    run ip netns exec host1 ip ad add fc00:000a::10/64 dev veth1
    run ip netns exec host1 ifconfig veth1 hw ether  1e:43:58:05:4d:e5
    run ip netns exec host1 ip -6 neigh add  fc00:a::a  lladdr 26:bb:7e:03:c0:62 nud permanent dev veth1
    run ip netns exec host1 ip link set veth1 up        
    run ip netns exec host1 ip -6 route add fc00::/16 via fc00:000a::a
    
    # routerA configuration
    #run ip netns exec routerA ip link set lo up
    #run ip netns exec routerA sysctl net.ipv6.conf.all.forwarding=1
    #run ip netns exec routerA sysctl net.ipv6.conf.all.seg6_enabled=1
    #run ip netns exec routerA sysctl net.ipv6.conf.vethA1.seg6_enabled=1    
    #run ip netns exec routerA ip ad add fc00:000a::a/64 dev vethA1
    #run ip netns exec routerA ip link set vethA1 up
    #run ip netns exec routerA sysctl net.ipv6.conf.vethAB.seg6_enabled=1
    #run ip netns exec routerA ip ad add fc00:00ab::a/64 dev vethAB
    #run ip netns exec routerA ip link set vethAB up
    #run ip netns exec routerA ip -6 route add fc00:000b::/64 encap seg6 mode encap segs fc00:00ab::b dev vethA1
    run ip netns exec routerA ip link set lo up
    run ip netns exec routerA sysctl net.ipv6.conf.all.forwarding=1
    run ip netns exec routerA sysctl net.ipv6.conf.all.seg6_enabled=1
    run sysctl net.ipv6.conf.vethA1.seg6_enabled=1    
    #run ip netns exec routerA ip ad add fc00:000a::a/64 dev vethA1
    run ifconfig vethA1 hw ether 26:bb:7e:03:c0:62 
    run ip link set vethA1 up
    run sysctl net.ipv6.conf.vethAB.seg6_enabled=1
    #run ip netns exec routerA ip ad add fc00:00ab::a/64 dev vethAB

    run ip a add fc00:00ab::a/64 dev vethAB
    run ip link set vethAB up
    #run ip netns exec routerA ip -6 route add fc00:000b::/64 encap seg6 mode encap segs fc00:00ab::b dev vethA1

    # routerB configuration
    run ip netns exec routerB ip link set lo up
    run ip netns exec routerB sysctl net.ipv6.conf.all.forwarding=1
    run ip netns exec routerB sysctl net.ipv6.conf.all.seg6_enabled=1
    run ip netns exec routerB sysctl net.ipv6.conf.vethB2.seg6_enabled=1
    run ip netns exec routerB ip ad add fc00:000b::b/64 dev vethB2
    run ip netns exec routerB ip link set vethB2 up
    run ip netns exec routerB sysctl net.ipv6.conf.vethBA.seg6_enabled=1
    run ip netns exec routerB ip ad add fc00:00ab::b/64 dev vethBA
    run ip netns exec routerB ifconfig vethBA hw ether 32:90:78:99:c1:32
    run ip netns exec routerB ip link set vethBA up
    run ip netns exec routerB ip -6 route add fc00:000a::/64 encap seg6 mode encap segs fc00:00ab::a dev vethB2
    
    # host2 configuration
    run ip netns exec host2 ip link set lo up
    run ip netns exec host2 ip ad add fc00:000b::10/64 dev veth2
    run ip netns exec host2 ifconfig veth2 hw ether 16:b5:08:6b:96:25 
    run ip netns exec host2 ip link set veth2 up
    run ip netns exec host2 ip -6 route add fc00::/16 via fc00:000b::b
}

destroy_network () {
    run ip netns  del host1
    run ip netns  del routerA
    run ip netns  del routerB
    run ip netns  del host2    
}

stop () {
    destroy_network
}
while getopts "cd" ARGS;
do
    case $ARGS in
    c ) create_network
        exit 1;;
    d ) destroy_network
        exit 1;;
    esac
done

cat << EOF
usage: sudo ./$(basename $BASH_SOURCE) <option>
option:
	-c : create_network
	-d : destroy_network
EOF

