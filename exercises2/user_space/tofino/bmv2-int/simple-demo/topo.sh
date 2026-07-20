#! /bin/bash
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

destroy_mirror() {
    run ip netns del m_ns1
    run ip netns del m_ns2
    run ip netns del m_ns3
}
create_mirror() {
    run ip netns add m_ns1
    run ip link add name mirror1 type veth peer name sw1_p3
    run ip link set mirror1 netns m_ns1 
    run ip netns exec m_ns1 ip link set mirror1 up
    run ip netns exec m_ns1  ifconfig mirror1 hw ether fa:9c:98:ef:a4:20
    run ip link set sw1_p3 up
    run ip netns add m_ns2
    run ip link add name mirror2 type veth peer name sw2_p3
    run ip link set mirror2 netns m_ns2 
    run ip netns exec m_ns2 ip link set mirror2 up
    run ip netns exec m_ns2  ifconfig mirror2 hw ether fa:9c:98:ef:a4:21
    run ip link set sw2_p3 up
    run ip netns add m_ns3
    run ip link add name mirror3 type veth peer name sw3_p3
    run ip link set mirror3 netns m_ns3
    run ip netns exec m_ns3 ip link set mirror3 up
    run ip link set sw3_p3 up
    run ip netns exec m_ns3  ifconfig mirror3 hw ether fa:9c:98:ef:a4:22
    run ip netns exec m_ns3  ip a add 10.0.4.2/16 dev mirror3
    run ip netns exec m_ns3  ip n add 10.0.3.2  dev mirror3 lladdr 52:49:c6:f8:3f:12
    run ip netns exec ns2  ip n add 10.0.4.2  dev host2 lladdr fa:9c:98:ef:a4:22
    #run    ifconfig sw1_p3 promisc
    #run    ifconfig sw2_p3 promisc
    #run    ifconfig sw2_p3 promisc
}
destroy_network () {
    run ip netns del ns1
    run ip netns del ns2
    run ip link del sw1_p2
    run ip link del sw2_p2
    run ip link del sw3_p2
}
### 创建网络命名空间
create_network () {
    run    ip link add name host1 type veth peer name sw1_p1
    run    ip link add name sw1_p2 type veth peer name sw2_p1
    run    ip link add name sw2_p2 type veth peer name sw3_p1
    run    ip link add name sw3_p2 type veth peer name host2
    run    ip link set sw1_p1 up
    run    ip link set sw1_p2 up
    run    ip link set sw2_p1 up
    run    ip link set sw2_p2 up
    run    ip link set sw3_p1 up
    run    ip link set sw3_p2 up
    #run    ifconfig sw1_p1 promisc
    #run    ifconfig sw1_p2 promisc
    #run    ifconfig sw2_p1 promisc
    #run    ifconfig sw2_p2 promisc
    #run    ifconfig sw3_p1 promisc
    #run    ifconfig sw3_p2 promisc
    run    ip netns add ns1
    run    ip netns add ns2
    run    ip link set host1 netns ns1
    run    ip link set host2 netns ns2
    run    ip netns exec ns1 ip link set  host1 up
    run    ip netns exec ns2 ip link set  host2 up
    run    ip netns exec ns1 ip a add 10.0.1.1/16 dev  host1
    run    ip netns exec ns1 ifconfig host1 hw ether 52:49:c6:f8:3f:11  
    run    ip netns  exec ns1 ethtool -K host1  tx-checksum-ip-generic off
    run    ip netns exec ns1 ip n add 10.0.3.2  dev host1 lladdr 2e:2e:21:21:00:12
    run    ip netns exec ns2 ip a add 10.0.3.2/16 dev  host2
    run    ip netns exec ns2 ifconfig host2 hw ether 52:49:c6:f8:3f:12  
    run    ip netns exec ns2 ip n add 10.0.1.1  dev host2 lladdr 52:49:c6:f8:3f:11
    run    ip netns  exec ns2 ethtool -K host2  tx-checksum-ip-generic off
}
while getopts "cdef" ARGS;
do
    case $ARGS in
    c ) create_network
        exit 1;;
    d ) destroy_network
        exit 1;;
    e ) create_mirror
        exit 1;;
    f ) destroy_mirror
        exit 1;;
    esac
done


