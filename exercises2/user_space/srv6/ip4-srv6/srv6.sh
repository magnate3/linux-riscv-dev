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

destroy_network () {
    run ip netns  del  r1 
    run ip netns  del  r2 
    run ip netns  del  r3 
    run ip netns  del  r4 
    run ip netns  del host-a    
    run ip netns  del host-b    
    run ip netns  del host-c    
}
### 创建网络命名空间
create_network () {
    ip netns add r1
    ip netns add r2
    ip netns add r3
    ip netns add r4
    ip netns add host-a
    ip netns add host-b
    ip netns add host-c
    arr=("r1" "r2" "r3" "r4" "host-a" "host-b" "host-c" )
    for value in ${arr[@]}
    do
	#echo $value
        ip netns exec $value sysctl -w net.ipv4.ip_forward=1
        ip netns exec $value sysctl -w net.ipv4.conf.all.rp_filter=0
        ip netns exec $value sysctl -w net.ipv6.conf.all.forwarding=1
        ip netns exec $value sysctl -w net.ipv6.conf.all.seg6_enabled=1
        ip netns exec $value sysctl -w net.ipv4.conf.default.rp_filter=0
        ip netns exec $value sysctl -w net.ipv6.conf.default.forwarding=1
        ip netns exec $value sysctl -w net.ipv6.conf.default.seg6_enabled=1
        ip netns exec $value sysctl -w net.ipv4.conf.lo.rp_filter=0
        ip netns exec $value sysctl -w net.ipv6.conf.lo.forwarding=1
        ip netns exec $value sysctl -w net.ipv6.conf.lo.seg6_enabled=1
    done
### 创建虚拟网卡设备
ip link add dev-a-r1 netns host-a type veth peer name dev-r1-a netns r1
ip link add dev-r1-r2 netns r1 type veth peer name dev-r2-r1 netns r2
ip link add dev-r2-r3 netns r2 type veth peer name dev-r3-r2 netns r3
ip link add dev-r3-c netns r3 type veth peer name dev-c-r3 netns host-c
ip netns exec r1 ip link set dev-r1-a up
ip netns exec host-a ip link set dev-a-r1 up
ip netns exec r1 ip link set dev-r1-r2 up
ip netns exec r2 ip link set dev-r2-r1 up
ip netns exec r2 ip link set dev-r2-r3 up
ip netns exec r3 ip link set dev-r3-r2 up
ip netns exec r3 ip link set dev-r3-c up
ip netns exec host-c ip link set dev-c-r3 up

ip link add dev-r1-r4 netns r1 type veth peer name dev-r4-r1 netns r4
ip link add dev-r2-r4 netns r2 type veth peer name dev-r4-r2 netns r4
ip link add dev-r3-r4 netns r3 type veth peer name dev-r4-r3 netns r4
ip link add dev-r4-b netns r4 type veth peer name dev-b-r4 netns host-b
ip netns exec r1 ip link set dev-r1-r4 up
ip netns exec r4 ip link set dev-r4-r1 up
ip netns exec r2 ip link set dev-r2-r4 up
ip netns exec r4 ip link set dev-r4-r2 up
ip netns exec r3 ip link set dev-r3-r4 up
ip netns exec r4 ip link set dev-r4-r3 up
ip netns exec host-b ip link set dev-b-r4 up
ip netns exec r4 ip link set dev-r4-b up

ip netns exec r1 ip link set lo up
ip netns exec r2 ip link set lo up
ip netns exec r3 ip link set lo up
ip netns exec r4 ip link set lo up
##===========================

### 配置地址、路由
ip netns exec r1 ip addr add 10.0.0.2/24 dev dev-r1-a
ip netns exec r1 ip addr add 2001:1a::1/64 dev dev-r1-a
ip netns exec r1 ip addr add 2001:12::1/64 dev dev-r1-r2
ip netns exec r1 ip addr add 2001:14::1/64 dev dev-r1-r4
ip netns exec r1 ip addr add fc00:1::1/64 dev lo

ip netns exec r2 ip addr add 2001:12::2/64 dev dev-r2-r1
ip netns exec r2 ip addr add 2001:23::1/64 dev dev-r2-r3
ip netns exec r2 ip addr add 2001:24::1/64 dev dev-r2-r4
ip netns exec r2 ip addr add fc00:2::2/64 dev lo

ip netns exec r3 ip addr add 10.0.1.2/24 dev dev-r3-c
ip netns exec r3 ip addr add 2001:23::2/64 dev dev-r3-r2
ip netns exec r3 ip addr add 2001:3c::1/64 dev dev-r3-c
ip netns exec r3 ip addr add 2001:34::1/64 dev dev-r3-r4
ip netns exec r3 ip addr add fc00:3::3/64 dev lo


ip netns exec r4 ip addr add 10.0.2.2/24 dev dev-r4-b
ip netns exec r4 ip addr add 2001:14::2/64 dev dev-r4-r1
ip netns exec r4 ip addr add 2001:24::2/64 dev dev-r4-r2
ip netns exec r4 ip addr add 2001:34::2/64 dev dev-r4-r3
ip netns exec r4 ip addr add 2001:4b::1/64 dev dev-r4-b
ip netns exec r4 ip addr add fc00:4::4/64 dev lo

ip netns exec host-a ip addr add 2001:1a::2/64 dev dev-a-r1
ip netns exec host-c ip addr add 2001:3c::2/64 dev dev-c-r3
ip netns exec host-b ip addr add 2001:4b::2/64 dev dev-b-r4

ip netns exec host-a ip addr add 10.0.0.1/24 dev dev-a-r1
ip netns exec host-c ip addr add 10.0.1.1/24 dev dev-c-r3
ip netns exec host-b ip addr add 10.0.2.1/24 dev dev-b-r4

ip netns exec host-a ip route add 10.0.1.0/24 via 10.0.0.2
ip netns exec host-a ip route add 10.0.2.0/24 via 10.0.0.2
ip netns exec host-c ip route add 10.0.0.0/24 via 10.0.1.2
ip netns exec host-b ip route add 10.0.0.0/24 via 10.0.2.2
#ip netns exec host-b ip route add 10.0.0.0/24 dev dev-b-r4

ip netns exec r1 ip sr tunsrc set fc00:1::1
ip netns exec r2 ip sr tunsrc set fc00:2::2
ip netns exec r3 ip sr tunsrc set fc00:3::3
ip netns exec r4 ip sr tunsrc set fc00:4::4

ip netns exec r1 ip -6 route add default via 2001:12::2

ip netns exec r2 ip -6 route add fc00:1::/64 via 2001:12::1
ip netns exec r2 ip -6 route add fc00:3::/64 via 2001:23::2

ip netns exec r3 ip -6 route add default via 2001:23::1
ip netns exec r3 ip -6 route add fc00:4::/64 via 2001:34::2
#ip netns exec r3 ip -6 route add fc00:1::bb/128 via 2001:23::1

ip netns exec r4 ip -6 route add default via 2001:4b::2
ip netns exec r4 ip route add default via 10.0.2.1
ip netns exec r4 ip -6 route add fc00:1::bb/128 via 2001:14::1
### 配置SRv6指令
ip netns exec r1 ip -6 route add fc00:1::bb/128 encap seg6local action End.DX4 nh4 10.0.0.1 dev dev-r1-a
ip netns exec r1 ip route add 10.0.2.0/24 encap seg6 mode encap segs fc00:3::bb,fc00:4::bb dev dev-r1-r2
ip netns exec r3 ip -6 route add fc00:3::bb/128 encap seg6local action End dev dev-r3-r4
ip netns exec r4 ip -6 route add fc00:4::bb/128 encap seg6local action End.DX4 nh4 10.0.2.1 dev dev-r4-b
ip netns exec r4 ip route add 10.0.0.0/24 encap seg6 mode encap segs fc00:1::bb dev dev-r4-r1
#ip netns exec r1 ip route add 10.0.1.0/24 encap seg6 mode encap segs fc00:3::bb dev dev-r1-a
#ip netns exec r3 ip -6 route add fc00:3::bb/128 encap seg6local action End.DX4 nh4 10.0.1.1 dev dev-r3-c
#ip netns exec r3 ip route add 10.0.0.0/24 encap seg6 mode encap segs fc00:1::bb dev dev-r3-c
}
:<<! 
###################
!
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

