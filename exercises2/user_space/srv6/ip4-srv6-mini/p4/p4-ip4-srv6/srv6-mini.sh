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
    #run ip netns  del  r1 
    run ip netns  del  r4 
    run ip netns  del host-a    
    run ip netns  del host-b    
}
### 创建网络命名空间
create_network () {
    #ip netns add r1
    ip netns add r4
    ip netns add host-a
    ip netns add host-b
    arr=("r4" "host-a" "host-b" )
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
ip link add dev-a-r1 netns host-a type veth peer name dev-r1-a 
ip netns exec host-a ip link set dev-a-r1 up

ip link add dev-r4-r1 netns r4 type veth peer name dev-r1-r4 
ip link add dev-r4-b netns r4 type veth peer name dev-b-r4 netns host-b
ip link set dev-r1-a up
ip link set dev-r1-r4 up
ip netns exec r4 ip link set dev-r4-r1 up
ip netns exec host-b ip link set dev-b-r4 up
ip netns exec r4 ip link set dev-r4-b up

ip netns exec r4 ip link set lo up
##===========================

### 配置地址、路由
ifconfig dev-r1-a hw ether 16:b5:08:6b:96:25
ip addr add 10.0.0.2/24 dev dev-r1-a
ip addr add 2001:1a::1/64 dev dev-r1-a
ip addr add 2001:14::1/64 dev dev-r1-r4




ip netns exec r4 ip addr add 10.0.2.2/24 dev dev-r4-b
ip netns exec r4 ip addr add 2001:14::2/64 dev dev-r4-r1
ip netns exec r4 ifconfig dev-r4-r1 hw ether 26:30:f2:fe:dd:c1
ip netns exec r4 ip addr add 2001:4b::1/64 dev dev-r4-b
ip netns exec r4 ip addr add fc00:4::4/64 dev lo

ip netns exec host-a ip addr add 2001:1a::2/64 dev dev-a-r1
ip netns exec host-b ip addr add 2001:4b::2/64 dev dev-b-r4

ip netns exec host-a ifconfig dev-a-r1 hw ether e2:fc:43:11:c0:20
ip netns exec host-a ip addr add 10.0.0.1/24 dev dev-a-r1
ip netns exec host-a ip  neigh add  10.0.0.2  lladdr 16:b5:08:6b:96:25 nud permanent dev dev-a-r1
ip netns exec host-b ip addr add 10.0.2.1/24 dev dev-b-r4

ip netns exec host-a ip route add 10.0.1.0/24 via 10.0.0.2
ip netns exec host-a ip route add 10.0.2.0/24 via 10.0.0.2
ip netns exec host-b ip route add 10.0.0.0/24 via 10.0.2.2
#ip netns exec host-b ip route add 10.0.0.0/24 dev dev-b-r4
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ip sr tunsrc set fc00:1::1
ip netns exec r4 ip sr tunsrc set fc00:4::4



ip netns exec r4 ip -6 route add default via 2001:4b::2
ip netns exec r4 ip route add default via 10.0.2.1
ip netns exec r4 ip -6 route add fc00:1::bb/128 via 2001:14::1
### 配置SRv6指令
ip netns exec r4 ip -6 route add fc00:4::bb/128 encap seg6local action End.DX4 nh4 10.0.2.1 dev dev-r4-b
ip netns exec r4 ip route add 10.0.0.0/24 encap seg6 mode encap segs fc00:1::bb dev dev-r4-r1
:<<! 
!
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

