#! /bin/bash

# This script will create(remove) veth/host attached to namespace
# and corresponding tap interface.
# Most of the code was copied from a script written by Tohru Kitamura. Thanks!!

if [[ $(id -u) -ne 0 ]] ; then echo "Please run with sudo" ; exit 1 ; fi

set -e

if [ -n "$SUDO_UID" ]; then
    uid=$SUDO_UID
else
    uid=$UID
fi

run () {
    echo "$@"
    "$@" || exit 1
}

silent () {
    "$@" 2> /dev/null || true
}

create_network () {
    echo "create_network"
    # Create network namespaces
    run ip netns add host0
    run ip netns add host1

    # Create veth
    run ip link add veth0 type veth peer name vtap0
    run ip link add veth1 type veth peer name vtap1
	# Create vtap pair to connect back to back
    run ip link add vtap11 type veth peer name vtap12
    run ip link set dev vtap11 up
    run ip link set dev vtap12 up
    run ip link add vtap13 type veth peer name vtap14
    run ip link set dev vtap13 up
    run ip link set dev vtap14 up
    run ip link add vtap15 type veth peer name vtap16
    run ip link set dev vtap15 up
    run ip link set dev vtap16 up

    # Connect veth between host0 and host1
    run ip link set veth0 netns host0
    run ip link set veth1 netns host1
    run ip link set dev vtap0 up
    run ip link set dev vtap1 up

    # Link up loopback and veth
    run ip netns exec host0 ip link set veth0 up
    run ip netns exec host0 ifconfig lo up
    run ip netns exec host1 ip link set veth1 up
    run ip netns exec host1 ifconfig lo up

    # Set IPv4 address
    run ip netns exec host0 ip addr add 172.20.0.1/24 dev veth0
    run ip netns exec host1 ip addr add 172.20.0.2/24 dev veth1
	# Set IPv6 address
	run ip netns exec host0 ip -6 addr add 2001:db8:a::1/64 dev veth0
	run ip netns exec host0 ifconfig veth0  hw ether  a2:13:5d:e3:5b:84
	run ip netns exec host0 ip -6 neigh add 2001:db8:a::2 lladdr  8e:a4:e3:14:21:97 dev veth0
	run ip netns exec host1 ip -6 addr add 2001:db8:a::2/64 dev veth1
	run ip netns exec host1 ifconfig veth1  hw ether  8e:a4:e3:14:21:97
	run ip netns exec host1 ip -6 neigh add 2001:db8:a::1 lladdr a2:13:5d:e3:5b:84 dev veth1

    run ip link set dev vtap0 up
    run ip link set dev vtap1 up
}

destroy_network () {
    echo "destroy_network"
    silent ip link del veth0
    silent ip link del veth1
    silent ip netns del host0
    silent ip netns del host1

    silent ip link del vtap11
    silent ip link del vtap13
    silent ip link del vtap15
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
