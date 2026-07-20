#/bin/bash
#ip -all netns exec ip link show
ip netns add nsgNB
ip netns add nsUPF
ip netns add nsDN
ip link add veth0 type veth peer  veth1
ip link add veth2 type veth peer  veth3
ip link set veth0 netns nsgNB
ip link set veth1 netns nsUPF
ip link set veth2 netns nsUPF
ip link set veth3 netns nsDN
ip netns exec nsgNB ip link set veth0 up
ip netns exec nsUPF ip link set veth1 up
ip netns exec nsUPF ip link set veth2 up
ip netns exec nsDN ip link set veth3 up
ip netns exec nsgNB ip addr add 20.0.0.1/24 dev veth0
ip netns exec nsUPF ip addr add 20.0.0.2/24 dev veth1
ip netns exec nsUPF ip addr add 20.0.1.1/24 dev veth2
ip netns exec nsDN ip addr add 20.0.1.2/24 dev veth3
