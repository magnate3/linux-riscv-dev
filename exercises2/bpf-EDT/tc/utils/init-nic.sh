ip netns del ns1
ip netns del ns2
ip netns add ns1
ip netns add ns2
ip link add veth0 type veth peer name veth1
ip link set veth0 netns ns1
ip link set veth1 netns ns2
ip netns exec ns1 ip link set dev veth0 up
ip netns exec ns2 ip link set dev veth1 up
ip netns exec ns1 ip addr add 10.10.10.10/24 dev veth0
ip netns exec ns2 ip addr add 10.10.10.11/24 dev veth1