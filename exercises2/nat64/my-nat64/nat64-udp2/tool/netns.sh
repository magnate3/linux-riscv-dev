ip netns add ns1
ip link add veth1 type veth peer name veth2
ip link set veth2 netns ns1
ip addr add 10.10.103.81/24 dev veth1
ip -n ns1 addr add 10.10.103.82/24 dev veth2
ifconfig veth1 hw ether 72:26:fe:61:ca:65
ip  link set veth1 up
ip  -n ns1 link set veth2 up
ip  -n ns1 ifconfig veth2 hw ether 22:55:4e:94:6f:4d
