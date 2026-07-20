#/bin/bash
ip netns add nat64
ip link add veth6 type veth peer name veth7
ip link set veth6 up
ifconfig veth6 hw ether  aa:70:e3:5e:3d:0a  
ip a add 64:ff9b::0a00:0102/96 dev veth6
ip a add 10.0.1.2/24 dev veth6
ip n add  64:ff9b::0a00:0101 dev veth6 lladdr  aa:80:e3:5e:3d:0b
ip link set veth7 up
ip link set veth7 netns nat64
ip netns exec nat64 ip a add 64:ff9b::0a00:0101/96 dev veth7
ip netns exec nat64 ip a add 10.0.1.1/24 dev veth7
ip netns exec nat64 ifconfig veth7 hw ether  aa:80:e3:5e:3d:0b  
ip netns exec nat64 ip n add  64:ff9b::0a00:0102 dev veth7 lladdr  aa:70:e3:5e:3d:0a

