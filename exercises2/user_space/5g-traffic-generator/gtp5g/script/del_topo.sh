#/bin/bash
#ip -all netns exec ip link show
ip netns del nsgNB
ip netns del nsUPF
ip netns del nsDN
