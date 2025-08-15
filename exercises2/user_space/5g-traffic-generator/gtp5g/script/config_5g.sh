#/bin/bash
#ip netns exec nsUPF ./gtp5g-link add gtp5gtest


ip netns exec nsUPF ip route add 60.0.0.0/24 dev gtp5gtest
ip netns exec nsDN ip route add default via 20.0.1.1
ip netns exec nsUPF ./gtp5g-tunnel add far gtp5gtest 1 --action 2
ip netns exec nsUPF ./gtp5g-tunnel add far gtp5gtest 2 --action 2 --hdr-creation 0 78 20.0.0.1 2152
ip netns exec nsUPF ./gtp5g-tunnel add pdr gtp5gtest 1 --pcd 1 --hdr-rm 0 --ue-ipv4 60.0.0.1 --f-teid 87 20.0.0.1 --far-id 1
ip netns exec nsUPF ./gtp5g-tunnel add pdr gtp5gtest 2 --pcd 2 --ue-ipv4 60.0.0.1 --far-id 2


ip netns exec nsUPF tc qdisc add dev veth1 ingress
ip netns exec nsUPF tc filter add dev veth1 parent ffff: protocol ip matchall action skbedit mark 1
ip netns exec nsUPF tc qdisc add dev veth2 ingress
ip netns exec nsUPF tc filter add dev veth2 parent ffff: protocol ip matchall action skbedit mark 2
echo "config complete \n"
