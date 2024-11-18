ip netns exec host1 ip addr add 192.168.0.1/32 dev lo
ip netns exec host1 gtp-link add gtp1 ip &
ip netns exec host1 gtp-tunnel add gtp1 v1 200 100 192.168.0.2 172.20.0.2
ip netns exec host1 ip route add 192.168.0.2/32 dev gtp1

ip netns exec host2 ip addr add 192.168.0.2/32 dev lo
ip netns exec host2 gtp-link add gtp2 ip &
ip netns exec host2 gtp-tunnel add gtp2 v1 100 200 192.168.0.1 172.20.0.1
ip netns exec host2 ip route add 192.168.0.1/32 dev gtp2
