ip tuntap add dev tap0 mode tap user root
ip link set dev tap0 up
brctl addbr br0 
ip link set tap0 master br0
ifconfig br0 up
ip a add 192.168.11.33/24 dev br0
iptables -t nat -A POSTROUTING  ! -d  192.168.11.22/24  -s 192.168.11.22 -o wlxe0e1a91deeb2 -j MASQUERADE
iptables -t nat -A POSTROUTING  ! -d  192.168.11.44/24  -s 192.168.11.44 -o wlxe0e1a91deeb2 -j MASQUERADE
