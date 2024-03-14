Note，如果有防火墙，需要放开udp 4789端口：
iptables -t nat -S
iptables -t nat -A PREROUTING -d  210.22.22.151   -p tcp -m tcp --dport 1685 -j DNAT --to-destination 10.10.16.48:6000
iptables -A DOCKER -d 172.17.0.4/32 ! -i docker0 -o docker0 -p tcp -m tcp --dport 6000 -j ACCEPT
iptables -t nat -I PREROUTING 1 -d  10.10.16.48   -p tcp -m tcp --dport 6000 -j DNAT --to-destination 172.17.0.4:6000
iptables -t nat  -A POSTROUTING ! -s 10.10.16.1/32 -d 192.168.117.0/24 -o eth0 -j SNAT --to-source 10.10.16.1
ip6tables -S | grep icmp | grep DROP
iptables -I INPUT -p udp --dport 4789 -j ACCEPT
iptables -nv -t nat -L POSTROUTING
iptables -nv -t nat -L POSTROUTING --line
iptables -t nat  -L POSTROUTING  -n --line-number 
iptables -t nat  -D POSTROUTING 4
### 不能写成 14.0.0.93/32
iptables -t nat -A PREROUTING -d  14.0.0.93   -p tcp -m tcp --dport 1069 -j DNAT --to-destination 10.10.16.81:22
 iptables -t nat -F
 iptables -t nat -A OUTPUT -p tcp --dport 1069 -j REDIRECT  --to-ports 22

================= 10.10.18.193 --> 192.168.117.144 via 10.10.16.81 ===============
iptables -t nat -A PREROUTING -d  10.10.16.81   -p tcp -m tcp --dport 1069 -j DNAT --to-destination 192.168.117.144:22
iptables -t nat -A POSTROUTING ! -s 10.10.16.81   -d  192.168.117.144   -j MASQUERADE
================= 10.10.18.193 --> 192.168.117.144 via 10.10.16.81 ===============


iptables -t nat -A POSTROUTING -o enp1s0 -j MASQUERADE
=================begoit 210.22.22.151 
iptables -t nat -A POSTROUTING ! -s 10.10.16.1   -d        192.168.117.122   -j MASQUERADE
=================begoit 210.22.22.151 
iptables -A INPUT -i evpn-vrf  -j ACCEPT
iptables -t nat -A POSTROUTING -o default_g -j MASQUERADE
