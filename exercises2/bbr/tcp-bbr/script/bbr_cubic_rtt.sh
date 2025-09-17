#tc qdisc add dev  enp5s0 root pfifo limit 1000000
# tc qdisc add dev  enp5s0 root fq_codel limit 1000000
# tc qdisc add dev enp5s0 root netem delay 10ms 15ms distribution normal
tc qdisc del dev enp5s0 root
# 为打向 5201 端口的流打标签 10
iptables -A OUTPUT -t mangle -p tcp --dport 5201 -j MARK --set-mark 10
# 为打向 5202 端口的流打标签 20
iptables -A OUTPUT -t mangle -p tcp --dport 5202 -j MARK --set-mark 20

tc qdisc add dev enp5s0 root handle 1: htb
tc class add dev enp5s0 parent 1: classid 1:1 htb rate 10gbit
tc class add dev enp5s0 parent 1:1 classid 1:10 htb rate 5gbit
tc class add dev enp5s0 parent 1:1 classid 1:20 htb rate 5gbit

# filter 1 关联标签 10 
tc filter add dev enp5s0 protocol ip parent 1:0 prio 1 handle 10 fw flowid 1:10
# filter 2 关联标签 20
tc filter add dev enp5s0 protocol ip parent 1:0 prio 1 handle 20 fw flowid 1:20

# 标签 10 的 5201 流时延 2ms，丢包 1%
tc qdisc add dev enp5s0 parent 1:10 handle 10: netem delay 10ms loss 1
# 标签 20 的 5202 流时延 20ms，丢包 1%
tc qdisc add dev enp5s0 parent 1:20 handle 20: netem delay 10ms loss 1
#tc qdisc add dev enp5s0 parent 1:20 handle 20: netem delay 10ms  15ms 50%  loss 3
