export DEV=veth1
tc qdisc del dev $DEV clsact
rm -f /sys/fs/bpf/tc/globals/rate_map
tc qdisc del dev $DEV root handle 1: mq
# clang -O2 -emit-llvm -c example.c -o - | llc -march=bpf -filetype=obj -o example.o
tc qdisc add dev $DEV root handle 1: mq
NUM_TX_QUEUES=$(ls -d /sys/class/net/$DEV/queues/tx* | wc -l)
for (( i=1; i<=$NUM_TX_QUEUES; i++ ))
do
    tc qdisc add dev $DEV parent 1:$(printf '%x' $i) \
        handle $(printf '%x' $((i+1))): fq
done
# tc qdisc add dev $DEV clsact
# tc filter add dev $DEV egress bpf direct-action obj example.o sec .text
tc filter show dev $DEV egress
