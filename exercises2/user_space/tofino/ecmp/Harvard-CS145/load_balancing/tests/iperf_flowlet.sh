#! /bin/bash

server=$((RANDOM % 16))
client=$((RANDOM % 16))
pods=$(($server / 4))
podc=$(($client / 4))
while [ "$pods" -eq "$podc" ]
do
    client=$((RANDOM % 16))
    podc=$(($client / 4))
done
aggl=$(($pods * 2 + 1))
aggr=$(($pods * 2 + 2))
server=$(($server + 1))
client=$(($client + 1))

rm -rf tcpdump_logs
mkdir tcpdump_logs

if [ "$1" -eq 1 ];
then
tcpdump -enn -i a$aggl-eth1 > tcpdump_logs/log${aggl}_1.output 2> /dev/null &
tcpdump -enn -i a$aggl-eth2 > tcpdump_logs/log${aggl}_2.output 2> /dev/null &
tcpdump -enn -i a$aggr-eth1 > tcpdump_logs/log${aggr}_1.output 2> /dev/null &
tcpdump -enn -i a$aggr-eth2 > tcpdump_logs/log${aggr}_2.output 2> /dev/null &
else
tcpdump -enn -i a$aggl-eth3 > tcpdump_logs/log${aggl}_1.output 2> /dev/null &
tcpdump -enn -i a$aggl-eth4 > tcpdump_logs/log${aggl}_2.output 2> /dev/null &
tcpdump -enn -i a$aggr-eth3 > tcpdump_logs/log${aggr}_1.output 2> /dev/null &
tcpdump -enn -i a$aggr-eth4 > tcpdump_logs/log${aggr}_2.output 2> /dev/null &
fi
for j in {1..1}
do
    ~/mininet/util/m h$server iperf3 -s --port 5001 2> /dev/null > /dev/null &
    sleep 0.5
    ~/mininet/util/m h$client iperf3 -c 10.0.0.$server -B 10.0.0.$client --port 5001 2> /dev/null > /dev/null &
    sleep 10
    pkill iperf3
done

pkill tcpdump
