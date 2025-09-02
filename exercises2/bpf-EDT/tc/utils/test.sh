#!/bin/bash

cmd1="iperf  -i 1  -c  10.10.10.10  -p 20288  -t 3"
cmd2="iperf -s -i 1  -p  20288  -B 10.10.10.10"



ip netns exec ns1 $cmd2 &
# pid2=$!

ip netns exec ns2 $cmd1 > /dev/null &
# pid1=$!

sleep 5

ip netns exec ns2 killall iperf
ip netns exec ns1 killall iperf


