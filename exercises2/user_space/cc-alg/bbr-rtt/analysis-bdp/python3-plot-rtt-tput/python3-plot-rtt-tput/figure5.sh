#!/bin/bash

# This shell script must be run using sudo.

run () {
    type=$1
    dir=$2

    oldpwd=$PWD
    mkdir -p $dir
    rm -rf $dir/*

    if [ "$type" = "iperf" ]
    then
	environment=vms
	flowtype=iperf
	flow=1
	maxq=150
    elif [ "$type" = "netperf" ]
    then
	environment=vms
	flowtype=netperf
	flow=0
	maxq=155

    elif [ "$type" = "mininet" ]
    then
	environment=mininet
	flowtype=iperf
	flow=1
	maxq=175
    else
	exit "Unknown experiment type $type"
    fi

    echo "running $type experiment..."
    #destip=`su $SUDO_USER -c "cat ~/.bbr_pair_ip"`
    python flows.py --fig-num 5 --time 8  --bw-net 10 --delay 20 --maxq $maxq --environment $environment --flow-type $flowtype --dir $dir
    #python flows.py --fig-num 5 --time 8 --dest-ip $destip --bw-net 10 --delay 20 --maxq $maxq --environment $environment --flow-type $flowtype --dir $dir
    chmod -R 0777 $dir
    su $SUDO_USER -c "tshark -2 -r $dir/flow_bbr.dmp -R ' tcp.analysis.ack_rtt'  -e frame.time_relative -e tcp.analysis.ack_rtt -Tfields -E separator=, > $dir/bbr_rtt.txt"
    #su $SUDO_USER -c "tshark -2 -r $dir/flow_bbr.dmp -R 'tcp.stream eq $flow && tcp.analysis.ack_rtt'  -e frame.time_relative -e tcp.analysis.ack_rtt -Tfields -E separator=, > $dir/bbr_rtt.txt"
    su $SUDO_USER -c "tshark -2 -r $dir/flow_cubic.dmp -R 'tcp.analysis.ack_rtt'  -e frame.time_relative -e tcp.analysis.ack_rtt -Tfields -E separator=, > $dir/cubic_rtt.txt"
    #su $SUDO_USER -c "tshark -2 -r $dir/flow_cubic.dmp -R 'tcp.stream eq $flow && tcp.analysis.ack_rtt'  -e frame.time_relative -e tcp.analysis.ack_rtt -Tfields -E separator=, > $dir/cubic_rtt.txt"

    python plot_ping.py -f $dir/bbr_rtt.txt $dir/cubic_rtt.txt --xlimit 8 -o $dir/figure5_$type.png
}

if [ "${1-all}" = "all" ]
then
    run "iperf" figure5_iperf
    run "netperf" figure5_netperf
    run "mininet" figure5_mininet
else
    run $1 figure5_$1
fi
