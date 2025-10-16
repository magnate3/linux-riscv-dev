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
    elif [ "$type" = "netperf" ]
    then
	environment=vms
	flowtype=netperf
    elif [ "$type" = "mininet" ]
    then
	environment=mininet
	flowtype=iperf
    else
	exit "Unknown experiment type $type"
    fi

    echo "running $type experiment..."

    destip=`su $SUDO_USER -c "cat ~/.bbr_pair_ip"`
    python flows.py --fig-num 6 --cong bbr --time 55 --bw-net 100 --delay 5 --maxq 1024 --environment $environment --flow-type $flowtype --dir $dir

    cd $dir
    echo "processing flows..."
    for i in 0 1 2 3 4; do
	captcp throughput -u Mbit --stdio flow$i.dmp > captcp$i.txt
	awk "{print (\$1+$i*2-1)(\",\")(\$2) }" < captcp$i.txt > captcp-csv$i.txt
    done
    cd $oldpwd
    python plot_throughput.py --xlimit 50 -f $dir/captcp-csv* -o $dir/figure6_$type.png
}

if [ "${1-all}" = "all" ]
then
    run "iperf" figure6_iperf
    run "netperf" figure6_netperf
    run "mininet" figure6_mininet
else
    run $1 figure6_$1
fi
