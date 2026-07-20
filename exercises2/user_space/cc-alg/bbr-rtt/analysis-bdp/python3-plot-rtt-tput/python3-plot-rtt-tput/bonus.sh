#!/bin/bash

# Note: Mininet must be run as root.  So invoke this shell script
# using sudo.

oldpwd=$PWD
dir=${1:-bonus}
mkdir -p $dir
rm -rf $dir/*

#!/bin/bash

# This shell script must be run using sudo.

run () {
    type=$1
    dir=$2

    oldpwd=$PWD
    mkdir -p $dir
    rm -rf $dir/*

    if [ "$type" = "smallbuffer" ]
    then
	maxq=200
    elif [ "$type" = "largebuffer" ]
    then
	maxq=600
    else
	exit "Unknown experiment type $type"
    fi

    echo "running $type experiment..."

    python flows.py --fig-num 7 --cong bbr --time 100 --bw-net 10 --delay 10 --maxq $maxq --environment mininet --flow-type iperf --dir $dir

    cd $dir
    echo "processing flows..."
    for i in 0 1; do
	captcp throughput -u Mbit -f 2 --stdio flow$i.dmp > captcp$i.txt
	awk '{print $1","$2 }' < captcp$i.txt > captcp-csv$i.txt
    done
    cd $oldpwd
    python plot_throughput.py --xlimit 100 -f $dir/captcp-csv* -o $dir/bonus_$type.png -l cubic bbr
}

if [ "${1-all}" = "all" ]
then
    run "smallbuffer" bonus_smallbuffer
    run "largebuffer" bonus_largebuffer
else
    run $1 bonus_$1
fi
