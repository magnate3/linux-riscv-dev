#!/bin/bash

dir=`date +%b%d--%H-%M`
start=`date`
time=20

server=triton01
client=triton02
sshopts="-t -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
workload=ib_read_bw
#set -e

function stopall {
    (pgrep -f ib_read | xargs sudo kill -9) > /dev/null 2>&1
    (pgrep -f ib_write | xargs sudo kill -9) > /dev/null 2>&1
    (pgrep -f roce | xargs sudo kill -9) > /dev/null 2>&1
}

function finish {
    echo -----------------------------------------
    echo started at $start
    echo finished at $(date)
    echo -----------------------------------------
    stopall
}

mkdir -p $dir

# Kill previous running instances
(pgrep -f roce | xargs sudo kill -9) > /dev/null 2>&1

for n in 1000000; do
for msize in 1; do
for p in 8; do
for q in 1; do
    edir=n$n-msize$msize-p$p-q$q
    mkdir -p $dir/$edir

    stopall

    if [ ! -d $dir/$edir ]; then
	exit
    fi

    # Start montor at the client, pass -p to log only the +ve rates -- i.e. don't log 0Gb/s
    for direction in tx rx; do
	ssh $sshopts $client python rocestats.py -i 0.1 --dir $direction > $dir/$edir/rate-$direction.txt &
    done

    # Start the server
    echo starting server at $server
    ssh $sshopts $server sudo python roce_40g_expt.py \
	-s -n $n --msize $msize -p $p -q $q --cmd $workload > $dir/$edir/server.txt 2>&1 &
    sleep 3

    # Start the client
    ssh $client $sshopts sudo python roce_40g_expt.py \
	-c $server -n $n --msize $msize -p $p -q $q --cmd $workload > $dir/$edir/client.txt 2>&1 &

    echo -----------------------------------------
    echo params $edir expt $dir
    echo -----------------------------------------
    echo Running experiment...

    # Run the experiment
    count=$time
    while [ "$count" -ne "0" ]; do
	sleep 1
	echo $count seconds remaining
	count=$(($count-1))
    done

    # Kill all
    jobs -p | xargs sudo kill -9
done
done
done
done

finish
