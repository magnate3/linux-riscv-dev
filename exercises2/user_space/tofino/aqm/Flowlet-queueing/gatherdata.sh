#!/bin/bash

MACHINES=("nhpcc416@172.16.1.101" "nhpcc416@172.16.1.102" "nhpcc416@172.16.1.103" "nhpcc416@172.16.1.104")

for ((i=8100;i<=8114;i++)) do
    logfile=iperf3c_${i}.log
    scp ${MACHINES[0]}:/tmp/${logfile} ./iperf3c_${i}.json
#    cat iperf3c_${i}.log | tail +4 | head -90 | awk '/sec/ && $5 ~ /[0-9]/ {print $7, $8}' > bw_${i}.txt
done

for ((i=8200;i<=8214;i++)) do
    logfile=iperf3c_${i}.log
    scp ${MACHINES[1]}:/tmp/${logfile} ./iperf3c_${i}.json
done

for ((i=8300;i<=8314;i++)) do
    logfile=iperf3c_${i}.log
    scp ${MACHINES[2]}:/tmp/${logfile} ./iperf3c_${i}.json
done
