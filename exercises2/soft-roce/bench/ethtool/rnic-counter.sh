#!/bin/bash
# https://community.mellanox.com/docs/DOC-2572 
# Author: Animesh Trivedi 

h1="TX(bytes)"
h2="RX(bytes)"
h3="TX(pkts)"
h4="RX(pkts)"
count=0
header=false
if [ $# -ne 1 ]; then
	echo "[ERROR] you have to tell me which NIC?"
	exit 1
fi
while true; do
    ARRAY0=($(ethtool -S $1 | grep rdma_unicast | awk '{print $2}'))
    sleep 1
    ARRAY1=($(ethtool -S $1 | grep rdma_unicast | awk '{print $2}'))
    
    RX_PKTS=$((${ARRAY1[0]} - ${ARRAY0[0]})) 
    RX_BYTES=$((${ARRAY1[1]} - ${ARRAY0[1]})) 
    TX_PKTS=$((${ARRAY1[2]} - ${ARRAY0[2]})) 
    TX_BYTES=$((${ARRAY1[3]} - ${ARRAY0[3]})) 
    if [ $(($count % 10)) == 0 ] && [ $header = true ]; then
            printf "\n%-10s  %-10s | %-10s %-10s \n" $h1 $h2 $h3 $h4 
    fi
    count=$(($count+1))
    printf "%-10s  %-10s | %-10s %-10s \n" $TX_BYTES $RX_BYTES $TX_PKTS $RX_PKTS 
done
