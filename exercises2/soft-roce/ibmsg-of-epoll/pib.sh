#!/bin/bash

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

if [[ $# -ne 1 ]]; then
	echo "Usage: pib.sh (start|stop)"
	exit 1
fi

if [[ "$1" == "start" ]]; then
	service rdma start
	modprobe pib
	service opensm start

    ip addr add 192.168.100.100/24 dev ib0                                                                                                                                                                                                         
    ip addr add 192.168.100.101/24 dev ib1
    ip addr add 192.168.100.102/24 dev ib2
    ip addr add 192.168.100.103/24 dev ib3

    ip link set ib0 up
    ip link set ib1 up
    ip link set ib2 up
    ip link set ib3 up

elif [[ "$1" == "stop" ]]; then
	service opensm stop
	rmmod pib
	service rdma stop
else
	echo "Usage: pib.sh (start|stop)"
	exit 1
fi
