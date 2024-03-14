#!/bin/bash
expdev(){
    echo "exp nvme dev [subsystem] = $1 [nvme dev] = $2 [namespace number]=$3 [ip]=$4 [port]=$5"
    mkdir /sys/kernel/config/nvmet/subsystems/$1/namespaces/$3
    echo -n $2 > /sys/kernel/config/nvmet/subsystems/$1/namespaces/$3/device_path
    echo 1 > /sys/kernel/config/nvmet/subsystems/$1/namespaces/$3/enable
    mkdir /sys/kernel/config/nvmet/ports/$3
    echo $4 > /sys/kernel/config/nvmet/ports/$3/addr_traddr
    echo rdma > /sys/kernel/config/nvmet/ports/$3/addr_trtype
    echo $5 > /sys/kernel/config/nvmet/ports/$3/addr_trsvcid
    echo ipv4 > /sys/kernel/config/nvmet/ports/$3/addr_adrfam
    ln -s /sys/kernel/config/nvmet/subsystems/$1 /sys/kernel/config/nvmet/ports/$3/subsystems/$1
}
#modprobe nvmet
#modprobe nvmet-rdma
#modprobe nvme-rdma

mkdir /sys/kernel/config/nvmet/subsystems/data_8
echo 1 > /sys/kernel/config/nvmet/subsystems/data_8/attr_allow_any_host
expdev data_8 /dev/nvme0n1 800 192.168.11.22 6600


#host
# nvme connect -t rdma  -n  data_8   -a 192.168.11.22 -s 6600
