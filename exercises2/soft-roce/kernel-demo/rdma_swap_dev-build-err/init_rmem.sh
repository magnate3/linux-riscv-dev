#!/bin/sh

if [ -n "$(cat /proc/swaps | grep /mnt/swap)" ]
then
  swapoff /mnt/swap
fi
insmod rmem_rdma.ko servers=10.10.49.89:18516:$((3*786432))

for s in $(ls /dev/rmem_rdma*);
do
  sleep 1
  mkswap -f $s
  sleep 1
  swapon $s
done
            
