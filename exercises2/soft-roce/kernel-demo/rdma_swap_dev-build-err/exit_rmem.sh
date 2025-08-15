#!/bin/sh
for s in $(ls /dev/rmem_rdma*)
do
  while [ -n "$(cat /proc/swaps | grep $s)" ]
  do
    swapoff -a
  done
done

umount /root/temp

while [ -n "$(lsmod | grep rmem_rdma)" ]
do
  rmmod rmem_rdma
done


