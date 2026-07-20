#!/bin/bash

#NICS=("enp41s0f1np1" "enp170s0f1np1")
NICS=("enp61s0f1np1")

echo "=== RoCEv2 Tuning Verification ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "----------------------------------"

for nic in "${NICS[@]}"; do
  dev=$(ibdev2netdev | grep -w "$nic" | awk '{print $1}')
  echo "NIC: $nic"
  echo "Device: ${dev:-Unknown}"
  cma_roce_mode -d ${dev} -p 1 -m 2

  echo 106 >  /sys/class/infiniband/"$dev"/tc/1/traffic_class
  echo 1 > /sys/class/net/"$nic"/ecn/roce_rp/enable/3
  echo 1 > /sys/class/net/"$nic"/ecn/roce_np/enable/3
  echo "----------------------------------"
done

