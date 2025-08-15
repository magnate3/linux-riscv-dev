#
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
#

#!/usr/bin/env bash
######################################################################
# 设置Mellanox ConnectX网卡优化参数                                     #
# 在集群测试环境中，针对目标集群的网络进行测试（应用、benchmark、打压工具），      #
# 根据测试结果调整交换机和网卡的设置，以达成整体最优网络。                       #
# 前提条件：完成Mellanox ConnectX网卡驱动的安装。                          #
######################################################################

for mlx_dev in $(ibdev2netdev |grep enp61s0f1np1 | awk '{print $1}')
do
   if_dev=$(ibdev2netdev | grep $mlx_dev | awk '{print $5}')
   echo "------------> Current: ${mlx_dev}:${if_dev}"
   # Check DCQCN is enabled on Prio 3
   IF_NAME="enp61s0f1np1"
   DEV_NAME="mlx5_1"
   cat /sys/class/net/$IF_NAME/ecn/roce_np/enable/3
   cat /sys/class/net/$IF_NAME/ecn/roce_rp/enable/3

   # Check counters related to DCQCN
   cat /sys/class/infiniband/$DEV_NAME/ports/1/hw_counters/np_cnp_sent
   cat /sys/class/infiniband/$DEV_NAME/ports/1/hw_counters/np_ecn_marked_roce_packets
   cat /sys/class/infiniband/$DEV_NAME/ports/1/hw_counters/rp_cnp_handled
done
