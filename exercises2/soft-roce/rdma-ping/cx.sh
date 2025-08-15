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
   #配置接口mtu 为4500
   ifconfig "${if_dev}" mtu 4500.                        
   #配置 RDMA-CM QP 默认 TOS为106
   cma_roce_tos -d "${mlx_dev}" -t 106
   #配置4队列pfc使能
   mlnx_qos -i "${if_dev}" --pfc 0,0,0,0,1,0,0,0 --trust dscp
   #配置dscp=26(tos/4)报文映射到4队列
   mlnx_qos -i ${if_dev} --dscp2prio set,26,4
   #配置 cnp报文dscp 为48；使能4队列ECN功能
   echo 48 >/sys/class/net/${if_dev}/ecn/roce_np/cnp_dscp
   echo 1 >/sys/class/net/${if_dev}/ecn/roce_np/enable/4
   echo 1 >/sys/class/net/${if_dev}/ecn/roce_rp/enable/4
done
