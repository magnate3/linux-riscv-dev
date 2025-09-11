#!/bin/bash

# 检查参数
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <interface> [output_file]"
    echo "Example: $0 eth0 tcp_net_info.txt"
    exit 1
fi

# 获取网卡名参数
INTERFACE=$1
shift  # 移除第一个参数

# 设置输出文件
OUT=${1:-tcp_net_info_$(hostname).txt}
echo "Collecting TCP and system network info for $INTERFACE into $OUT ..."

{
echo "===== Hostname ====="
hostname

echo -e "\n===== Kernel version ====="
uname -r

echo -e "\n===== TCP Congestion Control ====="
sysctl net.ipv4.tcp_congestion_control

echo -e "\n===== TCP Advanced Parameters ====="
sysctl -a | grep -E 'tcp_(rmem|wmem|adv_win_scale|limit_output_bytes|max_syn_backlog|max_orphans|low_latency|no_metrics_save)' 2>/dev/null

echo -e "\n===== NIC Offload Settings ($INTERFACE) ====="
ethtool -k $INTERFACE 2>/dev/null || echo "ethtool $INTERFACE not found"

echo -e "\n===== NIC Driver Info ====="
ethtool -i $INTERFACE 2>/dev/null || echo "ethtool $INTERFACE not found"

echo -e "\n===== IRQ Affinity ====="
for irq in $(grep $INTERFACE /proc/interrupts | awk '{print $1}' | sed 's/://'); do
  echo -n "IRQ $irq: "; cat /proc/irq/$irq/smp_affinity_list 2>/dev/null
done

echo -e "\n===== Current ss TCP stats ====="
ss -itmn | grep -A1 ESTAB

echo -e "\n===== MTU Info ====="
ip link | grep mtu

echo -e "\n===== NUMA Topology (if available) ====="
lscpu | grep -E 'NUMA node|CPU\(s\)'

echo -e "\n===== CPU Scaling Governor ====="
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "Not available"

echo -e "\n===== CPU Frequency ====="
lscpu | grep MHz

} > "$OUT"

echo "✅ Output saved to $OUT"