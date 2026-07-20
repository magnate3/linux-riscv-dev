#!/bin/bash

# 脚本用法: ./balance_irq.sh <网卡前缀> <NUMA节点列表>
# 例如: ./balance_irq.sh mlx5 0 1 2 3

if [ $# -lt 2 ]; then
    echo "Usage: $0 <nic_prefix> <numa_node1> <numa_node2> ..."
    exit 1
fi

NIC_PREFIX=$1
shift
NUMA_NODES=($@)

# 获取网卡的中断列表
IRQS=($(grep "${NIC_PREFIX}" /proc/interrupts | awk '{print $1}' | cut -d: -f1))

# 获取每个NUMA节点的CPU列表
declare -A NUMA_CPUS
declare -A NUMA_CPU_ARRAY
for node in "${NUMA_NODES[@]}"; do
    # 获取NUMA节点的CPU范围字符串
    cpu_ranges=$(lscpu -p=CPU,NODE | awk -F, -v node="$node" '$2 == node {print $1}' | sort -n | tr '\n' ',' | sed 's/,$//')
    NUMA_CPUS[$node]=$cpu_ranges

    # 将范围解析为单独的CPU列表
    cpu_list=()
    IFS=',' read -ra range_parts <<< "$cpu_ranges"
    for part in "${range_parts[@]}"; do
        if [[ $part == *-* ]]; then
            start=${part%-*}
            end=${part#*-}
            for ((cpu=start; cpu<=end; cpu++)); do
                cpu_list+=($cpu)
            done
        else
            cpu_list+=($part)
        fi
    done
    NUMA_CPU_ARRAY[$node]="${cpu_list[@]}"
done

# 打印每个NUMA节点的CPU信息(调试用)
for node in "${NUMA_NODES[@]}"; do
    echo "NUMA node$node CPUs: ${NUMA_CPUS[$node]}"
    echo "Expanded CPUs: ${NUMA_CPU_ARRAY[$node]}"
done

# 计算每个NUMA节点应该处理的IRQ数量
IRQS_PER_NODE=$(( ${#IRQS[@]} / ${#NUMA_NODES[@]} ))
EXTRA_IRQS=$(( ${#IRQS[@]} % ${#NUMA_NODES[@]} ))

# 为每个IRQ分配CPU
node_index=0
cpu_index=0
extra_assigned=0

for ((i=0; i<${#IRQS[@]}; i++)); do
    irq=${IRQS[$i]}

    # 选择当前NUMA节点
    current_node=${NUMA_NODES[$node_index]}
    IFS=' ' read -ra node_cpus <<< "${NUMA_CPU_ARRAY[$current_node]}"

    # 选择当前CPU
    cpu=${node_cpus[$cpu_index]}

    # 生成大端字节序的CPU掩码
    # 计算组和位(每组32个CPU)
    group=$((3 - (cpu / 32)))  # 大端序，组0在最高位
    bit=$((cpu % 32))

    # 初始化4个32位组
    mask=("00000000" "00000000" "00000000" "00000000")

    # 设置对应的位
    hex_value=$(printf "%08x" $((1 << bit)))
    mask[$group]=$hex_value

    # 格式化为逗号分隔的大端序掩码
    full_mask="${mask[0]},${mask[1]},${mask[2]},${mask[3]}"

    echo "Assigning IRQ $irq to CPU $cpu (Big Endian mask: $full_mask)"
    echo "$full_mask" > /proc/irq/$irq/smp_affinity

    # 更新索引
    cpu_index=$(( (cpu_index + 1) % ${#node_cpus[@]} ))
    if [ $cpu_index -eq 0 ]; then
        node_index=$(( (node_index + 1) % ${#NUMA_NODES[@]} ))
    fi
done

echo "IRQ balancing completed with Big Endian masks."