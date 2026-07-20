#!/bin/bash

echo "====== [ mlx5 IRQ Affinity Report ] ======"
date
echo

mlx5_irqs=$(grep -i 'mlx5.*TxRx' /proc/interrupts)

if [ -z "$mlx5_irqs" ]; then
    echo "❌ No mlx5 TxRx IRQs found!"
    exit 1
fi

cpu_count=$(grep -c ^processor /proc/cpuinfo)
irq_lines=$(grep -n 'mlx5.*TxRx' /proc/interrupts)

echo "Total CPUs: $cpu_count"
echo "Detected IRQ lines for mlx5 TxRx queues: $(echo "$irq_lines" | wc -l)"
echo

# 解析每一行 IRQ
while read -r line; do
    irq_num=$(echo "$line" | awk -F: '{print $2}' | awk '{print $1}')
    irq_name=$(echo "$line" | cut -d':' -f2- | awk '{$1=""; print $0}' | sed 's/^ *//')

    # 中断计数值
    counts=$(echo "$line" | cut -d':' -f3- | awk '{$1=""; print $0}')
    counts_arr=($counts)

    # 格式化显示中断分布：只展示非零
    per_cpu_summary=""
    for ((i=0; i<${#counts_arr[@]}; i++)); do
        val=${counts_arr[$i]}
        if [ "$val" -ne 0 ]; then
            per_cpu_summary+="$val<$i> "
        fi
    done

    # 读取绑定的 CPU 掩码
    irq_file="/proc/irq/$irq_num/smp_affinity"
    if [ -f "$irq_file" ]; then
        cpu_mask=$(cat "$irq_file")
        cpu_mask_bin=$(printf "%032d" "$(echo "obase=2; ibase=16; $cpu_mask" | bc)")
        affinity=""
        for ((i=0; i<${#cpu_mask_bin}; i++)); do
            if [ "${cpu_mask_bin: -1 - $i:1}" == "1" ]; then
                affinity+="CPU$i "
            fi
        done
    else
        affinity="unknown"
    fi

    echo "IRQ $irq_num [$irq_name]"
    echo "  → Bound to: $affinity"
    echo "  → IRQ counts per CPU : $per_cpu_summary"
    echo
done <<< "$mlx5_irqs"
