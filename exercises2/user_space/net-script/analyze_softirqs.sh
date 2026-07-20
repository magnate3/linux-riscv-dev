#!/bin/bash
# filepath: e:\develope\x-monitor\tools\bin\net\analyze_softirqs.sh

# 检查参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <interval_seconds>"
    echo "Example: $0 1"
    exit 1
fi

# 获取时间间隔参数
interval=$1

# 捕获 Ctrl+C 信号
trap 'echo -e "\nExiting..."; exit 0' INT

# 获取 CPU 核数
cpu_count=$(grep -c ^processor /proc/cpuinfo)

# 初始化保存上次值的数组
declare -A prev_rx_values
declare -A prev_tx_values

echo "=== SoftIRQ Analysis (NET_RX and NET_TX) ==="
echo "Sampling every $interval seconds. Press Ctrl+C to exit."
echo ""

while true; do
    echo "Timestamp: $(date)"
    
    # 收集软中断值
    declare -A irq_matrix  # key: irqname_cpu, value: count
    
    # 只收集 NET_RX 和 NET_TX
    for irq_name in "NET_RX" "NET_TX"; do
        line=$(grep -w "$irq_name" /proc/softirqs)
        if [ -z "$line" ]; then
            echo "⚠️  SoftIRQ $irq_name not found!"
            continue
        fi
        counts=$(echo "$line" | cut -d: -f2)

        i=0
        for count in $counts; do
            irq_matrix["$irq_name-$i"]=$count
            ((i++))
        done
    done

    # 打印表头
    printf "%-8s %12s %12s %12s %12s\n" "CPU" "NET_RX" "RX_DELTA" "NET_TX" "TX_DELTA"
    echo "------------------------------------------------------------------------"

    # 输出每个 CPU 的中断值及差值
    for ((cpu=0; cpu<cpu_count; cpu++)); do
        printf "%-8s" "CPU$cpu"
        
        # NET_RX 及差值
        rx_val=${irq_matrix["NET_RX-$cpu"]:-0}
        if [[ -n "${prev_rx_values[$cpu]}" ]]; then
            rx_delta=$((rx_val - prev_rx_values[$cpu]))
            if [ $rx_delta -gt 0 ]; then
                rx_delta_str="+$rx_delta"
            else
                rx_delta_str="$rx_delta"
            fi
        else
            rx_delta_str="N/A"
        fi
        prev_rx_values[$cpu]=$rx_val
        printf "%12s %12s" "$rx_val" "$rx_delta_str"
        
        # NET_TX 及差值
        tx_val=${irq_matrix["NET_TX-$cpu"]:-0}
        if [[ -n "${prev_tx_values[$cpu]}" ]]; then
            tx_delta=$((tx_val - prev_tx_values[$cpu]))
            if [ $tx_delta -gt 0 ]; then
                tx_delta_str="+$tx_delta"
            else
                tx_delta_str="$tx_delta"
            fi
        else
            tx_delta_str="N/A"
        fi
        prev_tx_values[$cpu]=$tx_val
        printf "%12s %12s\n" "$tx_val" "$tx_delta_str"
    done
    
    # 保持清晰的区隔
    echo ""
    
    # 等待指定时间
    sleep $interval
done