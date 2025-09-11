#!/bin/bash
# filepath: e:\develope\x-monitor\tools\bin\net\check_bql.sh

# 检查参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <interface> <interval_seconds>"
    exit 1
fi

NIC=$1
INTERVAL=$2

# 初始化变量存储上次的 inflight 值
declare -A prev_inflight

# 捕获 Ctrl+C 信号以便优雅退出
trap 'echo -e "\nExiting..."; exit 0' INT

# 打印表头
echo "Monitoring BQL for network interface: $NIC (Ctrl+C to exit)"
echo -e "Queue\tLimit\tLimit-Min\tInflight\tDelta"
echo "-----------------------------------------------------------"

# 无限循环，直到被 Ctrl+C 中断
while true; do
    for queue in $(ls /sys/class/net/$NIC/queues/ | grep tx-); do
        # 获取当前值
        limit=$(cat /sys/class/net/$NIC/queues/$queue/byte_queue_limits/limit 2>/dev/null || echo "N/A")
        limit_min=$(cat /sys/class/net/$NIC/queues/$queue/byte_queue_limits/limit_min 2>/dev/null || echo "N/A")
        inflight=$(cat /sys/class/net/$NIC/queues/$queue/byte_queue_limits/inflight 2>/dev/null || echo "0")

        # 计算差值
        if [[ -z "${prev_inflight[$queue]}" ]]; then
            delta="N/A"  # 第一次运行时没有对比值
        else
            delta=$((inflight - prev_inflight[$queue]))
            # 添加正负号以便更清晰地表示增减
            if [ $delta -gt 0 ]; then
                delta="+$delta"
            fi
        fi

        # 存储当前值为下次对比用
        prev_inflight[$queue]=$inflight

        # 输出一行，格式化为表格
        printf "%-7s\t%-7s\t%-10s\t%-10s\t%-10s\n" "$queue" "$limit" "$limit_min" "$inflight" "$delta"
    done

    # 添加时间戳
    echo -e "\nTimestamp: $(date '+%Y-%m-%d %H:%M:%S') - Next update in ${INTERVAL}s"
    echo "-----------------------------------------------------------"

    # 等待指定的间隔时间
    sleep $INTERVAL

    # 不再清除屏幕，直接添加新的表头分隔
    echo -e "\nMonitoring BQL for network interface: $NIC (Ctrl+C to exit)"
    echo -e "Queue\tLimit\tLimit-Min\tInflight\tDelta"
    echo "-----------------------------------------------------------"
done