#!/bin/bash

# 检查是否提供了时间间隔参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <时间间隔(秒)>"
    exit 1
fi

INTERVAL=$1
file="/proc/net/netstat"
declare -A prev_values

# 捕获CTRL+C信号
trap "echo -e '\n监控已停止'; exit 0" SIGINT

# 函数：采集并处理数据
collect_data() {
    # 临时存储当前值
    declare -A current_values

    # 读取/proc/net/netstat
    while read -r header && read -r values; do
        # 以空格切分
        read -a header_arr <<< "$header"
        read -a values_arr <<< "$values"

        # 提取前缀(如TcpExt:)
        prefix="${header_arr[0]}"

        # 从第二个元素开始处理
        for ((i=1; i<${#header_arr[@]}; i++)); do
            key="${prefix}${header_arr[$i]}"
            value="${values_arr[$i]}"
            current_values["$key"]=$value
        done
    done < "$file"

    # 输出标题行
    printf "%-40s %20s %20s\n" "METRIC" "CURRENT" "DELTA"
    printf "%-40s %20s %20s\n" "----------------------------------------" "--------------------" "--------------------"

    # 输出非零值及差值
    for key in "${!current_values[@]}"; do
        value="${current_values[$key]}"

        # 只处理非零值
        if [ "$value" -ne 0 ]; then
            if [[ -n "${prev_values[$key]}" ]]; then
                # 计算差值
                diff=$((value - prev_values[$key]))
                # 只输出有变化或首次出现的指标
                if [ $diff -ne 0 ] || [ $first_run -eq 1 ]; then
                    printf "%-40s %20d %20d\n" "$key" "$value" "$diff"
                fi
            else
                # 首次出现的指标
                printf "%-40s %20d %20s\n" "$key" "$value" "-"
            fi
        fi
    done

    # 保存当前值为下次比较的前值
    for key in "${!current_values[@]}"; do
        prev_values["$key"]=${current_values[$key]}
    done
}

echo "开始监控 /proc/net/netstat，间隔 $INTERVAL 秒 (按Ctrl+C终止)"
echo "---------------------------------------------------"

# 标记第一次运行
first_run=1

while true; do
    echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')]"
    collect_data

    # 第一次运行后设置为0
    first_run=0

    sleep $INTERVAL
done