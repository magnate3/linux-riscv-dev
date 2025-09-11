#!/bin/bash

# 参数1: 设备名列表 (空格分隔)，比如 "eth0 eth1"
# 参数2: 时间间隔（秒）
dev_list=($1)
interval=$2

# 检查参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 \"dev1 dev2 ...\" interval_seconds"
    exit 1
fi

# 提取字段名
read_fields() {
    local type=$1
    local header=$(grep '|' /proc/net/dev | tail -n1)
    local fields=()

    if [ "$type" = "Receive" ]; then
        fields=($(echo "$header" | awk -F'|' '{print $2}' | sed 's/^ *//g' | tr -s ' ' ' '))
    else
        fields=($(echo "$header" | awk -F'|' '{print $3}' | sed 's/^ *//g' | tr -s ' ' ' '))
    fi

    echo "${fields[@]}"
}

# 读取设备数据
read_dev_data() {
    local dev=$1
    local line=$(grep -w "$dev" /proc/net/dev | tr -s ' ' ' ')
    echo "$line" | cut -d':' -f2 | sed 's/^ *//'
}

# 采集一组数据
declare -A old_data

collect_data() {
    for dev in "${dev_list[@]}"; do
        values=($(read_dev_data "$dev"))
        for i in "${!values[@]}"; do
            old_data["$dev,$i"]=${values[$i]}
        done
    done
}

# 获取字段名
recv_fields=($(read_fields "Receive"))
trans_fields=($(read_fields "Transmit"))

# 首次采集
collect_data

# 捕获Ctrl+C中断优雅退出
trap "echo; echo 'Interrupted by user'; exit 0" SIGINT

# 无限循环
while true; do
    sleep "$interval"

    echo "==================== $(date '+%F %T') ===================="

    for dev in "${dev_list[@]}"; do
        echo "Device: $dev"

        values=($(read_dev_data "$dev"))

        echo "--- Receive ---"
        printf "%-20s %-15s %-15s\n" "Field" "Current" "Delta"
        for i in "${!recv_fields[@]}"; do
            idx=$i
            curr=${values[$idx]}
            prev=${old_data["$dev,$idx"]}
            diff=$((curr - prev))
            printf "%-20s %-15s %-15s\n" "${recv_fields[$i]}" "$curr" "$diff"
        done

        echo "--- Transmit ---"
        printf "%-20s %-15s %-15s\n" "Field" "Current" "Delta"
        for i in "${!trans_fields[@]}"; do
            idx=$((i + ${#recv_fields[@]}))
            curr=${values[$idx]}
            prev=${old_data["$dev,$idx"]}
            diff=$((curr - prev))
            printf "%-20s %-15s %-15s\n" "${trans_fields[$i]}" "$curr" "$diff"
        done
        echo
    done

    # 更新采样
    collect_data
done
