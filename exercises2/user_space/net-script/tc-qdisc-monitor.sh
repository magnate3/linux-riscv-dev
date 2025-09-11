#!/bin/bash

# 检查参数
if [ $# -ne 2 ]; then
    echo "用法: $0 <网络设备名> <时间间隔(秒)>"
    exit 1
fi

IFNAME=$1
INTERVAL=$2

# 验证设备是否存在
if ! ip link show "$IFNAME" &>/dev/null; then
    echo "错误: 网络设备 '$IFNAME' 不存在"
    exit 1
fi

# 捕获CTRL+C信号
trap "echo -e '\n监控已停止'; exit 0" SIGINT

# 存储上一次的值
declare -A prev_values

# 打印标题行
print_header() {
    printf "%-15s %-15s %-15s\n" "指标名" "当前值" "差值"
    printf "%-15s %-15s %-15s\n" "---------------" "---------------" "---------------"
}

# 解析tc输出并提取指标
collect_data() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local tc_output=$(tc -s qdisc show dev "$IFNAME")

    echo -e "\n时间戳: $timestamp"
    print_header

    # 提取Sent行和backlog行
    local sent_line=$(echo "$tc_output" | grep -i "sent")
    local backlog_line=$(echo "$tc_output" | grep -i "backlog")

    # 从Sent行提取信息，避免使用包含逗号和括号的部分
    local sent_bytes=$(echo "$sent_line" | awk '{print $2}')
    local sent_pkts=$(echo "$sent_line" | awk '{print $4}')

    # 使用sed提取括号内的值，并去掉标点符号
    local dropped=$(echo "$sent_line" | sed 's/.*dropped \([0-9]*\).*/\1/')
    local overlimits=$(echo "$sent_line" | sed 's/.*overlimits \([0-9]*\).*/\1/')
    local requeues=$(echo "$sent_line" | sed 's/.*requeues \([0-9]*\).*/\1/')

    # 提取backlog信息
    local backlog_bytes=$(echo "$backlog_line" | awk '{print $2}' | sed 's/b$//')
    local backlog_pkts=$(echo "$backlog_line" | awk '{print $3}' | sed 's/p$//')

    # 打印当前值和差值
    print_metric "sent_bytes" "$sent_bytes"
    print_metric "sent_pkts" "$sent_pkts"
    print_metric "dropped" "$dropped"
    print_metric "overlimits" "$overlimits"
    print_metric "requeues" "$requeues"
    print_metric "backlog_bytes" "$backlog_bytes"
    print_metric "backlog_pkts" "$backlog_pkts"
}

# 打印指标
print_metric() {
    local metric_name=$1
    local current_value=$2

    # 去掉可能存在的单位后缀(b, p等)和标点符号
    current_value=$(echo "$current_value" | sed 's/[^0-9]//g')

    # 空值处理
    if [[ -z "$current_value" ]]; then
        current_value=0
    fi

    # 如果是第一次运行或值不存在
    if [[ -z "${prev_values[$metric_name]}" ]]; then
        printf "%-15s %-15s %-15s\n" "$metric_name" "$current_value" "-"
    else
        # 计算差值
        local prev_value=${prev_values[$metric_name]}
        local delta=$((current_value - prev_value))
        printf "%-15s %-15s %-15s\n" "$metric_name" "$current_value" "$delta"
    fi

    # 保存当前值
    prev_values[$metric_name]=$current_value
}

echo "开始监控设备 $IFNAME 的TC队列统计信息，间隔 $INTERVAL 秒 (按Ctrl+C终止)"

# 循环收集数据
while true; do
    collect_data
    sleep "$INTERVAL"
done