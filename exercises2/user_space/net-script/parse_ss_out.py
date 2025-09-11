#!/usr/bin/env python3
import subprocess
import sys
import re
import time
import argparse
import signal
import datetime

# 处理 Ctrl+C 信号
running = True
def signal_handler(sig, frame):
    global running
    print("\nReceived Ctrl+C, stopping data collection...")
    running = False
signal.signal(signal.SIGINT, signal_handler)

def parse_ss_output(sport, output_file=None):
    # 获取命令输出
    cmd = ["ss", "-tinm", "state", "established", f"sport = :{sport}"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)

    # 获取时间戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 准备输出内容
    output_buffer = []
    output_buffer.append(f"==== Collection Time: {timestamp} ====")

    # 分行处理
    lines = result.stdout.strip().splitlines()
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i].strip()
        if not line or line.startswith("Recv-Q"):
            i += 1
            continue

        conn_match = re.search(r'(\d+\.\d+\.\d+\.\d+:\d+)\s+(\d+\.\d+\.\d+\.\d+:\d+)', line)
        if conn_match:
            local = conn_match.group(1)
            peer = conn_match.group(2)
            output_buffer.append("=" * 40)
            output_buffer.append(f"🟢 Local Address: {local}")
            output_buffer.append(f"🔵 Peer Address: {peer}")

            # 查看下一行是否包含 skmem 和 TCP 状态信息
            i += 1
            if i < total:
                sk_line = lines[i].strip()

                # --- 解析 skmem ---
                skmem_match = re.search(r'skmem:\((.*?)\)', sk_line)
                if skmem_match:
                    sk_fields = skmem_match.group(1).split(',')
                    skmem_map = {
                        'r': 'Receive Queue Memory',
                        'rb': 'Receive Buffer Size',
                        't': 'Send Queue Memory',
                        'tb': 'Send Buffer Size',
                        'f': 'Forward Allocation Memory',
                        'w': 'Write Queue Pending Data',
                        'o': 'Options Memory',
                        'bl': 'Backlog Queue Memory',
                        'd': 'Drop Count'
                    }
                    output_buffer.append("📦 Socket Memory Info (bytes):")
                    for field in sk_fields:
                        key_match = re.match(r'[a-z]+', field)
                        val_match = re.search(r'\d+', field)
                        if key_match and val_match:
                            key = key_match.group()
                            value = val_match.group()
                            desc = skmem_map.get(key, key)
                            output_buffer.append(f"  {desc}: {value}")

                # --- 解析 TCP 状态字段 ---
                output_buffer.append("📊 TCP Status Information:")
                details = sk_line  # 当前行可能含 TCP 信息

                # 修复：直接查找特定的速率参数
                pacing_rate_match = re.search(r'pacing_rate\s+(\d+)bps', details)
                delivery_rate_match = re.search(r'delivery_rate\s+(\d+)bps', details)

                if pacing_rate_match:
                    pacing_rate = int(pacing_rate_match.group(1))
                    output_buffer.append(f"  Pacing Rate (bps): {pacing_rate:,}")

                if delivery_rate_match:
                    delivery_rate = int(delivery_rate_match.group(1))
                    output_buffer.append(f"  Delivery Rate (bps): {delivery_rate:,}")

                # 常规 TCP 参数解析
                tcp_keys = ["rtt:", "bytes_sent:", "cwnd:", "send", "minrtt"]
                fields = {
                    "rtt": "Round Trip Time (ms)",
                    "ato": "ACK Timeout (ms)",
                    "mss": "Max Segment Size (bytes)",
                    "pmtu": "Path MTU (bytes)",
                    "rcvmss": "Receive MSS",
                    "advmss": "Advertised MSS",
                    "cwnd": "Congestion Window",
                    "ssthresh": "Slow Start Threshold",
                    "bytes_sent": "Bytes Sent",
                    "bytes_acked": "Bytes Acknowledged",
                    "bytes_received": "Bytes Received",
                    "segs_out": "Segments Out",
                    "segs_in": "Segments In",
                    "data_segs_out": "Data Segments Out",
                    "data_segs_in": "Data Segments In",
                    "send": "Send Rate (bps)",
                    "lastsnd": "Last Send Time (ms)",
                    "lastrcv": "Last Receive Time (ms)",
                    "lastack": "Last ACK Time (ms)",
                    "delivered": "Delivered Segments",
                    "app_limited": "Application Limited",
                    "busy": "Busy Time (ms)",
                    "unacked": "Unacknowledged Segments",
                    "rcv_space": "Receive Window Space",
                    "rcv_ssthresh": "Receive Slow Start Threshold",
                    "minrtt": "Minimum RTT (ms)"
                }

                if any(k in details for k in tcp_keys):
                    for key, desc in fields.items():
                        # 跳过已经单独处理的 pacing_rate 和 delivery_rate
                        if key in ["pacing_rate", "delivery_rate"]:
                            continue

                        match = re.search(rf"{key}:(\S+)", details)
                        if match:
                            value = match.group(1)
                            value_clean = re.sub(r'[a-zA-Z]+$', '', value)
                            try:
                                if key == "rtt":
                                    main_rtt = value.split('/')[0]
                                    output_buffer.append(f"  {desc}: {main_rtt} ms")
                                elif key == "send":
                                    output_buffer.append(f"  {desc}: {int(float(value_clean)):,} bps")
                                elif key in ["busy", "lastsnd", "lastrcv", "lastack", "minrtt"]:
                                    output_buffer.append(f"  {desc}: {float(value_clean)} ms")
                                elif key in ["cwnd", "rcv_space", "rcv_ssthresh", "delivered", "unacked", "ssthresh"]:
                                    output_buffer.append(f"  {desc}: {int(value_clean)}")
                                elif key == "app_limited":
                                    output_buffer.append(f"  {desc}: Yes" if value else "No")
                                else:
                                    output_buffer.append(f"  {desc}: {value}")
                            except ValueError:
                                output_buffer.append(f"  {desc}: Parse failed (raw: {value})")

                    # 检查是否存在 ssthresh，如果不存在则显示为 0
                    if not re.search(r"ssthresh:", details):
                        output_buffer.append(f"  Slow Start Threshold: 0")
                else:
                    output_buffer.append("📊 TCP Status Information: ⚠️ Not Found")
        else:
            i += 1

        i += 1

    # 输出到控制台
    for line in output_buffer:
        print(line)

    # 如果指定了输出文件，则写入文件
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            for line in output_buffer:
                f.write(line + '\n')
            f.write('\n')  # 每次采集后增加一个空行，便于区分


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Parse and display TCP socket information')
    parser.add_argument('sport', type=str, help='Source port to monitor')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='Collection interval in seconds (default: 1.0)')
    parser.add_argument('-o', '--output', type=str, help='Output file path')
    args = parser.parse_args()

    print(f"Monitoring TCP connections on port {args.sport}")
    print(f"Collection interval: {args.interval} seconds")
    print(f"Output file: {args.output if args.output else 'None (console only)'}")
    print("Press Ctrl+C to stop monitoring")
    print()

    # 循环采集数据，直到 Ctrl+C 中断
    try:
        while running:
            parse_ss_output(args.sport, args.output)
            time.sleep(args.interval)
    except Exception as e:
        print(f"Error occurred: {e}")

    print("Data collection stopped.")


if __name__ == "__main__":
    main()