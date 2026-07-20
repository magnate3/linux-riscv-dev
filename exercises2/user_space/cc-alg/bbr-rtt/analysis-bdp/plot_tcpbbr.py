#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
# from scipy import interpolate  # 移除scipy依赖，避免版本兼容性问题

def load_tcp_bbr_data(log_file):
    """加载TCP BBR统计数据"""
    print(f"Loading TCP BBR data from: {log_file}")
    
    try:
        # 读取日志文件
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # 提取有效的数据行 (包含 RAW DATA:)
        data_lines = [line.strip() for line in lines if 'RAW DATA:' in line]
        print(f"Loaded {len(data_lines)} data points")
        
        if not data_lines:
            print("No valid data lines found in the log file.")
            return None
        
        # 提取数据并创建DataFrame
        data = []
        for line in data_lines:
            try:
                # 解析时间戳
                timestamp_str = line.split('[')[1].split(']')[0]
                timestamp_ms = int(timestamp_str)
                
                # 解析IP地址和端口
                conn_part = line.split(']')[1].split('RAW DATA:')[0].strip()
                src_ip = conn_part.split(':')[0].strip()
                src_port = int(conn_part.split(':')[1].split('->')[0].strip())
                dst_part = conn_part.split('->')[1].strip()
                dst_ip = dst_part.split(':')[0].strip()
                dst_port = int(dst_part.split(':')[1].strip())
                
                # 解析RAW DATA部分
                raw_data = line.split('RAW DATA:')[1].strip()
                
                # 解析各种指标
                entry = {
                    'timestamp_ms': timestamp_ms,
                    'timestamp': pd.Timestamp.fromtimestamp(timestamp_ms / 1000000),  # 微秒转为时间戳
                    'src_ip': src_ip,
                    'src_port': src_port,
                    'dst_ip': dst_ip,
                    'dst_port': dst_port
                }
                
                # 提取RTT
                if 'rtt:' in raw_data:
                    rtt_parts = raw_data.split('rtt:')[1].split(' ')[0].split('/')
                    entry['rtt_ms'] = float(rtt_parts[0])
                    entry['rtt_var_ms'] = float(rtt_parts[1]) if len(rtt_parts) > 1 else 0.0
                
                # 提取CWND
                if 'cwnd:' in raw_data:
                    entry['cwnd'] = int(raw_data.split('cwnd:')[1].split(' ')[0])
                
                # 提取BBR特定指标
                if 'bbr:(' in raw_data:
                    bbr_part = raw_data.split('bbr:(')[1].split(')')[0]
                    
                    # 标记为BBR
                    entry['bbr_state'] = 'BBR'
                    
                    # 带宽 (bw:Xbps)
                    if 'bw:' in bbr_part:
                        bw_str = bbr_part.split('bw:')[1].split(',')[0]
                        if 'bps' in bw_str:
                            bw_val = float(bw_str.replace('bps', ''))
                            entry['bbr_bandwidth_bps'] = bw_val
                            entry['bbr_bandwidth_mbps'] = bw_val / 1_000_000
                    
                    # 最小RTT (mrtt:X)
                    if 'mrtt:' in bbr_part:
                        mrtt_str = bbr_part.split('mrtt:')[1].split(',')[0]
                        entry['bbr_min_rtt_ms'] = float(mrtt_str)
                    
                    # BBR拥塞增益 (cwnd_gain:X)
                    if 'cwnd_gain:' in bbr_part:
                        entry['bbr_cwnd_gain'] = float(bbr_part.split('cwnd_gain:')[1].split(',')[0])
                    
                    # BBR步调增益 (pacing_gain:X)
                    if 'pacing_gain:' in bbr_part:
                        entry['bbr_pacing_gain'] = float(bbr_part.split('pacing_gain:')[1].split(',')[0])
                else:
                    entry['bbr_state'] = 'NON-BBR'
                
                # 提取pacing_rate
                if 'pacing_rate' in raw_data:
                    pr_parts = raw_data.split('pacing_rate')[1].strip().split(' ')[0]
                    entry['pacing_rate_mbps'] = float(pr_parts.replace('bps', '')) / 1_000_000
                
                # 提取delivery_rate
                if 'delivery_rate' in raw_data:
                    dr_parts = raw_data.split('delivery_rate')[1].strip().split(' ')[0]
                    entry['delivery_rate_mbps'] = float(dr_parts.replace('bps', '')) / 1_000_000
                
                # 提取send速率
                if 'send ' in raw_data:
                    send_parts = raw_data.split('send ')[1].split('bps')[0]
                    entry['send_rate_mbps'] = float(send_parts) / 1_000_000
                
                # 提取bytes信息
                if 'bytes_sent:' in raw_data:
                    entry['bytes_sent_mb'] = float(raw_data.split('bytes_sent:')[1].split(' ')[0]) / (1024*1024)
                
                if 'bytes_acked:' in raw_data:
                    entry['bytes_acked_mb'] = float(raw_data.split('bytes_acked:')[1].split(' ')[0]) / (1024*1024)
                
                if 'bytes_received:' in raw_data:
                    entry['bytes_received_mb'] = float(raw_data.split('bytes_received:')[1].split(' ')[0]) / (1024*1024)
                
                if 'bytes_retrans:' in raw_data:
                    entry['bytes_retrans_mb'] = float(raw_data.split('bytes_retrans:')[1].split(' ')[0]) / (1024*1024)
                
                # 提取segments信息
                if 'unacked:' in raw_data:
                    entry['unacked_segments'] = int(raw_data.split('unacked:')[1].split(' ')[0])
                
                if 'segs_out:' in raw_data:
                    entry['segs_out'] = int(raw_data.split('segs_out:')[1].split(' ')[0])
                
                if 'segs_in:' in raw_data:
                    entry['segs_in'] = int(raw_data.split('segs_in:')[1].split(' ')[0])
        
                # 提取retrans信息 - 改进识别方法
                # 形式1: retrans:0/5
                if 'retrans:' in raw_data:
                    retrans_str = raw_data.split('retrans:')[1].split(' ')[0]
                    if '/' in retrans_str:
                        retrans_parts = retrans_str.split('/')
                        entry['retrans_current'] = int(retrans_parts[0])
                        entry['retrans_total'] = int(retrans_parts[1])
                
                # 提取sacked信息 - 改进识别方法
                # 形式1: sacked:6
                if 'sacked:' in raw_data:
                    try:
                        sacked_str = raw_data.split('sacked:')[1].split(' ')[0]
                        entry['sacked_packets'] = int(sacked_str)
                    except (ValueError, IndexError):
                        pass
                
                # 形式2: dsack_dups:5 (重复SACK)
                if 'dsack_dups:' in raw_data:
                    try:
                        dsack_str = raw_data.split('dsack_dups:')[1].split(' ')[0]
                        if 'sacked_packets' not in entry:
                            entry['sacked_packets'] = int(dsack_str)
                        else:
                            entry['sacked_packets'] += int(dsack_str)
                    except (ValueError, IndexError):
                        pass
                
                # 提取丢包信息 - 改进识别方法
                # 有多种可能的格式:
                # 1. lost:244
                # 2. lost 244 (没有冒号)
                # 3. 在其他标签后面: sacked:832 dsack_dups:5 lost:244
                if 'lost:' in raw_data:
                    try:
                        # 查找lost:后面的数字
                        lost_part = raw_data.split('lost:')[1]
                        # 提取第一个空格前的数字
                        lost_str = lost_part.split()[0]
                        entry['lost_packets'] = int(lost_str)
                    except (ValueError, IndexError) as e:
                        # 打印详细调试信息
                        # print(f"Error parsing lost packets: {e}")
                        # print(f"Lost part: {raw_data.split('lost:')[1] if 'lost:' in raw_data else 'N/A'}")
                        pass
                elif ' lost ' in raw_data:
                    # 处理无冒号格式
                    try:
                        lost_part = raw_data.split(' lost ')[1]
                        if lost_part[0].isdigit():  # 确保下一个字符是数字
                            lost_str = lost_part.split()[0]
                            entry['lost_packets'] = int(lost_str)
                    except (ValueError, IndexError):
                        pass
                
                # 提取重排序信息 - 改进识别方法
                # 形式1: reordering:31
                if 'reordering:' in raw_data:
                    try:
                        reord_str = raw_data.split('reordering:')[1].split(' ')[0]
                        entry['reordering'] = int(reord_str)
                    except (ValueError, IndexError):
                        pass
                
                # 形式2: reord_seen:1
                if 'reord_seen:' in raw_data:
                    try:
                        reord_seen_str = raw_data.split('reord_seen:')[1].split(' ')[0]
                        if 'reordering' not in entry:
                            entry['reordering'] = int(reord_seen_str)
                    except (ValueError, IndexError):
                        pass
                
                data.append(entry)
            except Exception as e:
                print(f"Error parsing line: {e}")
                print(f"Problematic line: {line}")
                continue
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            print("No data could be parsed from the log file.")
            return None
        
        # 计算相对时间（秒）
        # 重要：timestamp_ms单位是纳秒，需要除以1,000,000,000转换为秒
        df['time_sec'] = df['timestamp_ms'] / 1000000000  # 纳秒 -> 秒
        df['time_sec_zeroed'] = df['time_sec'] - df['time_sec'].min()  # 从0开始
        
        # 计算丢包率和重传率
        if 'retrans_total' in df.columns and 'segs_out' in df.columns:
            max_retrans = df['retrans_total'].max() if not df['retrans_total'].isnull().all() else 0
            max_segs = df['segs_out'].max()
            if max_segs > 0:
                df['retrans_rate'] = (max_retrans / max_segs) * 100
            else:
                df['retrans_rate'] = 0
        
        if 'lost_packets' in df.columns and 'segs_out' in df.columns:
            total_lost = df['lost_packets'].sum() if not df['lost_packets'].isnull().all() else 0
            max_segs = df['segs_out'].max()
            if max_segs > 0:
                df['packet_loss_rate'] = (total_lost / max_segs) * 100
            else:
                df['packet_loss_rate'] = 0
                
        # 数据清理
        numeric_columns = ['rtt_ms', 'rtt_var_ms', 'send_rate_mbps', 'pacing_rate_mbps', 
                          'delivery_rate_mbps', 'bytes_retrans_mb', 'cwnd', 'unacked_segments',
                          'bytes_sent_mb', 'bytes_acked_mb', 'bbr_bandwidth_mbps', 'bbr_min_rtt_ms',
                          'bbr_pacing_gain', 'bbr_cwnd_gain', 'lost_packets', 'retrans_current', 
                          'retrans_total', 'sacked_packets', 'reordering', 
                          'packet_loss_rate', 'retrans_rate']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 打印数据质量信息
        print(f"Data time range: {df['time_sec'].min():.2f}s - {df['time_sec'].max():.2f}s")
        print(f"BBR states found: {df['bbr_state'].unique()}")
        
        # 打印空值统计
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print("Missing data detected:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"  {col}: {count} missing values ({count/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_tcp_bbr_analysis(df, output_file):
    """绘制TCP BBR分析图表"""
    if df is None or df.empty:
        print("No data to plot.")
        return
    
    # 提高绘图性能
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 0.8
    plt.rcParams['agg.path.chunksize'] = 10000
    
    # 创建子图 - 增加到4个图表
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), dpi=120)
    plt.subplots_adjust(hspace=0.4)
    
    # 统一的标题样式
    title_style = dict(fontsize=14, fontweight='bold', 
                     bbox=dict(facecolor='lightblue', alpha=0.8, edgecolor='navy', boxstyle='round,pad=0.5'))
    
    # 使用预先计算好的时间值（time_sec_zeroed）
    # 打印时间范围信息以便调试
    print(f"Time range: {df['time_sec_zeroed'].min():.3f}s to {df['time_sec_zeroed'].max():.3f}s (duration: {df['time_sec_zeroed'].max()-df['time_sec_zeroed'].min():.3f}s)")
    
    # 1. 关键传输速率对比图（移除send_rate）
    ax1 = axes[0]
    
    # 绘制关键速率（不包括send_rate）
    if 'pacing_rate_mbps' in df.columns:
        ax1.plot(df['time_sec_zeroed'], df['pacing_rate_mbps'], 'g-', linewidth=2.0, label='Pacing Rate (Mbps)', alpha=0.8)
    
    if 'delivery_rate_mbps' in df.columns:
        ax1.plot(df['time_sec_zeroed'], df['delivery_rate_mbps'], 'r-', linewidth=2.0, label='Delivery Rate (Mbps)', alpha=0.8)
    
    if 'bbr_bandwidth_mbps' in df.columns:
        ax1.plot(df['time_sec_zeroed'], df['bbr_bandwidth_mbps'], 'purple', linewidth=2.0, label='BBR Bandwidth (Mbps)', alpha=0.9)
    
    # 添加统计信息（不包括send_rate）
    stats_text = ""
    for col, name in [('pacing_rate_mbps', 'Pacing'), ('delivery_rate_mbps', 'Delivery'), ('bbr_bandwidth_mbps', 'BBR BW')]:
        if col in df.columns and not df[col].isnull().all():
            avg_val = df[col].mean()
            max_val = df[col].max()
            stats_text += f"{name}: Avg {avg_val:.1f}, Max {max_val:.1f} Mbps\n"
    
    if stats_text:
        ax1.text(0.02, 0.98, stats_text.strip(), transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    ax1.set_ylabel('Bandwidth (Mbps)')
    ax1.set_title('TCP BBR Key Transmission Rates', **title_style)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # 2. RTT分析图
    ax2 = axes[1]
    
    if 'rtt_ms' in df.columns:
        ax2.plot(df['time_sec_zeroed'], df['rtt_ms'], 'r-', linewidth=2.0, label='RTT (ms)', alpha=0.8)
    
    if 'rtt_var_ms' in df.columns:
        ax2.plot(df['time_sec_zeroed'], df['rtt_var_ms'], 'orange', linewidth=1.5, label='RTT Variation (ms)', alpha=0.7)
    
    if 'bbr_min_rtt_ms' in df.columns:
        ax2.plot(df['time_sec_zeroed'], df['bbr_min_rtt_ms'], 'blue', linewidth=2.0, label='BBR MinRTT (ms)', alpha=0.9)
    
    # RTT统计信息
    stats_text = ""
    for col, name in [('rtt_ms', 'RTT'), ('rtt_var_ms', 'RTT Var'), ('bbr_min_rtt_ms', 'BBR MinRTT')]:
        if col in df.columns and not df[col].isnull().all():
            avg_val = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            stats_text += f"{name}: Avg {avg_val:.2f}, Range {min_val:.2f}-{max_val:.2f} ms\n"
    
    if stats_text:
        ax2.text(0.02, 0.98, stats_text.strip(), transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_ylabel('RTT (ms)')
    ax2.set_title('TCP BBR RTT Analysis', **title_style)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # 3. 拥塞窗口和传输中字节数
    ax3 = axes[2]
    
    if 'cwnd' in df.columns:
        ax3.plot(df['time_sec_zeroed'], df['cwnd'], 'b-', linewidth=2.0, label='CWND (segments)', alpha=0.8)
    
    if 'unacked_segments' in df.columns:
        ax3.plot(df['time_sec_zeroed'], df['unacked_segments'], 'r-', linewidth=2.0, 
                label='Unacked Segments (segments)', alpha=0.8)
    
    # 计算利用率：在途数据 / CWND
    if 'cwnd' in df.columns and 'unacked_segments' in df.columns:
        # 过滤异常数据：cwnd > 10 且利用率 < 150%
        valid_mask = (df['cwnd'] > 10) & (df['unacked_segments'] / df['cwnd'] <= 1.5)
        if valid_mask.sum() > 0:
            utilization = (df.loc[valid_mask, 'unacked_segments'] / df.loc[valid_mask, 'cwnd'] * 100).mean()
            max_util = (df.loc[valid_mask, 'unacked_segments'] / df.loc[valid_mask, 'cwnd'] * 100).max()
            min_util = (df.loc[valid_mask, 'unacked_segments'] / df.loc[valid_mask, 'cwnd'] * 100).min()
            stats_text = f"Average CWND Utilization: {utilization:.1f}% (Range: {min_util:.1f}%-{max_util:.1f}%)\n"
            stats_text += f"(Filtered {len(df) - valid_mask.sum()} outliers)\n"
        else:
            stats_text = "Average CWND Utilization: N/A\n"
        
        # 添加其他统计
        avg_cwnd = df['cwnd'].mean()
        max_cwnd = df['cwnd'].max()
        avg_unacked = df['unacked_segments'].mean()
        max_unacked = df['unacked_segments'].max()
        
        stats_text += f"CWND: Avg {avg_cwnd:.1f}, Max {max_cwnd:.1f} segments\n"
        stats_text += f"Unacked: Avg {avg_unacked:.1f}, Max {max_unacked:.1f} segments"
        
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    ax3.set_ylabel('Segments Count')
    ax3.set_title('TCP Congestion Window and Unacked Segments', **title_style)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # 4. 丢包分析（只显示丢包数量，不显示丢包率）
    ax4 = axes[3]
    
    # 清空当前轴以确保只有我们想要的内容
    ax4.clear()
    
    # 只绘制丢包数量
    if 'lost_packets' in df.columns and not df['lost_packets'].isnull().all():
        ax4.plot(df['time_sec_zeroed'], df['lost_packets'], 'red', linewidth=2.0, 
                label='Lost Packets', alpha=0.8, marker='o', markersize=3)
        
        # 设置Y轴标签
        ax4.set_ylabel('Lost Packets Count', fontsize=12)
        
        # 添加图例
        ax4.legend(loc='upper left')
    
    # 只保留丢包统计信息
    stats_text = ""
    
    # 丢包事件统计
    if 'lost_packets' in df.columns:
        valid_lost = df['lost_packets'].dropna()
        if len(valid_lost) > 0:
            loss_events = (valid_lost > 0).sum()
            total_lost = valid_lost.sum()
            max_lost = valid_lost.max()
            avg_lost = valid_lost[valid_lost > 0].mean() if loss_events > 0 else 0
            stats_text += f"Loss Events: {loss_events}/{len(valid_lost)} ({loss_events/len(valid_lost)*100:.1f}% of samples)\n"
            stats_text += f"Total Lost Packets: {total_lost}, Max Single Loss: {max_lost}\n"
            if loss_events > 0:
                stats_text += f"Average Loss per Event: {avg_lost:.1f} packets"
    
    if stats_text:
        ax4.text(0.02, 0.98, stats_text.strip(), transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    ax4.set_title('TCP Packet Loss Analysis', **title_style)
    ax4.grid(True, alpha=0.3)
    
    # 设置所有子图的X轴格式 - 适合短时间序列的显示
    for i, ax in enumerate(axes):
        # 确保X轴从0开始，并设置合适的上限
        max_time = df['time_sec_zeroed'].max()
        ax.set_xlim(0, max_time)
        
        # 短时间序列用小间隔
        if max_time <= 10:  # 10秒以内
            major_interval = 1    # 每秒一个刻度
        elif max_time <= 30:
            major_interval = 5    # 每5秒一个刻度
        else:
            major_interval = 10   # 每10秒一个刻度
        
        # 设置网格线
        ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.3)
        
        # 增强时间轴标签
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        
        # 确保刻度标签清晰可见
        ax.tick_params(axis='x', which='major', labelsize=10, pad=5)
        
        # 避免生成过多刻度
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        
        # 使用适合短时间序列的简单格式
        from matplotlib.ticker import FuncFormatter
        
        # 使用简单的秒格式
        def format_time(x, pos):
            return f"{x:.1f}s"
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    
    plt.tight_layout(pad=2.0)  # 增加间距，避免标签重叠
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"TCP BBR analysis charts saved to {output_file}")

def analyze_sampling_rate(df):
    """Analyze data sampling frequency"""
    if len(df) < 2:
        return None
    
    # 计算时间间隔
    time_diffs = df['time_sec'].diff().dropna()
    
    stats = {
        'total_samples': len(df),
        'duration': df['time_sec'].max() - df['time_sec'].min(),
        'avg_interval': time_diffs.mean(),
        'min_interval': time_diffs.min(),
        'max_interval': time_diffs.max(),
        'std_interval': time_diffs.std(),
        'avg_frequency': 1.0 / time_diffs.mean() if time_diffs.mean() > 0 else 0
    }
    
    print(f"📊 Sampling Frequency Analysis:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Total duration: {stats['duration']:.2f}s")
    print(f"   Average interval: {stats['avg_interval']*1000:.1f}ms")
    print(f"   Interval range: {stats['min_interval']*1000:.1f}ms - {stats['max_interval']*1000:.1f}ms")
    print(f"   Average frequency: {stats['avg_frequency']:.2f} Hz ({stats['avg_frequency']:.1f} samples/sec)")
    
    return stats

def resample_data(df, target_frequency_hz=10):
    """Resample data for higher time resolution (using NumPy interpolation, avoiding SciPy dependency)"""
    if len(df) < 3:
        print("Too few data points for resampling")
        return df
    
    print(f"🔄 Resampling data to {target_frequency_hz} Hz...")
    
    # 创建新的时间序列
    time_start = df['time_sec'].min()
    time_end = df['time_sec'].max()
    new_time = np.arange(time_start, time_end, 1.0/target_frequency_hz)
    
    # 为新的DataFrame创建基础结构
    new_df = pd.DataFrame({'time_sec': new_time})
    
    # 对数值列进行插值
    numeric_columns = ['rtt_ms', 'rtt_var_ms', 'send_rate_mbps', 'pacing_rate_mbps', 
                      'delivery_rate_mbps', 'bytes_retrans_mb', 'cwnd_kb', 'unacked_segments_kb',
                      'bytes_sent_mb', 'bytes_acked_mb', 'bbr_bandwidth_mbps', 'bbr_min_rtt_ms',
                      'bbr_pacing_gain', 'bbr_cwnd_gain', 'lost_packets', 'retrans_current', 
                      'retrans_total', 'sacked_packets', 'fackets', 'reordering', 
                      'packet_loss_rate', 'retrans_rate']
    
    for col in numeric_columns:
        if col in df.columns and not df[col].isnull().all():
            # 移除NaN值
            valid_mask = ~df[col].isnull()
            if valid_mask.sum() >= 2:  # 至少需要2个有效点
                try:
                    # 使用NumPy的线性插值
                    x_valid = df.loc[valid_mask, 'time_sec'].values
                    y_valid = df.loc[valid_mask, col].values
                    new_df[col] = np.interp(new_time, x_valid, y_valid)
                except Exception as e:
                    print(f"   Warning: Cannot interpolate column {col}: {e}")
                    new_df[col] = np.nan
            else:
                new_df[col] = np.nan
    
    # 对于状态列，使用前向填充
    if 'bbr_state' in df.columns:
        # 为每个新时间点找到最近的状态
        new_df['bbr_state'] = 'UNKNOWN'
        for i, t in enumerate(new_time):
            # 找到小于等于当前时间的最后一个状态
            valid_states = df[df['time_sec'] <= t]
            if len(valid_states) > 0:
                new_df.loc[i, 'bbr_state'] = valid_states.iloc[-1]['bbr_state']
    
    # 重新计算时间戳
    start_time = pd.to_datetime(df['timestamp'].iloc[0])
    new_df['timestamp'] = start_time + pd.to_timedelta(new_df['time_sec'], unit='s')
    
    print(f"   Resampling completed: {len(df)} -> {len(new_df)} data points")
    print(f"   New sampling frequency: {len(new_df)/(time_end-time_start):.2f} Hz")
    
    return new_df

def main():
    parser = argparse.ArgumentParser(description="TCP BBR Statistical Data Plotter")
    parser.add_argument("log_file", help="Path to the TCP BBR log file (txt)")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--resample", type=int, help="Resample data to specified frequency (Hz), e.g., --resample 10")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze sampling rate without plotting")
    args = parser.parse_args()
    
    # 确定输出文件路径
    output_file = args.output
    if not output_file:
        # 避免重复前缀，直接使用描述性文件名
        if args.resample:
            output_file = f"bbr_analysis_resampled_{args.resample}hz.png"
        else:
            output_file = f"bbr_analysis.png"
    
    # 输出到当前目录 (不包含路径前缀)
    if os.path.dirname(output_file) == '':
        print(f"Output will be saved to current directory: {os.getcwd()}/{output_file}")
    
    # 加载数据
    df = load_tcp_bbr_data(args.log_file)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # 确保 cwnd_kb 和 unacked_segments_kb 存在，以保持代码兼容性
    if 'cwnd' in df.columns and 'cwnd_kb' not in df.columns:
        df['cwnd_kb'] = df['cwnd']  # 实际单位是segments，但保持变量名一致
    
    if 'unacked_segments' in df.columns and 'unacked_segments_kb' not in df.columns:
        df['unacked_segments_kb'] = df['unacked_segments']  # 实际单位是segments
    
    # 分析采样频率
    sampling_stats = analyze_sampling_rate(df)
    
    if args.analyze_only:
        return
    
    # 可选的重采样
    if args.resample and args.resample > 0:
        df = resample_data(df, args.resample)
        
        # 重新分析重采样后的频率
        print("\nPost-resampling frequency analysis:")
        analyze_sampling_rate(df)
    
    # 绘制图表
    plot_tcp_bbr_analysis(df, output_file)
    print(f"TCP BBR analysis charts saved to {output_file}")
    
    # 打印基本统计信息
    print(f"\n📊 Basic Statistics:")
    print(f"   Total data points: {len(df)}")
    print(f"   Time range: {df['time_sec'].min():.2f}s - {df['time_sec'].max():.2f}s")
    
    if 'bbr_state' in df.columns:
        print(f"   BBR states: {', '.join(df['bbr_state'].unique())}")
    
    if 'bytes_acked_mb' in df.columns:
        # 检查是否有足够的数据点进行计算
        if len(df) > 1 and not df['bytes_acked_mb'].isnull().all():
            last_valid_idx = df['bytes_acked_mb'].last_valid_index()
            first_valid_idx = df['bytes_acked_mb'].first_valid_index()
            if last_valid_idx is not None and first_valid_idx is not None:
                total_data = df.loc[last_valid_idx, 'bytes_acked_mb'] - df.loc[first_valid_idx, 'bytes_acked_mb']
                total_time = df.loc[last_valid_idx, 'time_sec'] - df.loc[first_valid_idx, 'time_sec']
        avg_throughput = (total_data / total_time * 8) if total_time > 0 else 0
        print(f"   Total data transferred: {total_data:.2f} MB")
        print(f"   Average throughput: {avg_throughput:.2f} Mbps")
    
    # Key rates analysis (without send_rate)
    print(f"\n🚀 Key Rate Analysis:")
    
    rates_analysis = {}
    for col, name in [('pacing_rate_mbps', 'Pacing Rate'), 
                     ('delivery_rate_mbps', 'Delivery Rate'),
                     ('bbr_bandwidth_mbps', 'BBR Bandwidth')]:
        if col in df.columns and not df[col].isnull().all():
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                rates_analysis[name] = {
                    'avg': valid_data.mean(),
                    'max': valid_data.max(),
                    'min': valid_data.min(),
                    'std': valid_data.std()
                }
    
    for name, stats in rates_analysis.items():
        print(f"   {name}: Avg {stats['avg']:.1f}, Max {stats['max']:.1f}, Min {stats['min']:.1f} (±{stats['std']:.1f}) Mbps")
    
    # CWND utilization analysis
    if 'cwnd' in df.columns and 'unacked_segments' in df.columns:
        # 过滤异常数据：cwnd > 10 且利用率 < 150%  
        valid_mask = (df['cwnd'] > 10) & (df['unacked_segments'] / df['cwnd'] <= 1.5)
        if valid_mask.sum() > 0:
            utilization = (df.loc[valid_mask, 'unacked_segments'] / df.loc[valid_mask, 'cwnd'] * 100)
            avg_util = utilization.mean()
            max_util = utilization.max()
            min_util = utilization.min()
            print(f"\n📊 CWND Utilization Analysis (filtered):")
            print(f"   Average: {avg_util:.1f}%, Range: {min_util:.1f}% - {max_util:.1f}%")
            print(f"   Valid samples: {valid_mask.sum()}/{len(df)} (filtered {len(df) - valid_mask.sum()} outliers)")
            print(f"   Note: CWND Utilization = Unacked segments / CWND size (segments)")

    # 丢包分析（移除重传分析，只保留丢包相关统计）
    print(f"\n📉 Packet Loss Analysis:")
    
    # 丢包统计
    if 'lost_packets' in df.columns:
        valid_lost = df['lost_packets'].dropna()
        if not valid_lost.empty:
            loss_events = (valid_lost > 0).sum()
            total_lost = valid_lost.sum()
            max_lost = valid_lost.max()
            avg_lost = valid_lost[valid_lost > 0].mean() if loss_events > 0 else 0
            print(f"   Loss Events: {loss_events}/{len(valid_lost)} ({loss_events/len(valid_lost)*100:.1f}% of valid samples)")
            print(f"   Total Lost Packets: {total_lost}, Max Single Loss: {max_lost}")
            if loss_events > 0:
                print(f"   Average Loss per Event: {avg_lost:.1f} packets")
    
    # 丢包率统计
    if 'packet_loss_rate' in df.columns and not df['packet_loss_rate'].isnull().all():
        valid_loss_rates = df[df['packet_loss_rate'] > 0]['packet_loss_rate']
        if len(valid_loss_rates) > 0:
            avg_loss_rate = valid_loss_rates.mean()
            max_loss_rate = df['packet_loss_rate'].max()
            print(f"   Packet Loss Rate: Avg {avg_loss_rate:.3f}%, Max {max_loss_rate:.3f}%")
        else:
            print(f"   Packet Loss Rate: 0.000% (no packet loss detected)")

if __name__ == "__main__":
    plt.switch_backend('agg')  # 使用非交互式后端
    main() 