# -*- coding: utf-8 -*-
from optparse import OptionParser
import os

def process_file(file_path, cc, traffic):
    if not os.path.exists(file_path):
        return 0, 0, 0, 0, 0, 0  # 返回默认值    

    all_fct = []
    large_flow_fct = []
    small_flow_fct = []
    packets = []
    total_packet_size = 0
    all_start_time = []
    all_stop_time = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            data = line.strip().split(' ')
            packet_size = int(data[4])
            start_time = float(data[5])
            fct_value = float(data[6])

            total_packet_size += packet_size
            
            all_start_time.append(start_time)
            all_stop_time.append(start_time + fct_value)
            all_fct.append(fct_value)
            packets.append(packet_size)

            if packet_size >= 10 * 1024 * 1024:  # 10M in bytes
                large_flow_fct.append(fct_value)
            elif packet_size <= 100 * 1024:  # 100K in bytes
                small_flow_fct.append(fct_value)

    average_fct = sum(all_fct) / len(all_fct)
    large_flow_average_fct = sum(large_flow_fct) / len(large_flow_fct) if large_flow_fct else 0
    small_flow_average_fct = sum(small_flow_fct) / len(small_flow_fct) if small_flow_fct else 0

    all_fct.sort()
    index_99 = int(len(all_fct) * 0.99)
    index_95 = int(len(all_fct) * 0.95)

    fct_99 = all_fct[index_99]
    fct_95 = all_fct[index_95]

    all_start_time.sort()
    all_stop_time.sort()

    traffic_index = -1
    if traffic == "FbHdp":
        if cc == "newcc":
            traffic_index = int(len(all_fct) * 0.99)
        if cc == "hp":
            traffic_index = int(len(all_fct) * 0.999)
    if traffic == "WebSearch":
        if cc == "newcc":
            traffic_index = int(len(all_fct) * 0.96)
        if cc == "hp":
            traffic_index = int(len(all_fct) * 0.99)            
    total_time = all_stop_time[traffic_index] - all_start_time[0]
    average_goodput = total_packet_size * 8 / total_time / 256

    average_delay = sum(all_fct) / (total_packet_size / 1000)

    return average_fct, large_flow_average_fct, small_flow_average_fct, fct_99, fct_95, average_goodput, average_delay

if __name__=="__main__":
    parser = OptionParser()

    parser.add_option("-m", "--traffic_mode", dest="traffic_mode",
                      help="traffic_mode parameter value", default="WebSearch")

    options, args = parser.parse_args()

    traffic_loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    congestion_controls = ["dcqcn", "timely", "hp", "newcc"]

    # 初始化结果存储
    results = {cc: [] for cc in congestion_controls}

    # 处理每种拥塞控制算法的文件
    for cc in congestion_controls:
        for load in traffic_loads:
            file_path = f"../simulation/mix/fct_spine_leaf_{options.traffic_mode}_{load}_0.1_{cc}.txt"
            results[cc].append(process_file(file_path, cc, options.traffic_mode))

    # 输出平均FCT
    print("平均FCT (ms):")
    for cc in congestion_controls:
        print(f"{cc} ", end=" ")
        for res in results[cc]:
            print("{:.2f}".format(float(res[0]) / 1e6), end=" ")
        print()

    # 输出大流平均FCT
    print("\n大流平均FCT (ms):")
    for cc in congestion_controls:
        print(f"{cc} ", end=" ")
        for res in results[cc]:
            print("{:.2f}".format(float(res[1]) / 1e6), end=" ")
        print()

    # 输出小流平均FCT
    print("\n小流平均FCT (us):")
    for cc in congestion_controls:
        print(f"{cc} ", end=" ")
        for res in results[cc]:
            print("{:.2f}".format(float(res[2]) / 1e3), end=" ")
        print()

    # 输出第99%大的FCT
    print("\n第99%大的FCT (ms):")
    for cc in congestion_controls:
        print(f"{cc} ", end=" ")
        for res in results[cc]:
            print("{:.2f}".format(float(res[3]) / 1e6), end=" ")
        print()

    # 输出第95%大的FCT
    print("\n第95%大的FCT (ms):")
    for cc in congestion_controls:
        print(f"{cc} ", end=" ")
        for res in results[cc]:
            print("{:.2f}".format(float(res[4]) / 1e6), end=" ")
        print()

    # 输出平均吞吐量
    print("\n平均吞吐量 (Gbps):")
    for cc in congestion_controls:
        print(f"{cc} ", end=" ")
        for res in results[cc]:
            print("{:.2f}".format(float(res[5])), end=" ")
        print()

    # 输出平均吞吐量
    print("\n平均时延 (Gbps):")
    for cc in congestion_controls:
        print(f"{cc} ", end=" ")
        for res in results[cc]:
            print("{:.2f}".format(float(res[6])), end=" ")
        print()        