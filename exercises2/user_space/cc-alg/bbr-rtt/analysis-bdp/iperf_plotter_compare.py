import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os

if len(sys.argv) <= 2:
    print('Missing file names\nUsage: python3 iperf_plotter_compare.py file1.json file2.json')
    sys.exit(0)

input_file_tcp = sys.argv[1]
input_file_rdma = sys.argv[2]

output_file_name = f"throughput_comparison_TCP_RDMA"

def load_total_throughput(file_path):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
    time_intervals = json_data['intervals']
    end_time = [int(item['sum']['end']) for item in time_intervals]

    total_thr = []
    for interval in time_intervals:
        total_bits = sum(stream['bits_per_second'] for stream in interval['streams'])
        total_thr.append(total_bits / 1_000_000_000)  # Gbps
    return end_time, total_thr

# Carico i due file
time_tcp, thr_tcp = load_total_throughput(input_file_tcp)
time_rdma, thr_rdma = load_total_throughput(input_file_rdma)

# --- Plot ---
sns.set_style("whitegrid")
plt.figure(figsize=(10,6))

# Calcolo medie
mean_tcp = sum(thr_tcp) / len(thr_tcp)
mean_rdma = sum(thr_rdma) / len(thr_rdma)

# TCP in arancione
plt.plot(time_tcp, thr_tcp, marker='o', markersize=4, linewidth=2,
         color="tab:orange", label=f'TCP (avg: {mean_tcp:.2f} Gbps)')
plt.axhline(mean_tcp, linestyle='--', linewidth=1.2, color="tab:orange", alpha=0.6)

# RDMA in blu
plt.plot(time_rdma, thr_rdma, marker='s', markersize=4, linewidth=2,
         color="tab:blue", label=f'RDMA (avg: {mean_rdma:.2f} Gbps)')
plt.axhline(mean_rdma, linestyle='--', linewidth=1.2, color="tab:blue", alpha=0.6)

plt.title('TCP vs RDMA Throughput', fontsize=16, fontweight='bold')
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Throughput (Gbps)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_file_name}.png", dpi=300)

print("Comparison plot completed.")
