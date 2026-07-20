import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os

if len(sys.argv) <= 1:
    print('Missing file name argument\nUsage: python3 iperf_plotter.py file.json')
    sys.exit(0)

input_file_name = sys.argv[1]
output_file_name = os.path.splitext(input_file_name)[0]

with open(input_file_name) as json_file:
    json_data = json.load(json_file)

time_intervals = json_data['intervals']

end_time = [int(item['sum']['end']) for item in time_intervals]

# --- Throughput per stream ---
num_streams = len(time_intervals[0]['streams'])
throughput_streams = [[] for _ in range(num_streams)]

for interval in time_intervals:
    for i, stream in enumerate(interval['streams']):
        throughput_streams[i].append(stream['bits_per_second'] / 1_000_000_000)  # Gbps

# Set professional style
sns.set_style("whitegrid")

# --- Plot throughput for each stream ---
plt.figure(figsize=(10,6))
colors = sns.color_palette("tab10", num_streams)  # Distinct colors for each stream

for i in range(num_streams):
    mean_val = sum(throughput_streams[i]) / len(throughput_streams[i])
    
    # Modifica il label per includere la media
    plt.plot(
        end_time, 
        throughput_streams[i], 
        marker='o', markersize=4, linewidth=2, color=colors[i],
        label=f'Stream {i+1} (avg: {mean_val:.2f} Gbps)'
    )
    
    # Linea tratteggiata della media
    plt.axhline(mean_val, linestyle='--', linewidth=1.5, color=colors[i], alpha=0.5)

plt.title('Throughput per Stream', fontsize=16, fontweight='bold')
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Throughput (Gbps)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_file_name}_throughput.png", dpi=300)

print("Plot completed.")