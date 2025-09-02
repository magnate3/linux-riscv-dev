# import pyshark
# import matplotlib.pyplot as plt
# import pandas as pd
# import os

# algorithms = ['reno', 'tahoe', 'cubic']
# results = {}

# # Explicit path to tshark, if needed
# tshark_path = '/usr/bin/tshark'  # or wherever your `which tshark` says

# cap = pyshark.FileCapture('traces/reno.pcap', tshark_path=tshark_path)

# for alg in algorithms:
#     cap = pyshark.FileCapture(f'traces/{alg}.pcap', display_filter='tcp')
    
#     rtts = []
#     cwnd = []
#     throughput = []

#     timestamps = []
#     total_bytes = 0
#     start_time = None

#     for pkt in cap:
#         try:
#             ts = float(pkt.sniff_timestamp)
#             if start_time is None:
#                 start_time = ts

#             if hasattr(pkt.tcp, 'analysis_ack_rtt'):
#                 rtts.append(float(pkt.tcp.analysis_ack_rtt) * 1000)  # ms

#             # Bytes for throughput
#             if 'LEN=' in pkt.tcp._all_fields:
#                 total_bytes += int(pkt.length)

#             timestamps.append(ts - start_time)
#         except:
#             continue
#     cap.close()

#     # Compute throughput over the session
#     duration = timestamps[-1] if timestamps else 1
#     throughput_val = (total_bytes * 8) / (duration * 1e6)  # Mbps

#     # Save values
#     results[alg] = {
#         'RTT': rtts,
#         'Throughput': throughput_val,
#     }

# # Plot RTTs
# plt.figure(figsize=(10, 6))
# for alg in algorithms:
#     plt.plot(pd.Series(results[alg]['RTT']).rolling(10).mean(), label=alg)
# plt.title("RTT (ms) Over Time")
# plt.xlabel("Packet #")
# plt.ylabel("RTT (ms)")
# plt.legend()
# plt.grid()
# plt.savefig("plots/rtt_plot.png")
# plt.show()

# # Plot Throughput
# plt.figure(figsize=(8, 5))
# for alg in algorithms:
#     plt.bar(alg, results[alg]['Throughput'])
# plt.title("Throughput Comparison (Mbps)")
# plt.ylabel("Throughput (Mbps)")
# plt.grid(axis='y')
# plt.savefig("plots/throughput_plot.png")
# plt.show()









# import pyshark
# import matplotlib.pyplot as plt
# import os

# # Path to your capture file
# pcap_file = 'traces/reno.pcap'

# # Optional: Set tshark path manually if needed
# # Example: tshark_path = '/usr/bin/tshark'
# tshark_path = None

# # Try to open the file
# if not os.path.exists(pcap_file):
#     print(f"[!] File not found: {pcap_file}")
#     exit()

# try:
#     cap = pyshark.FileCapture(pcap_file, tshark_path=tshark_path, display_filter='tcp')
#     cap.load_packets()
# except Exception as e:
#     print(f"[!] Failed to load pcap: {e}")
#     exit()

# # --- Analysis ---
# rtts = []
# throughputs = []
# seqs = []
# times = []

# first_time = None
# packet_count = 0

# print("[*] Starting analysis...")

# for pkt in cap:
#     try:
#         tcp = pkt.tcp
#         ts = float(pkt.sniff_timestamp)
#         if first_time is None:
#             first_time = ts
#         time_offset = ts - first_time

#         if hasattr(tcp, 'analysis_ack_rtt'):
#             rtt = float(tcp.analysis_ack_rtt) * 1000  # in ms
#             rtts.append((time_offset, rtt))

#         if hasattr(tcp, 'seq') and hasattr(pkt, 'length'):
#             seqs.append(int(tcp.seq))
#             times.append(time_offset)
#             packet_count += 1

#     except AttributeError:
#         continue

# cap.close()

# print(f"[*] Packets analyzed: {packet_count}")

# # --- Plot RTT ---
# if rtts:
#     x_rtt, y_rtt = zip(*rtts)
#     plt.figure(figsize=(10, 4))
#     plt.plot(x_rtt, y_rtt, color='orange')
#     plt.title("TCP RTT over Time")
#     plt.xlabel("Time (s)")
#     plt.ylabel("RTT (ms)")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig("rtt_plot.png")
#     plt.show()
# else:
#     print("[!] No RTT data found.")

# # --- Plot CWND (approx using seq spacing) ---
# if seqs and times:
#     plt.figure(figsize=(10, 4))
#     plt.plot(times, seqs, color='blue')
#     plt.title("TCP Sequence Number over Time (CWND Approximation)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Sequence Number")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig("cwnd_plot.png")
#     plt.show()
# else:
#     print("[!] No sequence data found.")

# # --- Throughput estimate ---
# if seqs and times:
#     total_bytes = seqs[-1] - seqs[0]
#     duration = times[-1] - times[0]
#     throughput = total_bytes / duration / 1000  # KB/s
#     print(f"[*] Estimated throughput: {throughput:.2f} KB/s")
# else:
#     print("[!] Cannot compute throughput.")



# import pyshark
# import matplotlib.pyplot as plt
# import os

# # Function to run the analysis for each congestion control file
# def analyze_congestion_control(pcap_file, congestion_type, tshark_path=None):
#     if not os.path.exists(pcap_file):
#         print(f"[!] File not found: {pcap_file}")
#         return

#     try:
#         cap = pyshark.FileCapture(pcap_file, tshark_path=tshark_path, display_filter='tcp')
#         cap.load_packets()
#     except Exception as e:
#         print(f"[!] Failed to load pcap for {congestion_type}: {e}")
#         return

#     # --- Analysis ---
#     rtts = []
#     seqs = []
#     times = []

#     first_time = None
#     packet_count = 0

#     print(f"[*] Starting analysis for {congestion_type}...")

#     for pkt in cap:
#         try:
#             tcp = pkt.tcp
#             ts = float(pkt.sniff_timestamp)
#             if first_time is None:
#                 first_time = ts
#             time_offset = ts - first_time

#             if hasattr(tcp, 'analysis_ack_rtt'):
#                 rtt = float(tcp.analysis_ack_rtt) * 1000  # in ms
#                 rtts.append((time_offset, rtt))

#             if hasattr(tcp, 'seq') and hasattr(pkt, 'length'):
#                 seqs.append(int(tcp.seq))
#                 times.append(time_offset)
#                 packet_count += 1

#         except AttributeError:
#             continue

#     cap.close()

#     print(f"[*] Packets analyzed for {congestion_type}: {packet_count}")

#     return rtts, seqs, times

# # Paths to your pcap files for each congestion control type
# pcap_files = {
#     'Reno': 'traces/reno.pcap',
#     'Tahoe': 'traces/tahoe.pcap',
#     'Cubic': 'traces/cubic.pcap',
# }

# # Initialize lists for RTT and CWND data
# all_rtts = {}
# all_seqs = {}
# all_times = {}

# # Analyze all congestion control types
# for congestion_type, pcap_file in pcap_files.items():
#     rtts, seqs, times = analyze_congestion_control(pcap_file, congestion_type)
#     if rtts:
#         all_rtts[congestion_type] = rtts
#     if seqs and times:
#         all_seqs[congestion_type] = (seqs, times)

# # --- Plot RTT ---
# plt.figure(figsize=(10, 4))
# for congestion_type, rtts in all_rtts.items():
#     x_rtt, y_rtt = zip(*rtts)
#     plt.plot(x_rtt, y_rtt, label=f"{congestion_type} RTT")
# plt.title("TCP RTT over Time")
# plt.xlabel("Time (s)")
# plt.ylabel("RTT (ms)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("rtt_comparison.png")
# plt.show()

# # --- Plot CWND (approx using seq spacing) ---
# plt.figure(figsize=(10, 4))
# for congestion_type, (seqs, times) in all_seqs.items():
#     plt.plot(times, seqs, label=f"{congestion_type} CWND")
# plt.title("TCP Sequence Number over Time (CWND Approximation)")
# plt.xlabel("Time (s)")
# plt.ylabel("Sequence Number")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("cwnd_comparison.png")
# plt.show()

# # --- Throughput estimate ---
# for congestion_type, (seqs, times) in all_seqs.items():
#     total_bytes = seqs[-1] - seqs[0]
#     duration = times[-1] - times[0]
#     throughput = total_bytes / duration / 1000  # KB/s
#     print(f"[*] Estimated throughput for {congestion_type}: {throughput:.2f} KB/s")




# import pyshark
# import matplotlib.pyplot as plt
# import os

# # Function to run the analysis for each congestion control file
# def analyze_congestion_control(pcap_file, congestion_type, tshark_path=None):
#     if not os.path.exists(pcap_file):
#         print(f"[!] File not found: {pcap_file}")
#         return

#     try:
#         cap = pyshark.FileCapture(pcap_file, tshark_path=tshark_path, display_filter='tcp')
#         cap.load_packets()
#     except Exception as e:
#         print(f"[!] Failed to load pcap for {congestion_type}: {e}")
#         return

#     # --- Analysis ---
#     rtts = []
#     seqs = []
#     times = []
#     cwnds = []

#     first_time = None
#     last_seq = None

#     print(f"[*] Starting analysis for {congestion_type}...")

#     for pkt in cap:
#         try:
#             tcp = pkt.tcp
#             ts = float(pkt.sniff_timestamp)
#             if first_time is None:
#                 first_time = ts
#             time_offset = ts - first_time

#             # Calculate RTT
#             if hasattr(tcp, 'analysis_ack_rtt'):
#                 rtt = float(tcp.analysis_ack_rtt) * 1000  # in ms
#                 rtts.append((time_offset, rtt))

#             # Track sequence numbers and approximate CWND size
#             if hasattr(tcp, 'seq') and hasattr(pkt, 'length'):
#                 seqs.append(int(tcp.seq))
#                 times.append(time_offset)
#                 if last_seq is not None:
#                     cwnd_size = seqs[-1] - last_seq  # Approximate CWND as difference in seq numbers
#                     cwnds.append((time_offset, cwnd_size))
#                 last_seq = seqs[-1]

#         except AttributeError:
#             continue

#     cap.close()

#     print(f"[*] Packets analyzed for {congestion_type}: {len(seqs)}")

#     return rtts, cwnds, seqs, times

# # Paths to your pcap files for each congestion control type
# pcap_files = {
#     'Reno': 'traces/reno.pcap',
#     'Tahoe': 'traces/tahoe.pcap',
#     'Cubic': 'traces/cubic.pcap',
# }

# # Initialize lists for RTT and CWND data
# all_rtts = {}
# all_cwnds = {}
# all_seqs = {}
# all_times = {}

# # Analyze all congestion control types
# for congestion_type, pcap_file in pcap_files.items():
#     rtts, cwnds, seqs, times = analyze_congestion_control(pcap_file, congestion_type)
#     if rtts:
#         all_rtts[congestion_type] = rtts
#     if cwnds:
#         all_cwnds[congestion_type] = cwnds
#     if seqs and times:
#         all_seqs[congestion_type] = (seqs, times)

# # --- Plot RTT, CWND, and Throughput in Subplots ---
# fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True)

# # --- RTT Plot for each congestion type ---
# for idx, (congestion_type, rtts) in enumerate(all_rtts.items()):
#     ax = axes[0, idx]
#     x_rtt, y_rtt = zip(*rtts)
#     ax.plot(x_rtt, y_rtt, label=f"{congestion_type} RTT", linewidth=2)
#     ax.set_title(f"{congestion_type} - RTT over Time")
#     ax.set_ylabel("RTT (ms)")
#     ax.grid(True)
#     ax.legend()

# # --- CWND Plot for each congestion type ---
# for idx, (congestion_type, cwnds) in enumerate(all_cwnds.items()):
#     ax = axes[1, idx]
#     if cwnds:
#         x_cwnd, y_cwnd = zip(*cwnds)
#         ax.plot(x_cwnd, y_cwnd, label=f"{congestion_type} CWND", linewidth=2)
#         ax.set_title(f"{congestion_type} - CWND over Time")
#         ax.set_ylabel("CWND Size (bytes)")
#         ax.grid(True)
#         ax.legend()

# # --- Throughput Plot for each congestion type ---
# for idx, (congestion_type, (seqs, times)) in enumerate(all_seqs.items()):
#     ax = axes[2, idx]
#     total_bytes = seqs[-1] - seqs[0]
#     duration = times[-1] - times[0]
#     throughput = total_bytes / duration / 1000  # KB/s
#     ax.plot(times, [throughput] * len(times), label=f"{congestion_type} Throughput", linewidth=2)
#     ax.set_title(f"{congestion_type} - Throughput")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Throughput (KB/s)")
#     ax.grid(True)
#     ax.legend()

# # Adjusting the scale and making the x-axis range from 0 to around 8 seconds or 5 seconds
# for ax in axes.flat:
#     ax.set_xlim(0, 8)

# # Tight layout for clean plotting
# plt.tight_layout()

# # Saving and showing the plot
# plt.savefig("tcp_comparison_plots_separate.png")
# plt.show()

# # Print Throughput Estimates for each congestion control algorithm
# for congestion_type, (seqs, times) in all_seqs.items():
#     total_bytes = seqs[-1] - seqs[0]
#     duration = times[-1] - times[0]
#     throughput = total_bytes / duration / 1000  # KB/s
#     print(f"[*] Estimated throughput for {congestion_type}: {throughput:.2f} KB/s")







import pyshark
import matplotlib.pyplot as plt
import os

# Function to run the analysis for each congestion control file
def analyze_congestion_control(pcap_file, congestion_type, tshark_path=None):
    if not os.path.exists(pcap_file):
        print(f"[!] File not found: {pcap_file}")
        return

    try:
        cap = pyshark.FileCapture(pcap_file, tshark_path=tshark_path, display_filter='tcp')
        cap.load_packets()
    except Exception as e:
        print(f"[!] Failed to load pcap for {congestion_type}: {e}")
        return

    # --- Analysis ---
    rtts = []
    seqs = []
    times = []
    cwnds = []

    first_time = None
    last_seq = None

    print(f"[*] Starting analysis for {congestion_type}...")

    for pkt in cap:
        try:
            tcp = pkt.tcp
            ts = float(pkt.sniff_timestamp)
            if first_time is None:
                first_time = ts
            time_offset = ts - first_time

            # Calculate RTT
            if hasattr(tcp, 'analysis_ack_rtt'):
                rtt = float(tcp.analysis_ack_rtt) * 1000  # in ms
                rtts.append((time_offset, rtt))

            # Track sequence numbers and approximate CWND size
            if hasattr(tcp, 'seq') and hasattr(pkt, 'length'):
                seqs.append(int(tcp.seq))
                times.append(time_offset)
                if last_seq is not None:
                    cwnd_size = seqs[-1] - last_seq  # Approximate CWND as difference in seq numbers
                    cwnds.append((time_offset, cwnd_size))
                last_seq = seqs[-1]

        except AttributeError:
            continue

    cap.close()

    print(f"[*] Packets analyzed for {congestion_type}: {len(seqs)}")

    return rtts, cwnds, seqs, times

# Paths to your pcap files for each congestion control type
pcap_files = {
    'Reno': 'traces/reno.pcap',
    'Tahoe': 'traces/tahoe.pcap',
    'Cubic': 'traces/cubic.pcap',
}

# Initialize lists for RTT and CWND data
all_rtts = {}
all_cwnds = {}
all_seqs = {}
all_times = {}

# Analyze all congestion control types
for congestion_type, pcap_file in pcap_files.items():
    rtts, cwnds, seqs, times = analyze_congestion_control(pcap_file, congestion_type)
    if rtts:
        all_rtts[congestion_type] = rtts
    if cwnds:
        all_cwnds[congestion_type] = cwnds
    if seqs and times:
        all_seqs[congestion_type] = (seqs, times)

# --- Plot RTT, CWND, and Throughput in Subplots ---
fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True)

# --- RTT Plot for each congestion type ---
for idx, (congestion_type, rtts) in enumerate(all_rtts.items()):
    ax = axes[0, idx]
    x_rtt, y_rtt = zip(*rtts)
    ax.plot(x_rtt, y_rtt, label=f"{congestion_type} RTT", linewidth=2)
    ax.set_title(f"{congestion_type} - RTT over Time")
    ax.set_ylabel("RTT (ms)")
    
    ax.grid(True)
    ax.legend()

# --- CWND Plot for each congestion type (Increased scale) ---
for idx, (congestion_type, cwnds) in enumerate(all_cwnds.items()):
    ax = axes[1, idx]
    if cwnds:
        x_cwnd, y_cwnd = zip(*cwnds)
        ax.plot(x_cwnd, y_cwnd, label=f"{congestion_type} CWND", linewidth=2)
        ax.set_title(f"{congestion_type} - CWND over Time")
        ax.set_ylabel("CWND Size (bytes)")
        ax.grid(True)
        ax.legend()
        
        ax.set_xlim(0, 5)  # Increase the Y-axis range for better visibility

# --- Throughput Plot for each congestion type ---
for idx, (congestion_type, (seqs, times)) in enumerate(all_seqs.items()):
    ax = axes[2, idx]
    total_bytes = seqs[-1] - seqs[0]
    duration = times[-1] - times[0]
    throughput = total_bytes / duration / 1000  # KB/s
    ax.plot(times, [throughput] * len(times), label=f"{congestion_type} Throughput", linewidth=2)
    ax.set_title(f"{congestion_type} - Throughput")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (KB/s)")
    ax.grid(True)
    ax.legend()

# Adjusting the scale and making the x-axis range from 0 to around 50 seconds
for ax in axes.flat:
    ax.set_xlim(0, 1)

# Tight layout for clean plotting
plt.tight_layout()

# Saving and showing the plot
plt.savefig("tcp_comparison_17.png")
plt.show()

# Print Throughput Estimates for each congestion control algorithm
for congestion_type, (seqs, times) in all_seqs.items():
    total_bytes = seqs[-1] - seqs[0]
    duration = times[-1] - times[0]
    throughput = total_bytes / duration / 1000  # KB/s
    print(f"[*] Estimated throughput for {congestion_type}: {throughput:.2f} KB/s")
