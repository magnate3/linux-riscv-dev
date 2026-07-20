# analyze_iperf_json.py
# Parse iperf3 JSON outputs from run_tcp_cc_netns.sh and produce summary CSV + charts.
import json, sys, glob, statistics
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tcp_cc_results")
CSV = OUTDIR / "summary.csv"
CH_THR = OUTDIR / "throughput_summary.png"
CH_RTT = OUTDIR / "median_rtt_summary.png"

rows = []
for f in OUTDIR.glob("*.json"):
    data = json.load(open(f))
    # iperf3 JSON: end/streams/sum_sent/bits_per_second or sum_received
    try:
        bps = data["end"]["sum_received"]["bits_per_second"]
    except KeyError:
        bps = data["end"]["sum_sent"]["bits_per_second"]
    mbps = bps / 1e6
    name = f.stem  # e.g., S1_reno
    #parts = name.split("_")
    scenario = "High BDP"
    algo = name
    #algo = parts[1] if len(parts) > 1 else "unknown"
    # Parse ping file for RTTs
    ping_file = OUTDIR / f"{scenario}_{algo}_ping.txt"
    rtts = []
    if ping_file.exists():
        with open(ping_file) as pf:
            for line in pf:
                if "time=" in line:
                    try:
                        rtts.append(float(line.split("time=")[1].split()[0]))
                    except Exception:
                        pass
    med_rtt = statistics.median(rtts) if rtts else None
    rows.append({"Scenario": scenario, "Algorithm": algo, "Mean Throughput (Mbps)": round(mbps,2), "Median RTT (ms)": med_rtt})

df = pd.DataFrame(rows).sort_values(by=["Scenario","Algorithm"])
df.to_csv(CSV, index=False)

# Charts
if not df.empty:
    p1 = df.pivot_table(index="Scenario", columns="Algorithm", values="Mean Throughput (Mbps)")
    ax = p1.plot(kind="bar", figsize=(10,5))
    ax.set_title("Mean Throughput by Algorithm (from iperf3)")
    ax.set_ylabel("Mbps")
    ax.set_xlabel("Scenario")
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(CH_THR, dpi=200)
    plt.close(ax.get_figure())

    p2 = df.pivot_table(index="Scenario", columns="Algorithm", values="Median RTT (ms)")
    ax2 = p2.plot(kind="bar", figsize=(10,5))
    ax2.set_title("Median RTT by Algorithm (from ping)")
    ax2.set_ylabel("ms")
    ax2.set_xlabel("Scenario")
    ax2.get_figure().tight_layout()
    ax2.get_figure().savefig(CH_RTT, dpi=200)
    plt.close(ax2.get_figure())

print(f"Wrote {CSV}")
print(f"Charts: {CH_THR} {CH_RTT}")
