#!/usr/bin/env bash
# run_tcp_cc_netns.sh
# Reproducible TCP CC experiments using Linux network namespaces + tc netem/tbf + iperf3
# REQUIREMENTS: iproute2 (ip), tc, iperf3, jq, ping
# Run as root (sudo). Tested on modern Linux kernels (>=5.x).
# References: RFC 5681, RFC 6298, RFC 9438; iperf3 (-C), tc-netem, tbf.

set -euo pipefail

OUTDIR="${OUTDIR:-/tmp/tcp_cc_results}"
DURATION="${DURATION:-30}"         # seconds per run
IFACE_A="vethA"
IFACE_B="vethB"
NS_A="nsA"
NS_B="nsB"

cleanup() {
  ip netns del "$NS_A" 2>/dev/null || true
  ip netns del "$NS_B" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$OUTDIR"

echo "[*] Creating netns and veth pair ..."
cleanup
ip netns add "$NS_A"
ip netns add "$NS_B"
ip link add "$IFACE_A" type veth peer name "$IFACE_B"
ip link set "$IFACE_A" netns "$NS_A"
ip link set "$IFACE_B" netns "$NS_B"

ip netns exec "$NS_A" ip addr add 10.0.0.1/24 dev "$IFACE_A"
ip netns exec "$NS_B" ip addr add 10.0.0.2/24 dev "$IFACE_B"
ip netns exec "$NS_A" ip link set lo up
ip netns exec "$NS_B" ip link set lo up
ip netns exec "$NS_A" ip link set "$IFACE_A" up
ip netns exec "$NS_B" ip link set "$IFACE_B" up

# Ensure algorithms available
for mod in tcp_bbr tcp_vegas; do
  modprobe "$mod" 2>/dev/null || true
done

# Helper: reset qdisc
reset_qdisc() {
  ip netns exec "$NS_B" tc qdisc del dev "$IFACE_B" root 2>/dev/null || true
  ip netns exec "$NS_B" tc qdisc del dev "$IFACE_B" ingress 2>/dev/null || true
}

# Apply scenario shaping on NS_B (server egress / client ingress)
# We shape B-side egress, which is A-side ingress for forward direction.
apply_s1_high_bdp() {
  # 50Mbps, 100ms delay, reasonable queue
  reset_qdisc
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" root handle 1: netem delay 100ms
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" parent 1: handle 10: tbf rate 50mbit burst 64kbit latency 50ms
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" parent 10: pfifo limit 1000
}

apply_s2_bufferbloat() {
  # Big buffer (pfifo), same link
  reset_qdisc
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" root handle 1: netem delay 50ms
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" parent 1: handle 10: tbf rate 50mbit burst 256kbit latency 200ms
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" parent 10: pfifo limit 20000
}

apply_s3_wireless_loss() {
  # Random loss 1%, modest delay
  reset_qdisc
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" root handle 1: netem delay 40ms loss 1%
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" parent 1: handle 10: tbf rate 50mbit burst 64kbit latency 50ms
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" parent 10: pfifo limit 1000
}

apply_s4_competition() {
  # Two clients share the same shaped link
  reset_qdisc
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" root handle 1: netem delay 50ms
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" parent 1: handle 10: tbf rate 50mbit burst 128kbit latency 100ms
  ip netns exec "$NS_B" tc qdisc add dev "$IFACE_B" parent 10: pfifo limit 2000
}

iperf_server_start() {
  ip netns exec "$NS_B" pkill -f "iperf3 -s" 2>/dev/null || true
  ip netns exec "$NS_B" sh -c "iperf3 -s -1 > /dev/null 2>&1 &"
  sleep 0.5
}

run_one() {
  local scenario="$1" algo="$2"
  local outjson="${OUTDIR}/${scenario}_${algo}.json"
  echo "[*] Scenario=${scenario} Algo=${algo} ..."
  iperf_server_start
  # Ping in background to estimate RTT during run
  ip netns exec "$NS_A" sh -c "ping -i 0.2 -w ${DURATION} 10.0.0.2 > ${OUTDIR}/${scenario}_${algo}_ping.txt" &
  # iperf3 with per-connection CC (-C) if supported
  ip netns exec "$NS_A" sh -c "iperf3 -J -t ${DURATION} -c 10.0.0.2 -C ${algo} > ${outjson}"
  wait || true
}

run_competition() {
  local algo="$1"  # competing alg vs CUBIC
  local s="S4"
  echo "[*] Scenario=S4 competition ${algo} vs CUBIC ..."
  iperf_server_start
  # Two parallel clients, one with CUBIC, one with $algo
  ip netns exec "$NS_A" sh -c "ping -i 0.2 -w ${DURATION} 10.0.0.2 > ${OUTDIR}/${s}_${algo}_vs_CUBIC_ping.txt" &
  ip netns exec "$NS_A" sh -c "iperf3 -J -t ${DURATION} -c 10.0.0.2 -p 5201 -C cubic > ${OUTDIR}/${s}_CUBIC.json" &
  ip netns exec "$NS_A" sh -c "iperf3 -J -t ${DURATION} -c 10.0.0.2 -p 5202 -C ${algo} > ${OUTDIR}/${s}_${algo}.json" &
  wait || true
}

echo "[*] Running S1: High BDP"
apply_s1_high_bdp
for algo in reno cubic vegas bbr; do
  run_one "S1" "$algo" || true
done

echo "[*] Running S2: Bufferbloat"
apply_s2_bufferbloat
for algo in reno cubic vegas bbr; do
  run_one "S2" "$algo" || true
done

echo "[*] Running S3: Wireless loss 1%"
apply_s3_wireless_loss
for algo in reno cubic vegas bbr; do
  run_one "S3" "$algo" || true
done

echo "[*] Running S4: Competition vs CUBIC"
apply_s4_competition
# Compare BBR and (optionally) Vegas vs CUBIC; you can add others
run_competition "bbr" || true
run_competition "vegas" || true

echo "[*] Done. Outputs in ${OUTDIR}"
