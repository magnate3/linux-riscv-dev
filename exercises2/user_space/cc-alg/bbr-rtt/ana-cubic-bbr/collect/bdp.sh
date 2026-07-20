
#!/usr/bin/env bash
# Helper: reset qdisc
IFACE="enp61s0f1np1"
reset_qdisc() {
   tc qdisc del dev "$IFACE" root 2>/dev/null || true
   tc qdisc del dev "$IFACE" ingress 2>/dev/null || true
}

# Apply scenario shaping on NS_B (server egress / client ingress)
# We shape B-side egress, which is A-side ingress for forward direction.
# High BDP (50Mbps, 100ms)
apply_high_bdp() {
  # 50Mbps, 100ms delay, reasonable queue
  reset_qdisc
  tc qdisc add dev "$IFACE" root handle 1: netem delay 100ms
  tc qdisc add dev "$IFACE" parent 1: handle 10: tbf rate 50mbit burst 64kbit latency 50ms
  tc qdisc add dev "$IFACE" parent 10: pfifo limit 1000
}

apply_bufferbloat() {
  # Big buffer (pfifo), same link
  reset_qdisc
  tc qdisc add dev "$IFACE" root handle 1: netem delay 50ms
  tc qdisc add dev "$IFACE" parent 1: handle 10: tbf rate 50mbit burst 256kbit latency 200ms
  tc qdisc add dev "$IFACE" parent 10: pfifo limit 20000
}
apply_high_bdp
#apply_bufferbloat
