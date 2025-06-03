Hardware latency measurement based on PTP

This app relies on hardware timestamping to measure round trip latency.


0. PTP source clock is copied to the other ports
1. PTP SYNC packet is sent
2. TX timestamp is read, and sent in a subsequent FOLLOW_UP packet
3. On receive side, we read the RX timestamp of SYNC packet and compare
   with the FUP packet

