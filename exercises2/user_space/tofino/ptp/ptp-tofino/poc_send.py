#!/usr/bin/env python3

from scapy.all import *

INTERVAL = 1
COUNT = 10
INTERFACE = 'ens1'
ETH_P_802_EX1 = 0x88b5

# Send packets out interface 24/0 (60, 0x3C)
pkt = Ether(src='11:22:33:44:55:66', dst='aa:bb:cc:dd:ee:ff', type=ETH_P_802_EX1) / b'\x00\x3c' / (b'\x00' * 32) / (b'\xab' * 12)


sendp(pkt, iface=INTERFACE, inter=INTERVAL, count=COUNT)
