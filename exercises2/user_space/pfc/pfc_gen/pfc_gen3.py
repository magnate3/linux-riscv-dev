#!/usr/bin/env python3
import sys
import os
import argparse
import time
import scapy
from random import randint
#from scapy.contrib.mac_control import *
from scapy.all import *
from os import urandom


PAUSE_OPCODE	= 0x0001		# Pause frame IEEE 802.3x
PFC_OPCODE	= 0x0101		# Priority Flow Control IEEE 802.1Qbb
# sport_value = 1234
src_mac = "00:AA:BB:CC:DD:EE"
DEFAULT_DST_MAC = "01:80:c2:00:00:01"
#dst_mac = "00:90:fb:79:20:55"

dst_mac = DEFAULT_DST_MAC
from scapy.all import Ether, sendp

# Define Ethernet PFC Frame
pfc_frame = Ether(dst="01:80:C2:00:00:01", src="00:11:22:33:44:55", type=0x8808) / \
            b"\x01\x01"  # Opcode: 0x0101 (PFC)
            
# Append Priority Enable Vector (8 bits) and Pause Quanta (16 bits per priority)
pfc_frame = pfc_frame /b"\x08" + b"\x00\x40" + b"\x00\x00" * 7 
 # Enabling priority 3 with pause quanta

# Send PFC frame on a specific interface
if len(sys.argv) < 2:
      print("Usage: python3 pfc_gen.py <interface> [number of packets to send]")
      sys.exit(1)
else:
      interface = sys.argv[1]

      if len(sys.argv) < 3:
          num = 1
      else:
          num = sys.argv[2]
sendp(pfc_frame,iface=interface, count=int(num))
