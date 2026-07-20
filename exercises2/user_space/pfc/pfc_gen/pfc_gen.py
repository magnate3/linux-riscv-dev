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
final_frame = Ether(src=src_mac, dst=dst_mac, type=0x8808)
class_pause_times = [
                randint(1, 65535) ,  #C0
                randint(1, 65535) ,  #C1
                randint(1, 65535) ,  #C2
                randint(1, 65535) ,  #C3
                randint(1, 65535) ,  #C4
                randint(1, 65535) ,  #C5
                randint(1, 65535) ,  #C6
                randint(1, 65535)    #C7
            ]
#pfc_opcode = 0x0001
pfc_opcode = 0x0101
cev_value = sum(range(8))
class_pause_times_bytes = b''.join([time.to_bytes(2, byteorder='big') for time in class_pause_times])
pfc_payload = struct.pack("!H", pfc_opcode) + cev_value.to_bytes(2, byteorder='big') + class_pause_times_bytes
padding_length = 46 - len(pfc_payload)
padding = Padding(load=b'\x00' * padding_length)
final_frame = final_frame / pfc_payload / padding
if len(sys.argv) < 2:
      print("Usage: python3 pfc_gen.py <interface> [number of packets to send]")
      sys.exit(1)
else:
      interface = sys.argv[1]

      if len(sys.argv) < 3:
          num = 1
      else:
          num = sys.argv[2]
sendp(final_frame,iface=interface, count=int(num))
