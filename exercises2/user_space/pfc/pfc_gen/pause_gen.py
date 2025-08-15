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
pause_frame = Ether(
    dst="01:80:C2:00:00:01",  # 泛洪地址，用于流控暂停帧
    src="F4:84:8D:8A:0C:5C",  # 发送方 MAC 地址，例如：00:11:22:33:44:55
    type=0x8808  # 表示以太网控制帧
) / Raw(
    load="\x00\x01\xff\x79\x00\x00"  # 控制帧中的Pause Opcode和暂停时间
)
if len(sys.argv) < 2:
      print("Usage: python3 pfc_gen.py <interface> [number of packets to send]")
      sys.exit(1)
else:
      interface = sys.argv[1]

      if len(sys.argv) < 3:
          num = 1
      else:
          num = sys.argv[2]
sendp(pause_frame,iface=interface, count=int(num))
