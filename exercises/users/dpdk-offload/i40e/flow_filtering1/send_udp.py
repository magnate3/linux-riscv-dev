#!/usr/bin/python3
import logging
import argparse
from scapy.all import *
from scapy.utils import rdpcap
import random
import time
import signal
new_src_mac = "f4:1d:6b:f7:bf:ab"
#new_src_mac = "b0:08:75:5f:b8:5b"
new_dst_mac = "F4:1D:6B:F7:BF:96"
def main():
    print("send udp pkt")
    payload = 'a' * 100
    packet = IP(dst="10.11.11.65",id=12345)/UDP(dport=5000)/payload
    frags=fragment(packet,fragsize=500) 
    for frag in frags:
        pkt = Ether(src= new_src_mac,dst= new_dst_mac,type=0x800)/frag
        sendp(pkt,verbose=0,iface='eno2') #sending packet at layer 2
if __name__ == '__main__':
    main()
