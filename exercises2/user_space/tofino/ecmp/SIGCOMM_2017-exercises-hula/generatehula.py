#!/usr/bin/env python
import argparse
import sys
import socket
import random
import struct

from scapy.all import sendp, send, get_if_list, get_if_hwaddr, bind_layers
from scapy.all import Packet
from scapy.all import Ether, IP, UDP
from scapy.fields import *
from time import sleep
import crcmod

class Hula(Packet):
   fields_desc = [ BitField("dir", 0, 1),
                   BitField("qdepth", 0, 15),
                   XIntField("digest", None)]
   def post_build(self, p, pay):
        p += pay
        if self.digest is None:
            crc32 = crcmod.Crc(0x104c11db7, initCrc=0, xorOut=0xFFFFFFFF)
            crc32.update(str(p));
            c = bytes(bytearray.fromhex("%08x" % crc32.crcValue))
            p = p[:2]+ c +p[6:]
            #ck = checksum(p)
            #p = p[:2]+"\x00\x00"+chr(ck>>8)+chr(ck&0xff)+p[6:]
        return p

class SourceRoute(Packet):
   fields_desc = [ BitField("bos", 0, 1),
                   BitField("port", 0, 15)]

bind_layers(Ether, Hula, type=0x2345)
bind_layers(Hula, SourceRoute)
bind_layers(SourceRoute, SourceRoute, bos=0)
bind_layers(SourceRoute, IP, bos=1)

def main():
    period = 0
    if len(sys.argv) > 1:
        period = int(sys.argv[1])

            # src,     dst     , src routing , interface
    info = [
            ("10.0.1.0", "10.0.2.0", (2, 2, 1, 1), "s1-eth1"),
            ("10.0.1.0", "10.0.2.0", (3, 2, 1, 1), "s1-eth1"),
            ("10.0.1.0", "10.0.3.0", (2, 3, 1, 1), "s1-eth1"),
            ("10.0.1.0", "10.0.3.0", (3, 3, 1, 1), "s1-eth1"),
            ("10.0.2.0", "10.0.1.0", (2, 1, 2, 1), "s2-eth1"),
            ("10.0.2.0", "10.0.1.0", (3, 1, 2, 1), "s2-eth1"),
            ("10.0.2.0", "10.0.3.0", (2, 3, 2, 1), "s2-eth1"),
            ("10.0.2.0", "10.0.3.0", (3, 3, 2, 1), "s2-eth1"),
            ("10.0.3.0", "10.0.1.0", (2, 1, 3, 1), "s3-eth1"),
            ("10.0.3.0", "10.0.1.0", (3, 1, 3, 1), "s3-eth1"),
            ("10.0.3.0", "10.0.2.0", (2, 2, 3, 1), "s3-eth1"),
            ("10.0.3.0", "10.0.2.0", (3, 2, 3, 1), "s3-eth1")]


    try:
      while True:
          for e in info:
            ports = e[2]
            pkt =  Ether(src=get_if_hwaddr(e[3]), dst='ff:ff:ff:ff:ff:ff')
            pkt = pkt / Hula(dir=0, qdepth=0)
            pkt = pkt / SourceRoute(bos=0, port=ports[0])
            pkt = pkt / SourceRoute(bos=0, port=ports[1])
            pkt = pkt / SourceRoute(bos=0, port=ports[2])
            pkt = pkt / SourceRoute(bos=1, port=ports[3])
            pkt = pkt / IP(dst=e[1], src=e[0]) / UDP(dport=4321, sport=1234)
            #pkt.show2()
            sendp(pkt, iface=e[3], verbose=False)
          if period == 0:
              break;
          else:
              sleep(period)
    except KeyboardInterrupt:
        raise


if __name__ == '__main__':
    main()
