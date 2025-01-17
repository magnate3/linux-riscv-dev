#!/usr/bin/env python
import argparse
import sys
import socket
import random
import struct
from time import sleep

from scapy.all import sendp, send, get_if_list, get_if_hwaddr
from scapy.all import Packet
from scapy.all import Ether, IP, UDP, TCP

def get_if():
    ifs=get_if_list()
    iface=None # "h1-eth0"
    for i in get_if_list():
        if "wlan1" in i:
            iface=i
            break;
    if not iface:
        print("Cannot find eth0 interface")
        exit(1)
    return iface

def main(args):

    addr = socket.gethostbyname(args.ip)
    #iface = socket.gethostbyname(args.itf)
    iface = get_if()


    #print("sending on interface %s to %s" % (iface, str(addr)))
    #pkt =  Ether(src='00:01:0a:00:03:05',dst='e4:a4:71:cd:52:99')
    pkt =  Ether(src=get_if_hwaddr(iface), dst='9a:4b:89:ad:8a:59')
    #pkt =  Ether(src=get_if_hwaddr(iface),dst='00:00:0a:00:03:04')
    #ff:ff:ff:ff:ff:ff
    #00:00:0a:00:03:02
    if(args.l4 == 'tcp'):
        pkt = pkt /IP(dst=addr) / TCP(dport=args.port, sport=random.randint(49152,65535)) / args.m
    if(args.l4 == 'udp'):
        #pkt = pkt /IP(dst=addr,src='10.0.2.18') / UDP(dport=int(args.port), sport=1234) /
        pkt = pkt /IP(dst=addr) / UDP(dport=int(args.port), sport=1234) / \
b"\x20\x40\x02\xed\x00\x00\
\x00\x03\x13\x00\x00\x20\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x0a\x00\x03\x02\x00\x00\x0a\x00\x03\x02\x08\x00\x45\x5c\x00\xb5\
\x00\x01\x00\x00\x3e\x11\x63\xd9\x0a\x00\x01\x01\x0a\x00\x03\x02\
\xe0\x03\x00\x50\x00\xa1\x38\xe7\x10\x24\x00\x00\x20\x00\x0b\x07\
\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x04\x00\x01\
\x00\x00\x05\x7a\x00\x00\x00\x00\x00\x00\x00\x01\x4e\x0b\x22\x1f\
\x00\x00\x00\x01\x4e\x0b\x27\x99\x00\x00\x00\x04\x00\x00\x00\x01\
\x00\x00\x00\x00\x00\x00\x00\x02\x00\x01\x00\x02\x00\x00\x04\x0b\
\x00\x00\x00\x00\x00\x00\x00\x01\x4e\x1f\x51\xb3\x00\x00\x00\x01\
\x4e\x1f\x55\xbe\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x00\
\x00\x00\x00\x01\x00\x01\x00\x02\x00\x00\x04\xcd\x00\x00\x00\x00\
\x00\x00\x00\x01\x4e\x2b\x1a\x75\x00\x00\x00\x01\x4e\x2b\x1f\x42\
\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x00"
    pkt.show2()

    #iface="wlan0"
    for i in range(args.c):
        sendp(pkt, iface=iface, verbose=False)
        print("sending on interface %s to %s" % (iface, str(addr)))
    #    sleep(1)
 
#    iface="wlan0"
#    for i in range(args.c):
#        sendp(pkt, iface=iface, verbose=False)
#        print("sending on interface %s to %s" % (iface, str(addr)))
#        sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='receiver parser')
    parser.add_argument('--c', help='number of probe packets',
             type=int, action="store", required=False,
             default=1)
    parser.add_argument('--ip', help='dst ip',
                        type=str, action="store", required=True)
    parser.add_argument('--port', help="dest port", type=int,
                        action="store", required=True)
    parser.add_argument('--l4', help="layer 4 proto (tcp or udp)",
                        type=str, action="store", required=True)
    parser.add_argument('--m', help="message", type=str,
                        action='store', required=False, default="")     
    #parser.add_argument('--itf', help="interface", type=str, action='store', required=True, default="eth0")   
    args = parser.parse_args()
    main(args)
