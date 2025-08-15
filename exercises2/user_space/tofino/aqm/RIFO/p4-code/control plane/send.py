#!/usr/bin/env python
import sys
import time
from rifo_hdrs import *


def main():
    iface = sys.argv[1]
    pkt = Ether(src=get_if_hwaddr(iface), dst='ff:ff:ff:ff:ff:ff')/ IP(dst="192.168.45.20")/UDP(dport=9000)/ rifo(rank=100) 
    pkt.show2()  
    c = 1
    while c<=6000:
        try:
            sendp(probe_pkt , iface=iface)
            #time.sleep(0.5)
            c = c + 1
        except KeyboardInterrupt:
            sys.exit()

if __name__ == '__main__':
    main()
