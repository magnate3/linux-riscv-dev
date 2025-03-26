#!/usr/bin/env python

"""

Project role:
In project 6, we want to let the P4 switches create a telemetry header and
fill in the maximum queue length (enq_depth) through the path to the packet.
The queue length value indicates the congestion within the network.
To detect whether we successully implement the telemetry header in the network,
here we use a sender to create and send probe packets to a remote host, and use 
a receiver at the remote host to receive those probe packets. Then the receiver 
checks whether it can parse the telemetry header for probe packets and get a valid
queue length value. This script is the probe packet sender.

Description:
This script is to send some `probe` packets through the Mininet network
to get the queue length (enq_depth in P4) information. It creates and sends a probe
packet per 0.1 second, and the probe packet is marked by the destination port number
7777.

Run method:
probe_sender.py {dest_ip} {num_packets}
    * dest_ip: the IP address of the node running probe_receiver.py
    * num_packets: the number of probe packets to send

"""

import argparse
import sys
import socket
import random
import struct

from scapy.all import sendp, get_if_list, get_if_hwaddr
from scapy.all import Ether, IP, UDP, TCP
import time

def get_if():   
    """
    Get the interface eth0

    Errors
    ------
    If no interface has a name with "eth0",
    Exit the program with error code 1

    Return
    ------
    The interface name containing "eth0" substring
    """

    ifs=get_if_list()
    iface=None # "h1-eth0"
    for i in get_if_list():
        if "eth0" in i:
            iface=i
            break;
    if not iface:
        print ("Cannot find eth0 interface")
        exit(1)
    return iface

def main():
    """
    Parse destination host name and number of packets to send,
    and then send probe packets via interface hX-eth0 to the destination
    host, with destination TCP port 7777.
    Send one probe packet per 0.1 second.
    """

    if len(sys.argv)<3:
        print ('pass 2 arguments: <destination> <number_of_random_packets>')
        exit(1)

    num_packets_to_send = int(sys.argv[2])
    if num_packets_to_send < 0 or num_packets_to_send > 10000:
        print ("The number of packets sent should be between [0, 10000]")
        exit(1)

    addr = socket.gethostbyname(sys.argv[1])
    iface = get_if()

    print ("sending on interface %s to %s" % (iface, str(addr)))

    for _ in range(int(sys.argv[2])):
        pkt = Ether(src=get_if_hwaddr(iface), dst='ff:ff:ff:ff:ff:ff')
        pkt = pkt /IP(dst=addr) / UDP(dport=7777, sport=random.randint(2000,65535))
        # pkt = pkt /IP(dst=addr) / TCP(dport=7777, sport=random.randint(2000,65535))
        sendp(pkt, iface=iface, verbose=False)
        time.sleep(0.1)

if __name__ == '__main__':
    main()
