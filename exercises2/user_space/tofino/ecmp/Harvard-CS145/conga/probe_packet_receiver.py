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
queue length value. This script is the probe packet receiver.

Description:
This script is the `probe` packet receiver which receives and parses the
probe packets sent by the script probe_sender.py.
It parses the telemetry header in the probe packet, and get the queue length
information from the telemetry header which is assigned by the P4 switches in
the network.

Run method:
probe_receiver.py
    * No argument needed

"""

import sys
import os

from scapy.all import sniff, get_if_list, Ether, get_if_hwaddr, IP, Raw, Packet, BitField, bind_layers

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

    iface=None
    for i in get_if_list():
        if "eth0" in i:
            iface=i
            break;
    if not iface:
        print ("Cannot find eth0 interface")
        exit(1)
    return iface

class Telemetry(Packet):
    """
    The telemetry header class in the probe packets

    ...

    Attributes
    ----------
    fields_desc : list
        the list of fields in the telemetry header, 
        including enq_depth (the maximum queue length),
        and the nextHeaderType (the next header type to parse)
    """

    fields_desc = [ BitField("enq_depth", 0, 16),
                   #BitField("deq_depth", 0, 16),
                   BitField("nextHeaderType", 0, 16)]

def isNotOutgoing(my_mac):
    """
    This function returns a filter function for the packet
    sniffer, so the sniffer only needs to process incoming
    packets.

    Parameters
    ----------
    my_mac : str
        The MAC address string to compare with, so we can filter
        out packets that sent from our interface

    Return
    ------
    A function takes as input a scapy.packet class and returns
    whether the packet's source MAC address is the same as the 
    MAC address of our interface
    """

    my_mac = my_mac
    def _isNotOutgoing(pkt):
        return pkt[Ether].src != my_mac

    return _isNotOutgoing

def handle_pkt(pkt):
    """
    Parse the probe packets and print out the maximum queue length

    Parameters
    ----------
    pkt : scapy.packet
        The Scapy packet class containing all the header and payload
        information of a packet
    """

    ether = pkt.getlayer(Ether)

    telemetry = pkt.getlayer(Telemetry)
    print ("Queue Info:")
    print ("enq_depth", telemetry.enq_depth)
    #print ("deq_depth", telemetry.deq_depth)
    print

bind_layers(Ether, Telemetry, type=0x7777)


def main():
    """
    Get the first interface of this node, and sniff the incoming packets
    with Ethernet header and a telemetry header with ID 0x7777, print the 
    queuing length from the telemetry header for each packet
    """
    
    ifaces = list(filter(lambda i: 'eth' in i, os.listdir('/sys/class/net/')))
    iface = ifaces[0]
    print ("sniffing on %s" % iface)
    sys.stdout.flush()

    my_filter = isNotOutgoing(get_if_hwaddr(get_if()))

    sniff(filter="ether proto 0x7777", iface = iface,
          prn = lambda x: handle_pkt(x), lfilter=my_filter)

if __name__ == '__main__':
    main()
