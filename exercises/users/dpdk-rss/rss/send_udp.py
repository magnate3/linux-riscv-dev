#!/usr/bin/python3
import logging
import argparse
from scapy.all import *
from scapy.utils import rdpcap
import random
import time
import signal
import ipaddress, random, struct
new_src_mac = "48:57:02:64:ea:1e"
#new_src_mac = "b0:08:75:5f:b8:5b"
new_dst_mac = "44:a1:91:a4:9c:0b"
ethname='eno2'
def random_ip(network):
    network = ipaddress.IPv4Network(network)
    network_int, = struct.unpack("!I", network.network_address.packed) # make network address into an integer
    rand_bits = network.max_prefixlen - network.prefixlen # calculate the needed bits for the host part
    rand_host_int = random.randint(0, 2**rand_bits - 1) # generate random host part
    ip_address = ipaddress.IPv4Address(network_int + rand_host_int) # combine the parts 
    return ip_address.exploded
def test(network):
    print("send udp pkt")
    ip = random_ip(network)
    print(ip)
    ip2 = random_ip(network)
    payload = 'a' * 100
    src_port = random.randint(2000, 8000)
    packet = IP(dst=ip,src=ip2,id=12345)/UDP(dport=5000,sport=src_port)/payload
    frags=fragment(packet,fragsize=500) 
    for frag in frags:
        pkt = Ether(src= new_src_mac,dst= new_dst_mac,type=0x800)/frag
        sendp(pkt,verbose=0,iface=ethname) #sending packet at layer 2
    packet = IP(dst=ip,src=ip2,id=12345)/TCP(dport=5000,sport=src_port)/payload
    frags=fragment(packet,fragsize=500) 
    for frag in frags:
        pkt = Ether(src= new_src_mac,dst= new_dst_mac,type=0x800)/frag
        sendp(pkt,verbose=0,iface=ethname) #sending packet at layer 2
def main():
    print("send udp pkt")
    payload = 'a' * 100
    packet = IP(dst="10.11.11.65",src="10.11.11.66",id=12345)/UDP(dport=5000,sport=3333)/payload
    frags=fragment(packet,fragsize=500) 
    for frag in frags:
        pkt = Ether(src= new_src_mac,dst= new_dst_mac,type=0x800)/frag
        sendp(pkt,verbose=0,iface=ethname) #sending packet at layer 2
if __name__ == '__main__':
    main()
    #'''
    for _ in range(256):
       test(u"10.10.103.0/24")
       test(u"181.10.103.0/24")
       test(u"103.112.164.0/22")
       test(u"103.112.164.0/255.255.252.0")
    #'''
