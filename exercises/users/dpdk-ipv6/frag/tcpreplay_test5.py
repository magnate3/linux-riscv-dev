#!/usr/local/bin/python2.7

print "ping6 fragment that overlaps the last fragment with the tail"

#          |----|
#      |XXXXXXXX|
# |--------|

import os
#from addr import *
from scapy.all import *

SRC_IF =  'enahisic2i0'
SRC_MAC = "48:57:02:64:ea:1b"
#DST_MAC = "48:57:02:64:e7:ab"
DST_MAC = "b0:08:75:5f:b8:5b"
DST_IN6 = "fec0::1:323:45ff:fe67:8902"
SRC_OUT6 = "fec0::1:323:45ff:fe67:8901"
#DST_IN6 = "fe80::b208:75ff:fe5f:b85b"
##DST_IN6 = "fec0::1:323:45ff:fe67:8902"
##SRC_OUT6 = "fec0::1:323:45ff:fe67:8901"
#SRC_OUT6 = "fe80::4a57:2ff:fe64:ea1b"
conf.route6
conf.route6.ifadd (SRC_IF, 'fe80::997e:ea4a:6f5d:f076/64')
conf.route6.add (dst = DST_IN6, dev=SRC_IF)
def process_packet(packet):
        if IPv6 not in packet:
                exit(1)
        ip6 =  packet.getlayer(IPv6)
        #if  packet and packet.type == ETH_P_IPV6 and \
        if 0x3a == ip6.nh:
                if packet.haslayer(ICMPv6EchoRequest):
                        icmpv6 = packet.getlayer(ICMPv6EchoRequest)
                        print("Echo request, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"seq: ", icmpv6.seq) 
        elif 0x2c == ip6.nh:
            frag1 = packet.getlayer(IPv6ExtHdrFragment)
            if(None == frag1):
                    print "frag is none"
                    exit(2)
            if 0x3a == frag1.nh  and  0 == frag1.offset:
                    payload = frag1.payload
                    if payload.haslayer(ICMPv6EchoRequest):
                        id=packet.payload.payload.id
                        print "id=%#x" % (id)
                        if id != eid:
                                print "WRONG ECHO REPLY ID"
                                exit(2)
                        data=packet.payload.payload.data
                        print "payload=%s" % (data)
                        if data == payload:
                                print "ECHO REPLY"
                        else:
                            print "PAYLOAD!=%s" % (payload)
        else:
            print "no echo reply"
        exit(0)
def send_big_icmp6_v2():
        dpass
def send_big_icmp6():
        pid=os.getpid()
        eid=pid & 0xffff
        eth=[]
        # fragSize = 1500 ,fragSize = 1400
        packets = fragment6(IPv6(src=SRC_OUT6, dst=DST_IN6, hlim=63, fl=0)/IPv6ExtHdrFragment(id=random.randint(0, 65535),nh=58)/ ICMPv6EchoRequest(id=eid,seq=0x865,data='A'*2500), fragSize=1424)
        #packets = fragment6(Ether(src=SRC_MAC, dst=DST_MAC,type =ETH_P_IPV6)/IPv6(src=SRC_OUT6, dst=DST_IN6, hlim=63, fl=0)/IPv6ExtHdrFragment(id=random.randint(0, 65535),nh=58)/ ICMPv6EchoRequest(id=eid,seq=0x8765,data='A'*2500), fragSize=1024)
        for f in packets:
            eth.append(Ether(src=SRC_MAC, dst=DST_MAC,type =ETH_P_IPV6)/f)
            #sendp(f, iface=SRC_IF)
        sendp(eth, iface=SRC_IF)
        time.sleep(10)
def send_big_icmp6_v3():
        pid=os.getpid()
        eid=pid & 0xffff
        eth=[]
        ipv6 = IPv6(src=SRC_OUT6, dst=DST_IN6, hlim=63, fl=0)
        icmpv6 = ICMPv6EchoRequest(id=eid,seq=0x865,data='A'*2500)
        csum=scapy.layers.inet6.in6_chksum(58, ipv6/icmpv6, str(icmpv6))
        print("csum: ", csum)
        packets = fragment6(ipv6/IPv6ExtHdrFragment(id=random.randint(0, 65535),nh=58)/icmpv6, fragSize=1424)
        for f in packets:
            eth.append(Ether(src=SRC_MAC, dst=DST_MAC,type =ETH_P_IPV6)/f)
        sendp(eth, iface=SRC_IF)
        time.sleep(10)
def send_icmp6():
        pid=os.getpid()
        eid=pid & 0xffff
        packets2 = Ether(src=SRC_MAC, dst=DST_MAC,type =ETH_P_IPV6)/IPv6(src=SRC_OUT6, dst=DST_IN6) / ICMPv6EchoRequest(id=eid,data='A'*500)
        sendp(packets2, iface=SRC_IF)
def send_frag(interface,mac_source,sip,dip,mac_dst):
    myid=random.randrange(1,4294967296,1)  #generate a random fragmentation id 
    payload1=packet.Raw("AABBCCDD") 
    #payload2=packet.Raw("AABBCCDD"*2) 
    #icmpv6=ICMPv6EchoRequest(data=payload2) 
    icmpv6=ICMPv6EchoRequest(data=payload1) 
    ipv6_1=IPv6(src=sip, dst=dip, plen=24) 
    ipv6_2=IPv6(src=sip, dst=dip, plen=16) 
    csum=scapy.layers.inet6.in6_chksum(58, ipv6_1/icmpv6, str(icmpv6))
    icmpv6=ICMPv6EchoRequest(cksum=csum, data=payload1)
    frag1=IPv6ExtHdrFragment(offset=0, m=1, id=myid)
    frag2=IPv6ExtHdrFragment(offset=1, m=0, id=myid)
    packet1=ipv6_1/frag1/icmpv6
    packet2=ipv6_2/frag2/payload1
    layer2=Ether(src=mac_source,dst=mac_dst)
    sendp(layer2/packet2,iface=interface)
    sendp(layer2/packet1,iface=interface)
def send_frag_v2(interface,mac_source,sip,dip,mac_dst):
    ip6 = IPv6(src = sip, dst = dip, nh = 44)
    frag_1 = IPv6ExtHdrFragment(nh = 60, m = 1)
    dst_opt = IPv6ExtHdrDestOpt(nh = 58)
    frag_2 = IPv6ExtHdrFragment(nh = 58, offset = 4, m = 1)
    icmp_echo = ICMPv6EchoRequest(seq = 1)
    packet1= ip6/frag_1/dst_opt
    packet2= ip6/frag_2/icmp_echo
    layer2=Ether(src=mac_source,dst=mac_dst)
    sendp(layer2/packet2,iface=interface)
    sendp(layer2/packet1,iface=interface)
if __name__ == '__main__':
    #send_frag_v2(SRC_IF,SRC_MAC,SRC_OUT6,DST_IN6,DST_MAC)
    #send_frag(SRC_IF,SRC_MAC,SRC_OUT6,DST_IN6,DST_MAC)
    #send_big_icmp6()
    send_big_icmp6_v3()
    #send_icmp6()
