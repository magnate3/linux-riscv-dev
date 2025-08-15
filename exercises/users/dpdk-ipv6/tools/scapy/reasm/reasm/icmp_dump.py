#import scapy.all as scapy
from scapy.all import *
from defrag6 import *
from common import *
#from scapy.all import *
def IPv6Packet(ipv6):
    icmpv6 = ipv6.getlayer(ICMPv6EchoRequest)
    # We only support ping for now.
    if (ipv6.nh != IPPROTO_ICMPV6 or not icmpv6 or icmpv6.type != ICMPV6_ECHO_REQUEST or
        icmpv6.code != 0):
        return
def reasm_packet(pkt):
    if IPv6ExtHdrFragment in pkt:
        temp_pkt = None
        defrag = Frag6Reassembly()
        temp_pkt = defrag.Frag6Defrag(pkt)
        if temp_pkt == None:
            return
        else:
            pkt = temp_pkt
            pkt.summary()
            frag = pkt.getlayer(IPv6ExtHdrFragment) 
            #if 0x3a == pkt.nh:
            if 0x3a == frag.nh:
                #print(pkt)
                #print("reasm pkt has Echo request, src: ", pkt[IPv6].src, "dst: ", pkt[IPv6].dst)
                #print("reasm pkt has Echo request, src: ", pkt[IPv6].src, "dst: ", pkt[IPv6].dst, "icmp6 type :", pkt['ICMPv6'])
                payload = pkt.payload
                #icmp6 = ICMPv6(payload)
                #if 129 ==  payload.type:
                #if pkt.haslayer(ICMPv6EchoRequest):
                #isinstance(self.underlayer, _IPv6ExtHdr)
                #if isinstance(pkt.underlayer, ICMPv6EchoRequest):
                #if isinstance(pkt.underlayer, IPv6ExtHdrFragment):
                if payload.haslayer(ICMPv6EchoRequest):
                    #print("reasm pkt has Echo request, src: ", pkt[IPv6].src, "dst: ", pkt[IPv6].dst)
                    icmpv6 = payload.getlayer(ICMPv6EchoRequest)
                    print("reassm pkt has Echo request, src: ", pkt[IPv6].src, "dst: ", pkt[IPv6].dst,"frag id : ", frag.id,"seq: ", icmpv6.seq)
def process_packet(packet):
    packet.summary()
    #packet.show2()
    #destination_mac_address = packet[0][scapy.Ether].dst
    #destination_ip_address = packet[0][scapy.IP].dst
    #print("Received a packet with a destination MAC address of %s and a destination IP address of %s" % (destination_mac_address,destination_ip_address))
    if IPv6 not in packet:
        return
    '''
    if ICMPv6 in packet:
        if packet['ICMPv6']['type'] == 129:
            print("Echo request, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst)
    '''
    ip6 =  packet.getlayer(IPv6)
    if packet.haslayer(ICMPv6NDOptMTU):
        print("mtu, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst)
    if packet.haslayer(ICMPv6PacketTooBig):
        print("too big, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst)
    if 0x3a == ip6.nh:
        if packet.haslayer(ICMPv6EchoRequest):
            icmpv6 = packet.getlayer(ICMPv6EchoRequest)
            print("Echo request, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"seq: ", icmpv6.seq)
        elif packet.haslayer(ICMPv6EchoReply):
            icmpv6 = packet.getlayer(ICMPv6EchoReply)
            print("Echo reply , src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"seq: ", icmpv6.seq)
        elif packet.haslayer(ICMPv6ND_NA):
            icmpv6 = packet.getlayer(ICMPv6ND_NA)
            print("advert, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"type: ", packet[ICMPv6ND_NA].type, "tgt: ",  packet[ICMPv6ND_NA].tgt)
        elif packet.haslayer(ICMPv6ND_NS):
            icmpv6 = packet.getlayer(ICMPv6ND_NA)
            print("solicit, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"type: ", packet[ICMPv6ND_NS].type, "tgt: ",  packet[ICMPv6ND_NS].tgt)
    #if packet.haslayer(IPv6ExtHdrFragment):
    elif 0x2c == ip6.nh:
        frag = packet.getlayer(IPv6ExtHdrFragment) 
        reasm_packet(packet)
        if 0x3a == frag.nh  and  0 == frag.offset:
            more = frag.m
            payload = frag.payload
            if payload.haslayer(ICMPv6EchoRequest):
                icmpv6 = packet.getlayer(ICMPv6EchoRequest)
                print("first frag has Echo request, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"frag id : ", frag.id,"seq: ", icmpv6.seq)
            elif packet.haslayer(ICMPv6EchoReply):
                icmpv6 = packet.getlayer(ICMPv6EchoReply)
                print("first frag has Echo reply , src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"seq: ", icmpv6.seq)
        else:
            print("sub frag , src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"frag id : ", frag.id,"frag next proto: ", frag.nh, "frag offset: ", frag.offset)
    else:
        print("default process,src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst, "ip6 next proto: ",ip6.nh)
        #print("ip6 next proto:%d ",ip6.nh)
    # IPPROTO_ICMPV6
    #print("ipv6 packet nh %d"%packet[IPv6].nh)
    #if packet[IPv6].nh == 58:
        '''
        print("ipv6 packet nh %d"%packet[IPv6].type)
        if packet[IPv6].type == 128:
            print("Echo request, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst)
            #duplicate_tracker[packet[IPv6].src] += 1
            #pass
        elif packet[IPv6].type == 129:
            print("Echo reply, src: ", packet[IPv6].src)
            #duplicate_tracker[packet[IPv6].src] += 1
        elif packet[IPv6].type == 1:
            print("Destination unreachable, src: ", packet[IPv6].src)
            if packet[IPv6].code == 0:
                print("No route.")
            elif packet[IPv6].code == 1:
                #print("Admin prohibited.")
                print("Intended target: ", (packet[ICMPv6DestUnreach].payload).dst)
            elif packet[IPv6].code == 2:
                print("Beyond scope of source address.")
            elif packet[IPv6].code == 3:
                print("Address unreachable.")
            elif packet[IPv6].code == 4:
                print("Port unreachable.")
            elif packet[IPv6].code == 5:
                print("Source address failed ingress/egress policy.")
            elif packet[IPv6].code == 6:
                print("Reject route to destination.")
            elif packet[IPv6].code == 7:
                print("Error in source routing header.")
    
        elif packet[IPv6].type == 3:
           overall_stats["Time exceeded"] += 1
           #print("Time exceeded, src: ", packet[IPv6].src)
        '''
if __name__ == '__main__':
    ens6_traffic = sniff(iface="enahisic2i0", prn=process_packet)
