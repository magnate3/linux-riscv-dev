#import scapy.all as scapy
from scapy.all import *
from defrag6 import *
from common import *
from socket import htons 
#from scapy.all import *
IPV6_SRC = "2001:db8::a0a:6752"
#IPV6_SRC = "2001:db8::a0a:6751"
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
            #pkt.summary()
            frag = pkt.getlayer(IPv6ExtHdrFragment) 
            payload = frag.payload
            if 0x11 == frag.nh:
                print("reassm pkt has udp request, src: ", pkt[IPv6].src, "dst: ", pkt[IPv6].dst,"frag id : ", frag.id, \
                "udp dport: ",  payload[UDP].dport,  "udp sport: ",  payload[UDP].sport, "udp len: ", payload[UDP].len, \
                 "udp  chksum : " , payload[UDP].chksum)
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
    if 0x11 == ip6.nh:
        pass
        #print("reassm pkt has udp request, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst)
    elif 0x2c == ip6.nh:
        frag = packet.getlayer(IPv6ExtHdrFragment) 
        if IPV6_SRC == packet[IPv6].src :
            reasm_packet(packet)
        more = frag.m
        if 0x11  == frag.nh  and more and 0 == (int)(frag.offset):
            payload = frag.payload
            #print("frag pkt has udp request, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst)
            if payload.haslayer(UDP) and IPV6_SRC == packet[IPv6].src:
                print("firt frag has udp request, src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"frag id : ", frag.id, \
                "udp dport: ",  payload[UDP].dport,  "udp sport: ",  payload[UDP].sport, "udp len: ", payload[UDP].len, \
                 "udp  chksum : " , payload[UDP].chksum, "more: ", more, "offset: ", frag.offset, "frag payload : ", len(payload))
        else:
            if not ((int)(frag.offset) & htons(0xFFF9)):
                print("not a fragment fragme")
            else:
                if IPV6_SRC == packet[IPv6].src :
                    print("sub frag , src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst,"frag id : ", frag.id,\
                    "frag next proto: ", frag.nh, "frag offset: ", frag.offset, "more: ", more)
    else:
        pass
        #print("default process,src: ", packet[IPv6].src, "dst: ", packet[IPv6].dst, "ip6 next proto: ",ip6.nh)
if __name__ == '__main__':
    ens6_traffic = sniff(iface="nat64", prn=process_packet)
