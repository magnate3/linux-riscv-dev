 #!/usr/local/bin/python2.7
import argparse
from scapy.all import *
from scapy.utils import rdpcap
logging.getLogger('scrapy').setLevel(logging.WARNING)
src_ipv6="2001:db8::a0a:6752"
dst_ipv6="2001:db8::a0a:6751"
node82_ipv6="fe80::4a57:2ff:fe64:e7ae"
node82_mac="48:57:02:64:e7:ae"
node81_mac="48:57:02:64:ea:1e"
ndoe251_mac="44:a1:91:a4:9c:0b"
node81_ipv6="fe80::4a57:2ff:fe64:ea1e"
new_src_mac = node81_mac
#251_mac = "44:A1:91:A4:9C:0B"
new_dst_mac = node82_mac
dst_ip6 = node82_ipv6
src_ip6 = node81_ipv6
def generate_payload(length):
    payload = length * '0'
    return payload
 
def get_l2_l3_length(pkt):
    pkt_layer = pkt
    total_length = len(pkt)
    while pkt_layer.name != 'IP':
        pkt_layer = pkt_layer.payload
    l3_length = len(pkt_layer)
    l2_length = total_length - l3_length
    return l2_length, l3_length
def send_ip6(eth_pkt):
    ip_pkt = eth_pkt.getlayer('IPv6')   
    print("ip packet len: ", ip_pkt.plen)
	#ipv6.src, ipv6.dst
    if 0x2c == ip_pkt.nh:
        frag = eth_pkt.getlayer(IPv6ExtHdrFragment) 
        if 0x3a == frag.nh:
            if 0== frag.offset:
                more = frag.m
                payload = frag.payload
                if payload.haslayer(ICMPv6EchoRequest):
                    icmpv6 = payload.getlayer(ICMPv6EchoRequest)
                    icmp_data= icmpv6.data
                    print("first frag has Echo request, src: ", eth_pkt[IPv6].src, "dst: ", eth_pkt[IPv6].dst,"frag id : ", frag.id,"more: ",more,\
                            "seq: ", icmpv6.seq, "icmp data len: ",len(icmp_data))
                print(str(icmp_data) == 'a'*1440)
            else:
                more = frag.m
                payload = frag.payload
                print("first frag has Echo request, src: ", eth_pkt[IPv6].src, "dst: ", eth_pkt[IPv6].dst,"frag id : ", frag.id,"more :",more,\
                        "payload len: ", len(payload))
                print(payload)
                print(str(payload) == 'a'*92)
                print(str(raw(payload)) == 'a'*92)
    ip_pkt.src = src_ip6
    ip_pkt.dst = dst_ip6
    eth_pkt[Ether].src= new_src_mac  # i.e new_src_mac="00:11:22:33:44:55"
    eth_pkt[Ether].dst= new_dst_mac
    sendp(eth_pkt,verbose=0,iface='enahisic2i3') #sending packet at layer 2
def send_pcap(in_pcap):
    eht_pkt = None
    pkts=rdpcap(in_pcap)  # could be used like this rdpcap("filename",500) fetches first 500 pkts
    for pkt in pkts:
        if not (type(pkt) is  Ether):
            continue
        if pkt.haslayer(Dot1Q):
            #print(pkt[Dot1Q].vlan)
            #continue
            inner_pkt = pkt.getlayer(Dot1Q) 
            eth_pkt = Ether(src=pkt.src, dst=pkt.dst, type=inner_pkt.type)\
                              / inner_pkt.payload
        else:         
            eth_pkt = pkt
        if IPv6 in eth_pkt:
            send_ip6(eth_pkt)
            continue
        ip_pkt = eth_pkt.getlayer('IP')   
        if not ip_pkt:
                continue
        #eth_pkt = pkt
        proto = ip_pkt.fields['proto']
        if proto != 17 and proto != 6:
            print('Ignoring non-UDP/TCP packet,proto %d'%proto)
            continue
        eth_pkt[Ether].src= new_src_mac  # i.e new_src_mac="00:11:22:33:44:55"
        eth_pkt[Ether].dst= new_dst_mac
        l2_length, l3_length = get_l2_l3_length(eth_pkt)
        if (len(eth_pkt) - eth_pkt.len) != l2_length:
            eth_pkt = eth_pkt / generate_payload(eth_pkt.len-l3_length)
        # Delete current checksums
        #if IP in eth_pkt:
        del eth_pkt[IP].chksum
        if UDP in eth_pkt:
            del eth_pkt[UDP].chksum
        if TCP in eth_pkt:
            del eth_pkt[TCP].chksum
        sendp(eth_pkt) #sending packet at layer 2
        '''
        eth_pkt = Ether(pkt)
        if eth_pkt.type != 0x800:
            continue
        ip_pkt= eth_pkt[IP]
        proto = ip_pkt.fields['proto']
        if proto != 17 and proto != 6:
            print('Ignoring non-UDP/TCP packet,proto %d'%proto)
            continue
        eth_pkt[Ether].src= new_src_mac  # i.e new_src_mac="00:11:22:33:44:55"
        eth_pkt[Ether].dst= new_dst_mac
        l2_length, l3_length = get_l2_l3_length(eth_pkt)
        if (len(eth_pkt) - eth_pkt.len) != l2_length:
            eth_pkt = eth_pkt / generate_payload(eth_pkt.len-l3_length)
        # Delete current checksums
        if IP in eth_pkt:
            del eth_pkt[IP].chksum
        if UDP in eth_pkt:
            del eth_pkt[UDP].chksum
        if TCP in eth_pkt:
            del eth_pkt[TCP].chksum
        #pkt[IP].src= new_src_ip # i.e new_src_ip="255.255.255.255"
        #pkt[IP].dst= new_dst_ip
        sendp(eth_pkt) #sending packet at layer 2
        '''
def verify_cksum(ipv6,code1,id1,seq1,payload1):
    #payload1="a"*data_len
    #payload1=packet.Raw("a"*1532) 
    icmpv6=ICMPv6EchoRequest(code=code1, id=id1, seq= seq1,data=payload1)
    pkt = ipv6/icmpv6
    #pkt.show()
    csum=in6_chksum(58, pkt, str(icmpv6))
    return csum 
def verify_cksum_and_send(ipv6,code1,id1,seq1,payload1):
    eth = []
    '''
    set nh=ipv6.nh will cause error
    ipv6_new=IPv6(src=src_ip6, dst=dst_ip6, hlim=ipv6.hlim, fl=ipv6.fl,tc=ipv6.tc,plen=ipv6.plen,nh=ipv6.nh) 
    '''
    ipv6_new=IPv6(src=src_ip6, dst=dst_ip6, hlim=ipv6.hlim, fl=ipv6.fl,tc=ipv6.tc,plen=ipv6.plen) 
    icmpv6=ICMPv6EchoRequest(code=code1, id=id1, seq= seq1,data=payload1)
    pkt = ipv6_new/icmpv6
    #pkt.show()
    csum=in6_chksum(58, pkt, str(icmpv6))
    icmpv6.cksum = csum
    print("icmp6 cksum : ", csum)
    #icmpv6.cksum = 0xffee
    '''
    set error cksum will not get  ICMP6, echo reply
    icmpv6.cksum = 0xffee
    '''
    packets = fragment6(ipv6_new/IPv6ExtHdrFragment(id=random.randint(0, 65535),nh=58)/icmpv6, fragSize=1500)
    for f in packets:
        eth.append(Ether(src=new_src_mac, dst=new_dst_mac,type =ETH_P_IPV6)/f)
    for pkt in eth:
        pkt.show()
    print("begin to send, packet num %d",len(eth))
    sendp(eth,verbose=0,iface='enahisic2i3')
    return csum 
def send_udp_frag():
    eth=Ether(src=node81_mac, dst=ndoe251_mac,type =ETH_P_IPV6)
    ip6 = IPv6(src = src_ipv6, dst = dst_ipv6, nh = 44)
    udp_hdr_len = 8
    udp_data_len = 1508
    len1 = 1440 - udp_hdr_len
    payload1=packet.Raw("a"*len1) 
    payload2=packet.Raw("a"*40) 
    payload3=packet.Raw("a"*36) 
    #payload3=packet.Raw("a"*36 + '\0') 
    udp=UDP(sport=8080, dport=5080, len=udp_data_len + udp_hdr_len, chksum=0x1b3)
    frag_1 = IPv6ExtHdrFragment(nh = 17,id= 0x0000c911, offset = 0, m = 1)
    frag_2 = IPv6ExtHdrFragment(nh = 17,id= 0x0000c911, offset = 1440/8, m = 1)
    frag_3 = IPv6ExtHdrFragment(nh = 17,id= 0x0000c911, offset = 1480/8, m = 0)
    pkt1 = ip6/frag_1/udp/payload1
    sendp(eth/pkt1,verbose=0,iface='enahisic2i3')
    pkt2 = ip6/frag_2/payload2
    sendp(eth/pkt2,verbose=0,iface='enahisic2i3')
    pkt3 = ip6/frag_3/payload3
    sendp(eth/pkt3,verbose=0,iface='enahisic2i3')
def send_udp_frag2():
    eth=Ether(src=node81_mac, dst=ndoe251_mac,type =ETH_P_IPV6)
    udp_hdr_len = 8
    udp_data_len = 140
    ip6 = IPv6(src = src_ipv6, dst = dst_ipv6, plen=udp_data_len + udp_hdr_len, nh = 17)
    payload1=packet.Raw("a"*udp_data_len) 
    udp=UDP(sport=8080, dport=5080, len=udp_data_len + udp_hdr_len, chksum=0x0)
    pkt1 = ip6/udp/payload1
    packet_raw = raw(pkt1)
    udp_raw = packet_raw[40:]
    udp.chksum = 0
    csum = in6_chksum(socket.IPPROTO_UDP, ip6,udp_raw)
    pkt1[UDP].chksum = csum
    print("csum: 0x%x,origin_cksum : 0x%x" %(csum,udp.chksum)) 
    sendp(eth/pkt1,verbose=0,iface='enahisic2i3')
def send_udp_frag3():
    eth=Ether(src=node81_mac, dst=ndoe251_mac,type =ETH_P_IPV6)
    ip6 = IPv6(src = src_ipv6, dst = dst_ipv6, nh = 44)
    udp_hdr_len = 8
    udp_data_len = 2048
    #udp_data_len = 1488
    total = udp_hdr_len + udp_data_len
    sec_of  = (total/2)&~7
    len1 = sec_of - udp_hdr_len
    len2 =  total - sec_of
    fisrt_of = 0
    payload1=packet.Raw("a"*len1) 
    payload2=packet.Raw("a"*len2) 
    udp=UDP(sport=8080, dport=5080, len=udp_data_len + udp_hdr_len, chksum=0x0)
    total_payload = packet.Raw("a"*udp_data_len)
    packet_raw = raw(ip6/udp/total_payload)
    udp_raw = packet_raw[40:]
    udp.chksum = 0
    csum = in6_chksum(socket.IPPROTO_UDP, ip6,udp_raw)
    frag_1 = IPv6ExtHdrFragment(nh = 17,id= 0x0000c911, offset = 0, m = 1)
    frag_2 = IPv6ExtHdrFragment(nh = 17,id= 0x0000c911, offset = sec_of/8, m = 0)
    udp.chksum = csum
    pkt1 = ip6/frag_1/udp/payload1
    sendp(eth/pkt1,verbose=0,iface='enahisic2i3')
    pkt2 = ip6/frag_2/payload2
    sendp(eth/pkt2,verbose=0,iface='enahisic2i3')
def check_pcap_and_send(in_pcap):
    arr = []
    eth=Ether(src=node81_mac, dst=ndoe251_mac,type =ETH_P_IPV6)
    pkts=rdpcap(in_pcap)  # could be used like this rdpcap("filename",500) fetches first 500 pkts
    for pkt in pkts:
        if src_ipv6 == pkt[IPv6].src:
            sendp(eth/pkt,verbose=0,iface='enahisic2i3')
            arr.append(pkt)
    reasm_pkt = defragment6(arr)
    reasm_pkt.show()
    ipv6 = reasm_pkt.getlayer(IPv6)
    udp_hdr = reasm_pkt[UDP]
    origin_cksum = reasm_pkt[UDP].chksum
    print("udp sport: %d,dport : %d" %(udp_hdr.sport ,udp_hdr.dport)) 
    reasm_pkt[UDP].chksum = 0
    packet_raw = raw(reasm_pkt)
    udp_raw = packet_raw[40:]
    csum = in6_chksum(socket.IPPROTO_UDP, ipv6,udp_raw)
    print("csum: 0x%x,origin_cksum : 0x%x" %(csum,origin_cksum)) 
    #sendp(eth/reasm_pkt,verbose=0,iface='enahisic2i3')
    '''
    icmpv6 = reasm_pkt.getlayer(ICMPv6EchoRequest)
    icmp_data= icmpv6.data
    data_len = len(icmp_data)
    #len(ipv6.payload) = sizeof(icmp_hdr) + data_len
    print("data len: ", data_len, "ipv6 payload len: ", len(ipv6.payload))
    origin_cksum = icmpv6.cksum
    ipv6_cpy =IPv6(src=ipv6.src, dst=ipv6.dst, hlim=ipv6.hlim, fl=ipv6.fl,tc=ipv6.tc,plen=ipv6.plen,nh=ipv6.nh) 
    csum = verify_cksum(ipv6_cpy,icmpv6.code,icmpv6.id,icmpv6.seq,icmp_data)
    print("csum: %d,origin_cksum : %d" %(csum,origin_cksum)) 
    verify_cksum_and_send(ipv6,icmpv6.code,icmpv6.id,icmpv6.seq,icmp_data)
    '''
#------o--------------------------------------------
def command_line_args():
    """Helper called from main() to parse the command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcap', metavar='<input pcap file>',
                        help='pcap file to parse', required=True)
    #parser.add_argument('--csv', metavar='<output csv file>',
    #                    help='csv file to create', required=True)
    args = parser.parse_args()
    return args
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

#--------------------------------------------------
def main():
    """Program main entry"""
    args = command_line_args()

    if not os.path.exists(args.pcap):
        print('Input pcap file "{}" does not exist'.format(args.pcap))
        sys.exit(-1)

    """
    if os.path.exists(args.csv):
        print('Output csv file "{}" already exists, '
              'won\'t overwrite'.format(args.csv),
              file=sys.stderr)
        sys.exit(-1)
    for name in findAllFile(args.pcap):
        path = os.path.join(args.pcap, name)
        print(path)
        send_pcap(path)
    """
    print(args.pcap)
    if not os.path.exists(args.pcap):
        print('path not exist: '.format(args.pcap))
        sys.exit(-1)
    #check_pcap_and_send(args.pcap)
    #send_udp_frag()
    #send_udp_frag2()
    send_udp_frag3()
#--------------------------------------------------

if __name__ == '__main__':
    main()
