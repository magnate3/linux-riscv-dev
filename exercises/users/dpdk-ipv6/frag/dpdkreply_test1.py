 #!/usr/local/bin/python2.7
import argparse
from scapy.all import *
from scapy.utils import rdpcap
logging.getLogger('scrapy').setLevel(logging.WARNING)
node82_ipv6="fe80::4a57:2ff:fe64:e7ae"
node82_mac="48:57:02:64:e7:ae"
ndoe81_mac="48:57:02:64:ea:1e"
node81_ipv6="fe80::4a57:2ff:fe64:ea1e"
new_src_mac = ndoe81_mac
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
def check_pcap_and_send(in_pcap):
    arr = []
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
            arr.append(eth_pkt[IPv6])
    reasm_pkt = defragment6(arr)
    reasm_pkt.show()
    ipv6 = reasm_pkt.getlayer(IPv6)
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
#--------------------------------------------------
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
    #send_pcap(args.pcap)
    check_pcap_and_send(args.pcap)
#--------------------------------------------------

if __name__ == '__main__':
    main()
