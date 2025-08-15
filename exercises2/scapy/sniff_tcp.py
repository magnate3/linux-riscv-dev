#!/usr/bin/python3
from scapy.all import *
import sys,re,time
#from socket import htons 
from socket import *
client = '10.10.16.251:8788'
server = '10.10.16.81:7002'

(client_ip, client_port) = client.split(':')
(server_ip, server_port) = server.split(':')
client_sequence_offset=0
relative_offset_seq=0
# Translates Scapy TCP option name to p0f name
TCPOptions = {
    "MSS": "mss",
    "NOP": "nop",
    "WScale": "ws",
    "SAckOK": "sok",
    "SAck": "sack",
    "Timestamp": "ts",
}
def analyze_tcp_option(tcp_options):
    tcp_opt_mss = tcp_options['MSS'] if 'MSS' in tcp_options else '-1'
    tcp_opt_tsval = tcp_options['Timestamp'][0] if 'Timestamp' in tcp_options else '-1'
    tcp_opt_tsecr = tcp_options['Timestamp'][1] if 'Timestamp' in tcp_options else '-1'
    tcp_opt_wscale = tcp_options['WScale'] if 'WScale' in tcp_options else '-1'
    tcp_opt_uto = tcp_options['UserTimeout'] if 'UserTimeout' in tcp_options else '-1'
    tcp_opt_md5header = '0' if 'MD5header' in tcp_options else '-1'
def process_packet(ether_pkt):
    global client_sequence_offset
    global relative_offset_seq
    ip_check = False
    flag =''
    if 'type' not in ether_pkt.fields:
        # LLC frames will have 'len' instead of 'type'.
        # We disregard those
        return
    if ether_pkt.type != 0x0800:
        # disregard non-IPv4 packets
        return
    ip_pkt = ether_pkt[IP]
    if ip_pkt.proto != 6:
        # Ignore non-TCP packet
        return
    tcp_pkt = ip_pkt[TCP]
    if ip_pkt.src == client_ip and ip_pkt.dst == server_ip:
        ip_check = True
    # Determine the TCP payload length. IP fragmentation will mess up this
    # logic, so first check that this is an unfragmented packet
    if (ip_pkt.flags == 'MF') or (ip_pkt.frag != 0):
         print('No support for fragmented IP packets')
         return
    else:
        if ip_check and  tcp_pkt.dport  == int(server_port): 
            flag += str(tcp_pkt.flags) + " | detail: "
            if 'F' in str(tcp_pkt.flags):
                flag +='fin:'    
            if  'S' in str(tcp_pkt.flags):
                flag +='sync:'    
            tcp_payload_len = ip_pkt.len - (ip_pkt.ihl * 4) - (tcp_pkt.dataofs * 4)
            tcp_hdr_len = tcp_pkt.dataofs * 4
            #ip_pkt.show()
            if 'S' in str(tcp_pkt.flags) and 'A' not in str(tcp_pkt.flags): 
                client_sequence_offset = tcp_pkt.seq
            if 'A' in str(tcp_pkt.flags):
                relative_offset_seq = tcp_pkt.seq - client_sequence_offset
            source_origin_win = tcp_pkt.getfieldval("window")
            #tcp_pkt.setfieldval("window", new_win)
            # MSS
            tcp_options = tcp_pkt.getfieldval("options")
            mss_value = 0
            #analyze_tcp_option(tcp_options)
            '''
            if tcp_options:
                if tcp_options[0][0] == "MSS":
                    mss_value = tcp_options[0][1] 
                    #tcp_options[0] = ("MSS", mss_value)
                    #tcp_pkt.setfieldval("options", tcp_options)
                    print("mss value : ",mss_value)
            '''
            tcp_options_dict = dict(tcp_options)
            #mss = tcp_options_dict.get("MSS", 0)
            #scale = tcp_options_dict.get("WScale", 0)
            #olayout_lst = [TCPOptions[field_name] for field_name, value in tcp_options]
            #olayout = ",".join(olayout_lst)
            #print("option : ", tcp_options_dict, "olayout : ", olayout)
            print("tcp option : ", tcp_options_dict)
            print("ip id: ",ip_pkt.id, "ip plen: ",ip_pkt.len,  \
                   "src ip: ", ip_pkt.src, "dst ip: ", ip_pkt.dst, "src port: ", tcp_pkt.sport, "dst port: ", tcp_pkt.dport,\
                   "tcp seq : ", tcp_pkt.seq," --", tcp_pkt.seq + tcp_payload_len , \
                    "tcp flag: ", flag, " || tcp hdr len: ",tcp_hdr_len,\
                   " tcp plen: ", tcp_payload_len)
if __name__ == '__main__':
    ens6_traffic = sniff(iface="enp125s0f0", prn=process_packet)
