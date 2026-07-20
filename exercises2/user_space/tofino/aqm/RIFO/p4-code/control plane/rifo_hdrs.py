from scapy.all import *
TYPE_IPV4 = 0x0800
class rifo(Packet):
   fields_desc = [ BitField('rank',0,16)]


bind_layers(Ether, IP, type= TYPE_IPV4)
bind_layers(IP, TCP)
bind_layers(IP, TCP)
bind_layers(TCP, rifo)
bind_layers(UDP, rifo)