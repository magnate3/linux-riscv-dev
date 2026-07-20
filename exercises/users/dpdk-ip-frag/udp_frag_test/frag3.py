#!/usr/bin/python

from scapy.all import *
sip="10.10.103.81"
dip="100.20.0.82"
#dip="10.10.103.229"
payload="A"*496+"B"*500 +  "d"*300
packet=IP(src=sip,dst=dip,id=12345)/UDP(sport=1500,dport=4444)/payload

frags=fragment(packet,fragsize=500)
counter=1
for fragment in frags:
    print "Packet no#"+str(counter)
    print "==================================================="
    fragment.show() #displays each fragment
    counter+=1
    send(fragment)
