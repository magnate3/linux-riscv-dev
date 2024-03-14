#########################################################################################
# IPv6 Fragments: Routing Header before Frag Header
#
# The script is used to generate fragments in which a destination option header is
# added after the IPv6 Fragment Header
#
# The following lines need to be executed only once in the scapy console.
#########################################################################################
sip = '7000::1'
dip = '7000::2'
conf.route6
conf.route6.ifadd ('eth1', '7000::1/64')
conf.route6.add (dst = dip, dev="eth1")

#########################################################################################
# The code below generates the IPv6 Fragments
#########################################################################################
frags = fragment6(IPv6(src=sip, dst=dip) / IPv6ExtHdrFragment() / IPv6ExtHdrDestOpt(options=RouterAlert()) / ICMPv6EchoRequest(data='A'*1600), 1024)
for f in frags:
    packet = Ether(src="00:21:9b:6d:b3:f5",dst="C4:ED:BA:99:FF:01")/f
    sendp(packet, iface="eth1")

