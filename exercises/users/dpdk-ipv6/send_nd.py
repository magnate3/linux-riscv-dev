#!/usr/bin/env python

import sys
from optparse import OptionParser
from scapy.all import *

def NS(src_mac, src_addr):
	eth = Ether(dst = "33:33:00:00:00:01", src = src_mac)
	ipv6 = IPv6(dst = "ff02::1", src = src_addr)
	ns = ICMPv6ND_NS(tgt = src_addr)
	lladdr = ICMPv6NDOptSrcLLAddr(lladdr = src_mac)
	pkt = eth/ipv6/ns/lladdr
	pkt.show2()
	sendp(pkt)

def NA(src_mac, src_addr, dst_mac, dst_addr):
	eth = Ether(dst = dst_mac, src = src_mac)
	ipv6 = IPv6(dst = dst_addr, src = src_addr)
	na = ICMPv6ND_NA(tgt = src_addr, R = 0)
	#lladdr = ICMPv6NDOptDstLLAddr(lladdr = src_mac)
	#sendp(eth/ipv6/na/lladdr)
	pkt = eth/ipv6/na
	pkt.show2()
	sendp(pkt)

def RS(src_mac, src_addr):
	eth = Ether(dst = "33:33:00:00:00:02", src = src_mac)
	ipv6 = IPv6(dst = "ff02::2", src = src_addr)
	rs = ICMPv6ND_RS()
	lladdr = ICMPv6NDOptSrcLLAddr(lladdr = src_mac)
	sendp(eth/ipv6/rs/lladdr)

def RA(src_mac, src_addr):
	eth = Ether(dst = "33:33:00:00:00:01", src = src_mac)
	ipv6 = IPv6(dst = "ff02::1", src = src_addr)
	ra = ICMPv6ND_RA(routerlifetime = 1800)
	lladdr = ICMPv6NDOptSrcLLAddr(lladdr = src_mac)
	mtu = ICMPv6NDOptMTU(mtu = 1500)
	pre = ICMPv6NDOptPrefixInfo(prefixlen = 64, validlifetime = 0x80, \
			preferredlifetime = 0x80, prefix = "2000::")
	sendp(eth/ipv6/ra/lladdr/mtu/pre)

def RD(src_mac, src_addr, dst_mac, dst_addr, rtg_addr, rdt_addr):
	eth = Ether(dst = dst_mac, src = src_mac)
	ipv6 = IPv6(dst = dst_addr, src = src_addr)
	rd = ICMPv6ND_Redirect(dst = rdt_addr, tgt = rtg_addr)
	#lladdr = ICMPv6NDOptDstLLAddr(lladdr = src_mac)
	#rdh = ICMPv6NDOptRedirectedHdr()
	#sendp(eth/ipv6/rd/rdh)
	sendp(eth/ipv6/rd)

def main():
	if len(sys.argv) == 1:
		parser.error("incorrect number of arguments")
		sys.exit(1)

	parser = OptionParser()
	parser.add_option("-t", "--type", dest="type", help="Message type: ns, na, rs, ra, rd")
	parser.add_option("-m", "--src_mac", dest="src_mac", help="Source mac address(Route A)")
	parser.add_option("-a", "--src_addr", dest="src_addr", help="Source IPv6 fe80 address(Route A)")
	parser.add_option("-D", "--dst_mac", dest="dst_mac", help="Dest IPv6 MAC address(Host A)")
	parser.add_option("-d", "--dst_addr", dest="dst_addr", help="Dest IPv6 address(Host A)")
	parser.add_option("-g", "--redirect_target", dest="rtg_addr", help="Redirect Gateway IPv6 address(Gateway B)")
	parser.add_option("-r", "--redirect_dest", dest="rdt_addr", help="Redirect Global Dest IPv6 address(Host B)")
	parser.add_option("-i", "--interface", dest="iface", help="Source interface to send")

	(opt, args) = parser.parse_args()

	if opt.iface:
		conf.iface = opt.iface

	if opt.type == "ns":
		NS(opt.src_mac, opt.src_addr)
	elif opt.type == "na":
		NA(opt.src_mac, opt.src_addr, opt.dst_mac, opt.dst_addr)
	elif opt.type == "rs":
		RS(opt.src_mac, opt.src_addr)
	elif opt.type == "ra":
		RA(opt.src_mac, opt.src_addr)
	elif opt.type == "rd":
		RD(opt.src_mac, opt.src_addr, opt.dst_mac, opt.dst_addr, opt.rtg_addr, opt.rdt_addr)
	else:
		print ("unknown type, exit")

if __name__ == '__main__':
	main()
