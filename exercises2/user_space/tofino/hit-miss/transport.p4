#ifndef _TRANSPORT_P4_
#define _TRANSPORT_P4_

#include "common_config.p4"

// ---------------------------------------------------------------------------
// Transport parser
// ---------------------------------------------------------------------------

parser TransportParser(packet_in pkt,
		       out header_t hdr,
#ifdef __TARGET_TOFINO__
		       out common_metadata_t meta,
		       out ingress_intrinsic_metadata_t ig_intr_md
#endif
#ifdef __TARGET_PSA__
		       inout common_metadata_t meta,
		       in psa_ingress_parser_input_metadata_t istd,
		       in EMPTY resubmit_meta,
		       in EMPTY recirculate_meta
#endif
#ifdef __TARGET_V1__
		       inout common_metadata_t meta,
		       inout standard_metadata_t stm
#endif
		   ) {
    state start {
#ifdef __TARGET_TOFINO__
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        meta.rx_port = ig_intr_md.ingress_port;
#endif
#ifdef __TARGET_PSA__
        meta.rx_port = istd.ingress_port;
#endif
#ifdef __TARGET_V1__
        meta.rx_port = stm.ingress_port;
#endif
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.eth);
        transition select(hdr.eth.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
//            ETHERTYPE_IPV6: parse_ipv6;
            ETHERTYPE_ARP : parse_arp;
            default: accept;  // TODO: add VLAN and MAC-in-MAC support later
        }
    }

    state parse_arp {
        pkt.extract(hdr.arp);
        transition select(hdr.arp.htype, hdr.arp.ptype) {
            (ARP_HTYPE_ETHERNET, ARP_PTYPE_IPV4) : parse_arp2;
            default : accept;
        }
    }

    state parse_arp2 {
        transition select(hdr.arp.hlen, hdr.arp.plen) {
            (ARP_HLEN_ETHERNET,  ARP_PLEN_IPV4) : parse_arp_ipv4;
            default : accept;
        }
    }

    state parse_arp_ipv4 {
        pkt.extract(hdr.arp_ipv4);
        transition accept;
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }
     
/*    state parse_ipv6 {
        pkt.extract(hdr.ipv6);
        transition accept;
    }*/
}


// ---------------------------------------------------------------------------
// Transport Deparser 
// ---------------------------------------------------------------------------

#ifdef __TARGET_PSA__
control TransportDigestGen(inout header_t hdr,
			   in common_metadata_t meta
			  ) {
    Digest <mac_learn_digest_data>() mac_learn_digest;
    Digest <arp_digest_data>() arp_digest;

    apply {
	if (meta.gen_mac_digest == 1) {
	    mac_learn_digest.pack({hdr.digest_hack.mac_addr, meta.rx_port});
	}
        if (meta.gen_arp_digest == 1) {
	    arp_digest.pack({meta.d32, hdr.eth.dst_addr});
	}
    }
}
#endif

control TransportDeparser(packet_out pkt,
#ifdef __TARGET_TOFINO__
                          inout header_t hdr,
                          in common_metadata_t meta,
                          in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md
#endif
#ifdef __TARGET_PSA__
                          out EMPTY clone_i2e_meta,
                          out EMPTY resubmit_meta,
                          out EMPTY normal_meta,
                          inout header_t hdr,
                          in common_metadata_t meta,
                          in psa_ingress_output_metadata_t istd
#endif
#ifdef __TARGET_V1__
                          in header_t hdr
#endif
                      ) {
#ifdef __TARGET_TOFINO__
    Digest <mac_learn_digest_data>() mac_learn_digest;
    Digest <arp_digest_data>() arp_digest;
#endif

    apply {
#ifdef __TARGET_PSA__
	TransportDigestGen.apply(hdr, meta);
#endif
#ifdef __TARGET_TOFINO__
	if (ig_dprsr_md.digest_type == 1) {
	    mac_learn_digest.pack({hdr.digest_hack.mac_addr, meta.rx_port});
	}
	if (ig_dprsr_md.digest_type == 2) {
	    arp_digest.pack({meta.d32, hdr.eth.dst_addr});
	}
#endif
	pkt.emit(hdr);
    }
}


// ---------------------------------------------------------------------------
// L2 functionality
// ---------------------------------------------------------------------------

control L2_in (inout header_t hdr,
	       inout common_metadata_t meta,
#ifdef __TARGET_TOFINO__
	       inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
	       inout ingress_intrinsic_metadata_for_tm_t ig_tm_md,
#endif
               out ipv4_addr_t _my_ip
              ) {
    mac_addr_t _my_mac = EPG_VIRT_MAC;
#ifndef __TARGET_V1__
    Counter<bit<32>, bit<1>>(1, CounterType_t.PACKETS) mac_error;
#else
    counter(64, CounterType.packets_and_bytes) mac_error;
#endif

/*
#ifndef __TARGET_V1__
    Register<ipv4_addr_t, bit<1>>(1) MyIPReg;
    RegisterAction<ipv4_addr_t, bit<1>, ipv4_addr_t>(MyIPReg) get_ip = {
	void apply(inout ipv4_addr_t data, out ipv4_addr_t result) {
            result = data;
	}
    };
#else
    register<ipv4_addr_t>(1) MyIPReg;
#endif
*/

    action mac_learn_notify() {
        hdr.digest_hack.setValid();
        hdr.digest_hack.mac_addr = hdr.eth.src_addr;
#ifdef __TARGET_TOFINO__ // WORKAROUND - MAC digest will have priority on Tofino
	ig_dprsr_md.digest_type = 1;
#else
	meta.gen_mac_digest = 1;
#endif
    }

    action nop() {}

    action drop() {
#ifdef __TARGET_TOFINO__
	ig_dprsr_md.drop_ctl = 1;
        exit;
#else
        meta.drop = 1;
#endif
    }

    action send(PortId_t p) {
#ifdef __TARGET_TOFINO__
	ig_tm_md.ucast_egress_port = p;
	exit;
#else
	meta.send = 1;
        meta.tx_port = p;
#endif
    }

    table mac_learn {
        key = {
            hdr.eth.src_addr : exact;
        }
	actions = {
	    nop;
	    @defaultonly mac_learn_notify;
	}
        size = 1024;
        default_action = mac_learn_notify;
#ifdef TEST_CONST_ENTRIES
		const entries = {
			0x111111111111 : nop;
		}
#endif
    }

    action arp_reply() {
        hdr.eth.dst_addr = hdr.arp_ipv4.sha;
        hdr.eth.src_addr = _my_mac;
        hdr.arp.oper     = ARP_OPER_REPLY;
        hdr.arp_ipv4.tha = hdr.arp_ipv4.sha;
        hdr.arp_ipv4.tpa = hdr.arp_ipv4.spa;
        hdr.arp_ipv4.sha = _my_mac;
        hdr.arp_ipv4.spa = _my_ip;
    }

    apply {
	_my_ip = STATIC_UPF_ADDR;
        if (!hdr.eth.isValid()) {
            mac_error.count(0);
            drop();
        }
	else {
/*
#ifndef __TARGET_V1__
	    _my_ip = get_ip.execute(0);
#else
            _my_ip = MyIPReg.get(0);
#endif
*/
	    mac_learn.apply();

	    if (hdr.arp.isValid()) {
		if (hdr.arp.oper == ARP_OPER_REQUEST) {
		    if (hdr.arp_ipv4.isValid()) {
			if (hdr.arp_ipv4.tpa == _my_ip) {
			    arp_reply();
			    send(meta.rx_port);
			}
			else drop();
		    }
		    else drop();
		}
		else drop();
	    }
	}
    }
}


control L2_out (inout header_t hdr,
                inout common_metadata_t meta
#ifdef __TARGET_TOFINO__
		,inout ingress_intrinsic_metadata_for_tm_t ig_tm_md
#endif
               ) {
#ifndef __TARGET_V1__
    Counter<bit<32>, bit<1>>(1, CounterType_t.PACKETS) mac_lookup_error;
#else
    counter(64, CounterType.packets_and_bytes) mac_lookup_error;
#endif

    action send(PortId_t p) {
#ifdef __TARGET_TOFINO__
	ig_tm_md.ucast_egress_port = p;
//	exit;
#else
	meta.send = 1;
        meta.tx_port = p;
#endif
    }

    action mac_bcast() { // FIXME: TBD, currently only fix DMAC works
    }

    table mac_forward {
        key = {
            hdr.eth.dst_addr : exact;
        }
        actions = {
            send;
            @defaultonly mac_bcast;
        }
        const default_action = mac_bcast();
        size = 1024;
        
#ifdef TEST_CONST_ENTRIES
        const entries = {
            0x000000000000 : send((PortId_t)0);
#ifdef CONST_MAC
            (CONST_MAC): send(CONST_SEND_PORT);
#endif
        }
#endif

    }

    apply {
        if (!mac_forward.apply().hit) { // L2 FWD (sets egress port)
            mac_lookup_error.count(0);
        }
    }
}


// ---------------------------------------------------------------------------
// L3 functionality
// ---------------------------------------------------------------------------

control Router (inout header_t hdr,
                inout common_metadata_t meta
#ifdef __TARGET_TOFINO__
		,inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md
#endif
               ) {
    ipv4_addr_t _nexthop4 = 0;
    ipv4_addr_t _key = 0;

#ifndef __TARGET_V1__
    Counter<bit<32>, bit<1>>(1, CounterType_t.PACKETS) arp_error;
    Counter<bit<32>, bit<1>>(1, CounterType_t.PACKETS) ip_lookup_error;
#else
    counter(1, CounterType.packets_and_bytes) arp_error;
    counter(1, CounterType.packets_and_bytes) ip_lookup_error;
#endif

    action drop() {
#ifdef __TARGET_TOFINO__
	ig_dprsr_md.drop_ctl = 1;
        exit;
#else
        meta.drop = 1;
#endif
    }

    action set_mac(mac_addr_t dstMac, ether_type_t type) {
        hdr.eth.src_addr = EPG_VIRT_MAC;
        hdr.eth.dst_addr = dstMac;
        hdr.eth.ether_type = type;
    }

    action arp_miss_notify() {
        meta.d32 = _nexthop4;
#ifdef __TARGET_TOFINO__
	ig_dprsr_md.digest_type = 2;
#else
	meta.gen_arp_digest = 1;
#endif
    }

    table arp {
        key = { _nexthop4 : exact; }
        actions = {
            set_mac;
            @defaultonly arp_miss_notify;
        }
        const default_action = arp_miss_notify();
        size = 64;
#ifdef TEST_CONST_ENTRIES
        const entries = {
            0x00000003 : set_mac(0x000000000000, 0x0800);
            0x00000004 : set_mac(0x000000000000, 0x0800);
            0x04000000 : set_mac(0x000000000000, 0x0800);
            0x0000000d : set_mac(0x000000000001, 0x0800);
#ifdef CONST_ARP_IP
            (CONST_ARP_IP): set_mac(CONST_MAC, 1);
#endif
        }
#endif
    }

    action route4(ipv4_addr_t nh) {
        _nexthop4 = nh;
    }

    table ipv4_forward {
        key = {
//            _key : lpm;
            hdr.ipv4.dst_addr : lpm;
        }
        actions = {
            route4;
	}
        size = 10000;
#ifdef TEST_CONST_ENTRIES
        const entries = {
            2 : route4(32w3);
            3 : route4(32w13);
            4 : route4(32w4);
            0xc800_0002 : route4(32w4);
            0x1400_0001 : route4(32w4);
            0x0100_000a : route4(32w5);
#ifdef CONST_ARP_IP
            (CONST_IPV4): route4(CONST_ARP_IP);
#endif
        }
#endif
    }

    apply {
/*	if (hdr.ipv4.isValid()) { _key = hdr.ipv4.dst_addr; }
	else if (hdr.ipv4_x.isValid()) { _key = hdr.ipv4_x.dst_addr; meta.yaf = 1; }
	else { meta.err = 1; }*/

	if (ipv4_forward.apply().miss) { ip_lookup_error.count(0); drop(); }
	else if (arp.apply().miss) { arp_error.count(0); drop(); }
    }
}


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Transport pipeline - only needed if transport is the main P4 functionality
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#ifndef _MAIN_FN_DEFINED_
#define _MAIN_FN_DEFINED_ 1

control TransportIngress(inout header_t hdr,
                         inout common_metadata_t meta,
#ifdef __TARGET_TOFINO__
                         in ingress_intrinsic_metadata_t ig_intr_md,
                         in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
                         inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
                         inout ingress_intrinsic_metadata_for_tm_t ig_tm_md
#endif
#ifdef __TARGET_PSA__
                         in psa_ingress_input_metadata_t istd,
                         inout psa_ingress_output_metadata_t ostd
#endif
#ifdef __TARGET_V1__
                         inout standard_metadata_t stm
#endif
                 ) {
#ifndef __TARGET_V1__
    Counter<bit<64>, bit<1>>(1, CounterType_t.PACKETS_AND_BYTES) rx; // TODO: do a per port ctr later
    Counter<bit<64>, bit<1>>(1, CounterType_t.PACKETS_AND_BYTES) tx;
#else
    counter(64, CounterType.packets_and_bytes) rx;
    counter(64, CounterType.packets_and_bytes) tx;
#endif

    action send(PortId_t port) {
#ifdef __TARGET_TOFINO__
        ig_tm_md.ucast_egress_port = port;
#endif
#ifdef __TARGET_PSA__
        ostd.drop = false;
        ostd.egress_port = port;
#endif
#ifdef __TARGET_V1__
        stm.egress_spec = port;
#endif
    }

    action drop() {
#ifdef __TARGET_TOFINO__
        ig_dprsr_md.drop_ctl = 1;
#endif
#ifdef __TARGET_PSA__
        ostd.drop = true;
#endif
#ifdef __TARGET_V1__
        mark_to_drop(stm);
#endif
    }

    ipv4_addr_t _my_ip = 0;
    apply {
        rx.count(0);
#ifdef __TARGET_TOFINO__
	L2_in.apply(hdr, meta, ig_dprsr_md, ig_tm_md, _my_ip);
#else
	L2_in.apply(hdr, meta, _my_ip);
#endif
#ifdef __TARGET_V1__
	if (meta.gen_mac_digest == 1) {
	    digest<mac_learn_digest>((bit<32>)MAC_LEARN_DIGEST, { hdr.eth.src_addr, meta.rx_port } );
	}
#endif
#ifndef __TARGET_TOFINO__
	if (meta.drop == 1) { drop(); exit; }
	if (meta.send == 1) { send(meta.tx_port); exit; }
#endif

#ifdef __TARGET_TOFINO__
	Router.apply(hdr, meta, ig_dprsr_md);
#else
	Router.apply(hdr, meta);
#endif
#ifdef __TARGET_V1__
	if (meta.gen_arp_digest == 1) {
	    digest<arp_digest>((bit<32>)ARP_MISS_DIGEST, { meta.d32, hdr.eth.dst_addr } );
	}
#endif
#ifndef __TARGET_TOFINO__
	if (meta.drop == 1) { drop(); exit; }
	if (meta.send == 1) { send(meta.tx_port); exit; }
#endif

#ifdef __TARGET_TOFINO__
	L2_out.apply(hdr, meta, ig_tm_md);
#else
	L2_out.apply(hdr, meta);
#endif
#ifndef __TARGET_TOFINO__
	send(meta.tx_port);
#endif
	tx.count(0);
    }
}

#ifdef __TARGET_TOFINO__
Pipeline(TransportParser(),
         TransportIngress(),
         TransportDeparser(),
         CommonEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()
//         TrEgressParser(),
//         TrEgress(),
//         TrEgressDeparser()
        ) t_pipe;
Switch(t_pipe) main;
#endif // TOFINO

#ifdef __TARGET_PSA__
IngressPipeline(TransportParser(),
                TransportIngress(),
                TransportDeparser()) t_ipipe;
EgressPipeline(EmptyEgressParser(),
               EmptyEgress(),
               EmptyEgressDeparser()) t_epipe;
PSA_Switch(t_ipipe, PacketReplicationEngine(), t_epipe, BufferingQueueingEngine()) main;
#endif // PSA

#ifdef __TARGET_V1__
V1Switch(TransportParser(), MyVerifyChecksum(), TransportIngress(), EmptyEgress(), Ipv4ComputeChecksum(), TransportDeparser()) main;
#endif // V1MODEL

#endif // _MAIN_FN_DEFINED_

#endif // _TRANSPORT_P4_
