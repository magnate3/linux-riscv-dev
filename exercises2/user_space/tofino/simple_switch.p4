/*******************************************************************************
 * BAREFOOT NETWORKS CONFIDENTIAL & PROPRIETARY
 *
 * Copyright (c) 2019-present Barefoot Networks, Inc.
 *
 * All Rights Reserved.
 *
 * NOTICE: All information contained herein is, and remains the property of
 * Barefoot Networks, Inc. and its suppliers, if any. The intellectual and
 * technical concepts contained herein are proprietary to Barefoot Networks, Inc.
 * and its suppliers and may be covered by U.S. and Foreign Patents, patents in
 * process, and are protected by trade secret or copyright law.  Dissemination of
 * this information or reproduction of this material is strictly forbidden unless
 * prior written permission is obtained from Barefoot Networks, Inc.
 *
 * No warranty, explicit or implicit is provided, unless granted under a written
 * agreement with Barefoot Networks, Inc.
 *
 ******************************************************************************/

#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#include "common/headers.p4"
#include "common/util.p4"


struct partition_t {
    bit<10> partition_index;
}

struct metadata_t {
    partition_t partition;
	pkt_custom_type_t  pkt_custom_type;
    internal_h internal_hdr;
    bridge_h   bridge_hdr;
	bit<8>     bypass_flag;
}

struct pair {
    bit<32>     first;
    bit<32>     second;
}

// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------
parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    TofinoIngressParser() tofino_parser;

    state start {
        tofino_parser.apply(pkt, ig_intr_md);
        //transition parse_ethernet;
		transition accept;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition accept;
    }
	
	/*
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select (hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }*/

}


// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------
control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {

    Checksum() ipv4_checksum;
    apply {
        hdr.ipv4.hdr_checksum = ipv4_checksum.update({
            hdr.ipv4.version,
            hdr.ipv4.ihl,
            hdr.ipv4.diffserv,
            hdr.ipv4.total_len,
            hdr.ipv4.identification,
            hdr.ipv4.flags,
            hdr.ipv4.frag_offset,
            hdr.ipv4.ttl,
            hdr.ipv4.protocol,
            hdr.ipv4.src_addr,
            hdr.ipv4.dst_addr});
        //pkt.emit(hdr);
        pkt.emit(ig_md.internal_hdr);
        pkt.emit(ig_md.bridge_hdr);
		
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.custom_metadata1);	
        pkt.emit(hdr.custom_metadata2);	
        pkt.emit(hdr.custom_metadata3);			
    }
}



control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

    action fw_to(PortId_t port, bypass_flag_t bypass_flag) {
        ig_intr_tm_md.ucast_egress_port = port;
        ig_intr_dprsr_md.drop_ctl = 0x0;
        ig_md.bypass_flag = bypass_flag;		
    }
	
    table port {
        key = {
            ig_intr_md.ingress_port : ternary;
        }

        actions = {
            fw_to;
        }
		
        size = 1024;
    }
	
    apply {
		

        port.apply();

        ig_intr_tm_md.bypass_egress = 1w1;	
    }
}

// ---------------------------------------------------------------------------
// Egress Parser
// ---------------------------------------------------------------------------
parser SwitchEgressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {

    TofinoEgressParser() tofino_parser;

    state start {
        tofino_parser.apply(pkt, eg_intr_md);
        transition parse_internal_hdr;
    }

    state parse_internal_hdr {
        pkt.extract(eg_md.internal_hdr);
        eg_md.bridge_hdr.setInvalid();
        transition select(eg_md.internal_hdr.header_type) {
            internal_header_t.NONE: parse_ethernet;
            internal_header_t.BRIDGE_HDR: parse_bridge_hdr;
            default: accept;
        }
    }

    state parse_bridge_hdr {
        pkt.extract(eg_md.bridge_hdr);
        transition parse_ethernet;
    }
	
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(pkt.lookahead<bit<16>>()) {
            ETHERTYPE_CUSTOM_1: parse_custom_metadata1;
            default: accept;
        }
    }
	
    state parse_custom_metadata1 {
        pkt.extract(hdr.custom_metadata1);
        transition select(pkt.lookahead<bit<16>>()) {
            ETHERTYPE_CUSTOM_2: parse_custom_metadata2;
            default: accept;
        }
    }	

    state parse_custom_metadata2 {
        pkt.extract(hdr.custom_metadata2);
        transition select(pkt.lookahead<bit<16>>()) {
            ETHERTYPE_CUSTOM_3: parse_custom_metadata3;
            default: accept;
        }
    }	

    state parse_custom_metadata3 {
        pkt.extract(hdr.custom_metadata3);
        transition accept;
    }	

}


// ---------------------------------------------------------------------------
// Egress 
// ---------------------------------------------------------------------------
control SwitchEgress(
        inout header_t hdr,
        inout metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprsr,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_md_for_oport) {

    action fw_del_custom() {

        eg_intr_md_for_dprsr.drop_ctl = 0x0;  
		hdr.custom_metadata1.setInvalid();
		hdr.custom_metadata2.setInvalid();
		hdr.custom_metadata3.setInvalid();
    }
	
	action fw_and_set_custom1() {
        eg_intr_md_for_dprsr.drop_ctl = 0x0;  
		hdr.custom_metadata1.setValid();
		hdr.custom_metadata1.ether_type = ETHERTYPE_CUSTOM_1;
    }

	action fw_and_set_custom2() {
        eg_intr_md_for_dprsr.drop_ctl = 0x0;  
		hdr.custom_metadata2.setValid();
		hdr.custom_metadata2.ether_type = ETHERTYPE_CUSTOM_2;
    }

	action fw_and_set_custom12() {
        eg_intr_md_for_dprsr.drop_ctl = 0x0;  
		
		hdr.custom_metadata1.setValid();
		hdr.custom_metadata1.ether_type = ETHERTYPE_CUSTOM_1;		
		hdr.custom_metadata2.setValid();
		hdr.custom_metadata2.ether_type = ETHERTYPE_CUSTOM_2;
    }

	action fw_and_set_custom3() {
        eg_intr_md_for_dprsr.drop_ctl = 0x0;  
		
		hdr.custom_metadata3.setValid();
		hdr.custom_metadata3.ether_type = ETHERTYPE_CUSTOM_3;
    }
	
    table pkt_custom {
        key = {
            eg_md.pkt_custom_type : exact;
        }

        actions = {
            fw_del_custom;
			fw_and_set_custom1;
			fw_and_set_custom2;
			fw_and_set_custom12;
			fw_and_set_custom3;
        }
		
        size = 1024;
    }
	
    apply {
	    eg_md.pkt_custom_type = eg_md.bridge_hdr.pkt_custom_type;
        pkt_custom.apply();
		
		eg_md.internal_hdr.setInvalid();
		eg_md.bridge_hdr.setInvalid();
    }
}

// ---------------------------------------------------------------------------
// Egress Deparser
// ---------------------------------------------------------------------------
control SwitchEgressDeparser(packet_out pkt,
                              inout header_t hdr,
                              in metadata_t eg_md,
                              in egress_intrinsic_metadata_for_deparser_t 
                                eg_intr_dprsr_md
                              ) {

    apply {
        //pkt.emit(hdr);
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.custom_metadata1);	
        pkt.emit(hdr.custom_metadata2);	
        pkt.emit(hdr.custom_metadata3);			
    }
}
		 
/*Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;*/ 

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;
		 
Switch(pipe) main;
