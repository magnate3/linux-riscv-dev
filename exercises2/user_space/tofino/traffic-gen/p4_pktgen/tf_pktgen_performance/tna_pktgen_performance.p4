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

#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#include "common/headers.p4"
#include "common/util.p4"


struct headers {
    pktgen_timer_header_t timer;
    ethernet_h ethernet;
    ipv4_h ipv4;
}

parser SwitchIngressParser(
       packet_in packet,
       out headers hdr,
       out empty_metadata_t md,
       out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        packet.extract(ig_intr_md);
        packet.advance(PORT_METADATA_SIZE);
	
	/*
	transition select(ig_intr_md.ingress_port) {
	    134 : parse_pktgen_timer;
	    default : reject;
	}
	*/
		
        pktgen_timer_header_t pktgen_timer_hdr = packet.lookahead<pktgen_timer_header_t>();
        transition select(pktgen_timer_hdr.app_id) {
            1 : parse_pktgen_timer;
            default : accept;
        }
    }

    state parse_pktgen_timer {
        packet.extract(hdr.timer);
        transition parse_ethernet;
    }
    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition parse_ipv4;
    }
    state parse_ipv4 {
	packet.extract(hdr.ipv4);
	transition accept;
    }
}


control SwitchIngressDeparser(
        packet_out pkt,
        inout headers hdr,
        in empty_metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {

    apply {
        pkt.emit(hdr);
    }
}


control SwitchIngress(
        inout headers hdr,
        inout empty_metadata_t md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

    Counter<bit<32>, bit<9>>(10, CounterType_t.PACKETS_AND_BYTES) indirect_counter;
    DirectCounter<bit<32>>(CounterType_t.PACKETS_AND_BYTES) direct_counter;
    bit<9> index;
    action set_port(PortId_t port) {
	ig_intr_tm_md.ucast_egress_port = port;
	direct_counter.count();
    }
    table t {
	key = { ig_intr_md.ingress_port : exact; }
	actions = {
	    set_port;	
	}
	counters = direct_counter;
    }

    apply {
	// For checking app_id is correct.
	if (hdr.timer.isValid()) {
	    index = 5w0 +++ hdr.timer.app_id;
	    indirect_counter.count(index);
	}
	// If it is not set to invalid, the packets will not go out. Why does this function make packets stuck?
	hdr.timer.setInvalid();
	t.apply();
        ig_intr_tm_md.bypass_egress = 1w1;
    }
}


Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;
