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

/*
TF1 : 68, 69, 70, 71
TF2 : pipe0: 1, 6, 7. other pipes's ports : 1, 2, 3, 4, 5, 6, 7
*/
#if __TARGET_TOFINO__ == 1
#define PKTGEN_PORT 68
#else
#define PKTGEN_PORT 134 // pipe1 port 6
#endif

// I want to generate this fucking packet!!!!!!!!!!!!!
header app_ctx_h {
	bit<128> data;
}
header ethernet_h{
	bit<48> dst;
	bit<48> src;
	bit<16> etype;
}

struct headers {
	pktgen_recirc_header_t recirc_dprsr;
	app_ctx_h app_ctx;
	ethernet_h ethernet;
}
struct metadata {
	bit<8>  app_id;
}

parser SwitchIngressParser(
		packet_in packet,
		out headers hdr,
		out metadata md,
		out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        packet.extract(ig_intr_md);
        packet.advance(PORT_METADATA_SIZE);

        transition select(ig_intr_md.ingress_port) {
            PKTGEN_PORT : parse_pktgen;
            default : parse_ethernet;
        }
    }
	state parse_pktgen {
		pktgen_deparser_header_t pktgen_pd_hdr = packet.lookahead<pktgen_deparser_header_t>();
		transition select(pktgen_pd_hdr.app_id) {
      		3 : parse_pktgen_recirc_dprsr;
			default : accept;
		}
	}
	state parse_pktgen_recirc_dprsr {
		md.app_id = 3;
		packet.extract(hdr.recirc_dprsr);
		packet.extract(hdr.app_ctx);
		transition accept;
	}
	state parse_ethernet {
		md.app_id = 10;
		packet.extract(hdr.ethernet);
		transition accept;
	}
}


control SwitchIngressDeparser(
		packet_out pkt,
		inout headers hdr,
		in metadata ig_md,
		in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {

	Pktgen() pgen;
	apply {
		if (ig_intr_dprsr_md.pktgen == 1w1) {
			pgen.emit(hdr.app_ctx);
		}
		pkt.emit(hdr.app_ctx);
		pkt.emit(hdr.ethernet);
	}
}


control SwitchIngress(
		inout headers hdr,
		inout metadata md,
		in ingress_intrinsic_metadata_t ig_intr_md,
		in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
		inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
		inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

	action set_dprsr_trig_hdr(bit<128> h) {
		hdr.app_ctx.data = h;
		ig_intr_dprsr_md.pktgen = 1w1;
		ig_intr_dprsr_md.pktgen_address = 0;
		ig_intr_dprsr_md.pktgen_length = 64;
	}
	action noaction() {}
	table dprsr {
		actions = { set_dprsr_trig_hdr; }
		default_action = set_dprsr_trig_hdr(0x88889999aaaabbbbccccddddeeeeffff);
		size = 16;
	}
	action set_port(PortId_t port) {
		ig_intr_tm_md.ucast_egress_port = port;
	}
	table forward {
		key = { ig_intr_md.ingress_port : exact; }
		actions = { set_port; noaction; }
		default_action = noaction();
	}

	apply {
		if (ig_intr_md.ingress_port == 136) {
			dprsr.apply();
		}
		forward.apply();
	}

}

parser EmptyEgressParser(
		packet_in pkt,
		out headers hdr,
		out metadata eg_md,
		out egress_intrinsic_metadata_t eg_intr_md) {
	state start {
		transition accept;
	}
}

control EmptyEgressDeparser(
		packet_out pkt,
		inout headers hdr,
		in metadata eg_md,
		in egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md) {
	apply {}
}

control EmptyEgress(
		inout headers hdr,
		inout metadata eg_md,
		in egress_intrinsic_metadata_t eg_intr_md,
		in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
		inout egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md,
		inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
	apply {}
}



Pipeline(SwitchIngressParser(),
		SwitchIngress(),
		SwitchIngressDeparser(),
		EmptyEgressParser(),
		EmptyEgress(),
		EmptyEgressDeparser()) pipe;

Switch(pipe) main;
