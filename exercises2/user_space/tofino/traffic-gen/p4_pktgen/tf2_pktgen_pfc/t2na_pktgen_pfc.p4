#include <t2na.p4>

struct port_md_t {
}

struct metadata {
	port_md_t port_md;
	bit<4>  app_id;
	bit<528> index;
}

header app_ctx_h {
	@pa_container_size("ingress", "hdr.app_ctx.data", 32, 32, 32, 32)
	bit<128> data;
}
header ethernet_h {
	bit<48> dst;
	bit<48> src;
	bit<16> etype;
}
struct headers {
	pktgen_pfc_header_t pfc;
	app_ctx_h app_ctx;
	ethernet_h ethernet;
}

struct pvs_entry_t { bit<4> app_id; }

parser iPrsr(packet_in packet, out headers hdr, out metadata md,
		out ingress_intrinsic_metadata_t ig_intr_md) {
	value_set<pvs_entry_t>(1)  pfc_apps;

	state start {
		packet.extract(ig_intr_md);
		transition select(ig_intr_md.resubmit_flag) {
			1 : parse_resubmit;
			0 : parse_port_md;
		}
	}
	state parse_resubmit {
		packet.advance(128);
		packet.advance(64);
		transition reject;
	}
	state parse_port_md {
		transition select(ig_intr_md.ingress_port) {
			6  : parse_pktgen;
			134: parse_pktgen;
			262: parse_pktgen;
			390: parse_pktgen;
			default : parse_ethernet;
		}
	}
	state parse_pktgen {
		bit<4> app_id = packet.lookahead<bit<8>>()[3:0];
		md.app_id = 3;
		transition select(app_id) {
			pfc_apps : parse_pktgen_pfc;
			default  : reject;
		}
	}
	state parse_pktgen_pfc {
		packet.extract(hdr.pfc);
		packet.extract(hdr.app_ctx);
		transition accept;
	}
	state parse_ethernet {
		md.app_id = 10;
		packet.extract(hdr.ethernet);
		transition accept;
	}
}

control i(inout headers hdr, inout metadata md,
		in ingress_intrinsic_metadata_t ig_intr_md,
		in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
		inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
		inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {


	action set_port(PortId_t port) {
		ig_intr_tm_md.ucast_egress_port = port;
	}
		
	table forward {
		key = { ig_intr_md.ingress_port : exact; } 
		actions = { set_port; }
	}

	apply {
		// 136, 144 are lying on pipeline 1
		forward.apply();
		if (hdr.pfc.isValid()) {
		    // hdr.pfc.setInvalid();
		}
	}
}

control iDprsr(packet_out packet, inout headers hdr, in metadata md,
			in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
	apply {
		packet.emit(hdr);
		/*
		packet.emit(hdr.pfc);
		packet.emit(hdr.app_ctx);
		packet.emit(hdr.ethernet);
		*/
	}
}

parser ePrsr(packet_in packet, out headers hdr, out metadata md,
		out egress_intrinsic_metadata_t eg_intr_md) {
	state start { transition reject; }
}

control e(inout headers hdr, inout metadata md, in egress_intrinsic_metadata_t eg_intr_md,
		in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
		inout egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprs,
		inout egress_intrinsic_metadata_for_output_port_t eg_intr_md_for_oport) {
	apply {}
}

control eDprsr(packet_out packet, inout headers hdr, in metadata md,
		in egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprs) {
	apply {}
}

Pipeline(iPrsr(), i(), iDprsr(), ePrsr(), e(), eDprsr()) hohoho;
Switch(hohoho) main;
