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
#define RECIRC_PORT 134
#include <t2na.p4>
#else
#define RECIRC_PORT 68
#include <tna.p4>
#endif

#include "common/headers.p4"
#include "common/util.p4"

struct metadata {
	bit<8> app_id;
}

struct headers {
    pktgen_recirc_header_t recirc_pattern;
    ethernet_h ethernet;
    vlan_tag_h vlan_tag;
    ipv4_h ipv4;
    tcp_h tcp;
    udp_h udp;
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
			RECIRC_PORT: parse_pktgen;
            default : parse_ethernet;
        }
    }
    state parse_pktgen {
        pktgen_recirc_header_t pktgen_hdr = packet.lookahead<pktgen_recirc_header_t>();
        transition select(pktgen_hdr.app_id) {
            3 : parser_pktgen_recirc_pattern;
            default : accept;
        }
    }
    state parser_pktgen_recirc_pattern {
        packet.extract(hdr.recirc_pattern);
        transition parse_ethernet;
    }
    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition accept;
    }
}

control SwitchIngressDeparser(
        packet_out pkt,
        inout headers hdr,
        in metadata ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {
    apply {
        pkt.emit(hdr);
    }
}

control SwitchIngress(
        inout headers hdr,
        inout metadata md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

    action drop() {
        ig_intr_dprsr_md.drop_ctl = 0x1;
    }

    action match_r(PortId_t port) {
        hdr.recirc_pattern.setInvalid();
        ig_intr_tm_md.ucast_egress_port = port;
        ig_intr_tm_md.bypass_egress = 1w1;
    }

    table check_recirculate {
        key = {
            hdr.recirc_pattern.app_id   : exact;
        }
        actions = {
            match_r;
        }
    }

    action set_port(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port;
    }
    table forward {
        key = { ig_intr_md.ingress_port : exact; }
        actions = { set_port; }
    }

    apply {
        forward.apply();
        check_recirculate.apply();

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
