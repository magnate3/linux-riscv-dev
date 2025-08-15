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


/*
TF1 : 68, 69, 70, 71
TF2 : pipe0: 1, 6, 7. other pipes's ports : 1, 2, 3, 4, 5, 6, 7
*/
#if __TARGET_TOFINO__ == 1
#define PKTGEN_PORT 68
#else
#define PKTGEN_PORT 134 // pipe1 port 6
#endif

struct headers {
    pktgen_timer_header_t timer;
    pktgen_port_down_header_t port_down;
    ethernet_h ethernet;
    vlan_tag_h vlan_tag;
    ipv4_h ipv4;
    tcp_h tcp;
    udp_h udp;
}

parser SwitchIngressParser(
       packet_in packet,
       out headers hdr,
       out empty_metadata_t md,
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
        pktgen_port_down_header_t pktgen_pd_hdr = packet.lookahead<pktgen_port_down_header_t>();
        transition select(pktgen_pd_hdr.app_id) {
                2 : parse_pktgen_port_down;
                default : parse_ethernet;
        }
    }
    state parse_pktgen_port_down {
        packet.extract(hdr.port_down);
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

    Counter<bit<32>, bit<9>>(256, CounterType_t.PACKETS_AND_BYTES) indirect_counter;

    action drop() {
        ig_intr_dprsr_md.drop_ctl = 0x1;
    }

    action match(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port;
        ig_intr_tm_md.bypass_egress = 1w1;
    }
    // t table is used for timer pktgen app
    table t {
        key = {
            hdr.timer.pipe_id : exact;
            hdr.timer.app_id  : exact;
            hdr.timer.batch_id : exact;
            hdr.timer.packet_id : exact;
            ig_intr_md.ingress_port : exact;
        }
        actions = {
            match;
        }
        //const default_action = NoAction();
        size = 1024;
    }
    // p table is used for port_down pktgen app
    table p {
        key = {
            hdr.port_down.pipe_id   : exact;
            hdr.port_down.app_id    : exact;
            hdr.port_down.port_num  : exact;
            hdr.port_down.packet_id : exact;
            ig_intr_md.ingress_port : exact;
        }
        actions = {
            match;
        }
        //const default_action = NoAction();
        size = 1024;
    }

    apply {
        indirect_counter.count(ig_intr_md.ingress_port);
        if (p.apply().hit) {
            hdr.port_down.setInvalid();
        } else {
            ig_intr_tm_md.ucast_egress_port = 144;
        }
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
