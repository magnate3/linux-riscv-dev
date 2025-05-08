#include <core.p4>
#include <tna.p4>

#define CPU_PORT 192
#define ETH_P_802_EX1 0x88b5


struct metadata_t {}
struct egress_metadata_t {}

header ethernet_h {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

header poc_ctrl_h {
    bit<2> state;
    bit<5> pad;
    bit<9> port;
}

header ts48_h {
    bit<16> pad;
    bit<48> ts;
}

header ts64_h {
    bit<64> ts;
}

struct header_t {
    pktgen_timer_header_t pktgen;
    ptp_metadata_t ptp_md;
    ethernet_h ethernet;
    poc_ctrl_h poc_ctrl;
    ts64_h s1_e; // TS6 on the originating switch
    ts48_h s2_i; // TS1 on the peer switch
    ts64_h s2_e; // TS6 on the peer switch
    ts48_h s1_i; // TS1 on the originating switch
}

parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition select(ig_intr_md.ingress_port) {
            68      : parse_pktgen;
            default : parse_ethernet;
        }
    }

    state parse_pktgen {
        pkt.extract(hdr.pktgen);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETH_P_802_EX1: parse_poc_ctrl;
            default: accept;
        }
    }

    state parse_poc_ctrl {
        pkt.extract(hdr.poc_ctrl);
        pkt.extract(hdr.s1_e);
        pkt.extract(hdr.s2_i);
        pkt.extract(hdr.s2_e);
        pkt.extract(hdr.s1_i);
        transition accept;
    }
}

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in    ingress_intrinsic_metadata_t ig_intr_md,
        in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md)
{

    action set_s1_ingress() {
        // Packet is received back at originating switch, set ingress timestamp and deliver to CPU
        hdr.s1_i.ts = ig_intr_md.ingress_mac_tstamp;
        ig_tm_md.ucast_egress_port = CPU_PORT; // Send to CPU
    }

    action set_s2_ingress() {
        // Packet is received by peer switch, set ingress timestamp and set egress to ingress port
        hdr.s2_i.ts = ig_intr_md.ingress_mac_tstamp;
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;  // Send back to SW1
    }

    action send_to_peer() {
        // Packet is from control-plane, set egress to based on control header
        ig_tm_md.ucast_egress_port = hdr.poc_ctrl.port; // Send to Peer
    }

    action drop_packet() {
        ig_dprsr_md.drop_ctl = 0x1; // Drop packet.
    }

    action set_egress(PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
    }

    table process_poc_pkt {
        key = {
            hdr.ethernet.ether_type : exact;
            hdr.poc_ctrl.state : exact;
        }
        actions = {
            set_s1_ingress;
            set_s2_ingress;
            send_to_peer;
            @defaultonly NoAction;
        }
        const default_action = NoAction();
        const entries = {
            (ETH_P_802_EX1, 0): send_to_peer(); // First time in source switch
            (ETH_P_802_EX1, 1): set_s2_ingress(); // In Peer switch
            (ETH_P_802_EX1, 2): set_s1_ingress(); // Back in source switch
        }
    }

    table l1_forwarding {
        key = {
            ig_intr_md.ingress_port : exact;
        }
        actions = {
            set_egress;
            drop_packet;
        }
        const entries = {
            68 : set_egress(60); // Pktgen -> 24/0
            // 60 : set_egress(60); // 24/0 -> 20/0
            28 : set_egress(28); // 20/0 -> 24/0
        }
        default_action = drop_packet;
    }

    apply {
        if (hdr.poc_ctrl.isValid()) process_poc_pkt.apply();
        else {
            l1_forwarding.apply();
            hdr.pktgen.setInvalid();
        }
    }
}

control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
         pkt.emit(hdr);
    }
}

parser SwitchEgressParser(
        packet_in pkt,
        out header_t hdr,
        out egress_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {

    state start {
        pkt.extract(eg_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETH_P_802_EX1: parse_poc_ctrl;
            default: accept;
        }
    }

    state parse_poc_ctrl {
        pkt.extract(hdr.poc_ctrl);
        pkt.extract(hdr.s1_e);
        pkt.extract(hdr.s2_i);
        pkt.extract(hdr.s2_e);
        pkt.extract(hdr.s1_i);
        transition accept;
    }
}

control SwitchEgress(
        inout header_t hdr,
        inout egress_metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprsr,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_md_for_oport) {

    // action set_s1_egress() {
    //     // Packet is from control-plane, Enable TS6 and TS7
    //     eg_intr_md_for_oport.capture_tstamp_on_tx = 1; // Enable TS7
    //     eg_intr_md_for_oport.update_delay_on_tx = 1; // Enable TS6
    //     hdr.s1_e.setInvalid(); // Remove placeholder
    //     hdr.ptp_md.setValid();
    //     hdr.ptp_md.udp_cksum_byte_offset = 8w0;
    //     hdr.ptp_md.cf_byte_offset = 8w16;
    //     hdr.ptp_md.updated_cf = 48w0;
    //     hdr.poc_ctrl.state = 1; // 0 => 1
    // }
    //
    // action set_s2_egress() {
    //     // Packet is returning to source switch, Enable TS6 and TS7
    //     eg_intr_md_for_oport.capture_tstamp_on_tx = 1; // Disable TS7
    //     eg_intr_md_for_oport.update_delay_on_tx = 1; // Enable TS6
    //     hdr.s2_e.setInvalid(); // Remove placeholder
    //     hdr.ptp_md.setValid();
    //     hdr.ptp_md.udp_cksum_byte_offset = 8w0;
    //     hdr.ptp_md.cf_byte_offset = 8w32;
    //     hdr.ptp_md.updated_cf = 48w0;
    //     hdr.poc_ctrl.state = 2; // 1 => 2
    // }

    action set_ets(bit<8> byte_offset, bit<2> poc_state) {
        // Enable TS6 and TS7, increment state
        eg_intr_md_for_oport.capture_tstamp_on_tx = 1; // Disable TS7
        eg_intr_md_for_oport.update_delay_on_tx = 1; // Enable TS6
        hdr.s2_e.setInvalid(); // Remove placeholder
        hdr.ptp_md.setValid();
        hdr.ptp_md.udp_cksum_byte_offset = 8w0;
        hdr.ptp_md.cf_byte_offset = byte_offset;
        hdr.ptp_md.updated_cf = 48w0;
        hdr.poc_ctrl.state = poc_state;
    }

    table process_poc_pkt {
        key = {
            hdr.ethernet.ether_type : exact;
            hdr.poc_ctrl.state : exact;
        }
        actions = {
            // set_s1_egress;
            // set_s2_egress;
            set_ets;
            @defaultonly NoAction;
        }
        const default_action = NoAction();
        const entries = {
            //(ETH_P_802_EX1, 0): set_s1_egress(); // First time in source switch
            //(ETH_P_802_EX1, 1): set_s2_egress(); // In Peer switch
            (ETH_P_802_EX1, 0): set_ets(16, 1); // First time in source switch
            (ETH_P_802_EX1, 1): set_ets(32, 2); // In Peer switch
            // (0x0000, 2): NoAction(); // Back in source switch, no action needed
        }
    }

    apply {
        if (hdr.poc_ctrl.isValid()) process_poc_pkt.apply();
    }
}

control SwitchEgressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in egress_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
    apply {
        pkt.emit(hdr);
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;

Switch(pipe) main;
