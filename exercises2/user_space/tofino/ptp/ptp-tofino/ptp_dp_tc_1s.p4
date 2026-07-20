#include <core.p4>
#include <tna.p4>

#define CPU_PORT 192
#define ETH_P_IP   0x0800
#define ETH_P_IPV6 0x86DD
#define ETH_P_1588 0x88F7

struct ingress_metadata_t {
    bit<64> ingress_timestamp;
}
struct egress_metadata_t {}

header ethernet_h {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

header ptp_h {
    bit<4> transportSpecific;
    bit<4> messageType;
    bit<4> reserved_1;
    bit<4> versionPTP;
    bit<16> messageLength;
    bit<8> domainNumber;
    bit<8> reserved_2;
    bit<16> flagField;
}

header ptp_correctionField_h {
    bit<64> value;
}

struct header_t {
    ptp_metadata_t ptp_metadata;
    ethernet_h ethernet;
    // IPv4
    // IPv6
    // UDP
    ptp_h ptp;
    ptp_correctionField_h ptp_correctionField;

}

parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out ingress_metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition select(ig_intr_md.ingress_port) {
            default : parse_ethernet;
        }
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            // ETH_P_IP: parse_ipv4
            // ETH_P_IPV6: parse_ipv6
            ETH_P_1588: parse_ptp;
            default: accept;
        }
    }

    state parse_ptp {
        pkt.extract(hdr.ptp);
        pkt.extract(hdr.ptp_correctionField);
        transition accept;
    }
}

control SwitchIngress(
        inout header_t hdr,
        inout ingress_metadata_t ig_md,
        in    ingress_intrinsic_metadata_t ig_intr_md,
        in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md)
{

    action drop_packet() {
        ig_dprsr_md.drop_ctl = 0x1; // Drop packet.
    }

    action set_egress(PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
    }

    action ptp_modify_correctionField() {
        hdr.ptp_correctionField.value = hdr.ptp_correctionField.value - ig_md.ingress_timestamp;
    }

    table ptp {
        key = {
            hdr.ptp.messageType & 0xC : exact;
        }
        size = 1;
        actions = {
            ptp_modify_correctionField;
            @defaultonly NoAction;
        }
        const default_action = NoAction();
        const entries = {
            0 : ptp_modify_correctionField();
        }
    }

    table forwarding {
        key = {
            ig_intr_md.ingress_port : exact;
        }
        size = 256;
        actions = {
            set_egress;
            drop_packet;
        }
        const default_action = drop_packet;
        const entries = {
            0 : set_egress(1);
            1 : set_egress(0);
        }
    }

    apply {
        if (hdr.ptp.isValid()) {
            ig_md.ingress_timestamp[63:16] = ig_intr_md.ingress_mac_tstamp;
            ptp.apply();
        }
        forwarding.apply();
    }
}

control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in ingress_metadata_t ig_md,
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
        transition select(eg_intr_md.egress_port) {
            default : parse_ethernet;
        }
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETH_P_1588: parse_ptp;
            default: accept;
        }
    }

    state parse_ptp {
        pkt.extract(hdr.ptp);
        pkt.extract(hdr.ptp_correctionField);
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

    action enable_ts6(bit<8> udp_cksum_byte_offset, bit<8> cf_byte_offset) {
        eg_intr_md_for_oport.update_delay_on_tx = 1;
        hdr.ptp_metadata.setValid();
        hdr.ptp_metadata.udp_cksum_byte_offset = udp_cksum_byte_offset;
        hdr.ptp_metadata.cf_byte_offset = cf_byte_offset;
        hdr.ptp_metadata.updated_cf = hdr.ptp_correctionField.value[63:16];
        hdr.ptp_correctionField.setInvalid();
    }

    table ptp {
        key = {
            hdr.ethernet.ether_type : exact;
            hdr.ptp.messageType & 0xC: exact;
        }
        size = 1;
        actions = {
            enable_ts6;
            @defaultonly NoAction;
        }
        const default_action = NoAction();
        const entries = {
            (ETH_P_1588, 0) : enable_ts6(0, 22);
        }
    }

    apply {
        if (hdr.ptp.isValid()) ptp.apply();
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
