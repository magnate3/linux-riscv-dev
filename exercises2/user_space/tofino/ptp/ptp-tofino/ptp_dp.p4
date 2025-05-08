#include <core.p4>
#include <tna.p4>

#define CPU_PORT 192
#define ETH_P_1588 0x88F7

struct metadata_t {}
struct egress_metadata_t {}

header cpu_h {
    bit<7> pad;
    bit<9> port;
    bit<48> timestamp;
}

header ethernet_h {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

header ptp_common_h {
    bit<4> transportSpecific;
    bit<4> messageType;
    bit<4> reserved_1;
    bit<4> versionPTP;
    bit<16> messageLength;
    bit<8> domainNumber;
    bit<8> reserved_2;
    bit<16> flagField;
    bit<64> correctionField;
    bit<32> reserved_3;
    bit<80> sourcePortIdentity;
    // or
    // bit<64> sourcePortIdentity_clockIdentity;
    // bit<16> sourcePortIdentity_portNumber;
    bit<16> sequenceId;
    bit<8> controlField;
    bit<8> logMessageInterval;
}

struct header_t {
    cpu_h cpu;
    ethernet_h ethernet;
    ptp_common_h ptp;
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
            CPU_PORT : parse_cpu;
            default : parse_ethernet;
        }
    }

    state parse_cpu {
        pkt.extract(hdr.cpu);
        transition parse_ethernet;
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

    action drop_packet() {
        ig_dprsr_md.drop_ctl = 0x1; // Drop packet.
    }

    action set_egress(PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
    }

    table forwarding {
        key = {
            ig_intr_md.ingress_port : exact;
        }
        actions = {
            set_egress;
            drop_packet;
        }
        default_action = drop_packet;
    }

    apply {
        if (hdr.cpu.isValid()) {
            set_egress(hdr.cpu.port);
            hdr.cpu.setInvalid();
        } else if (hdr.ptp.isValid()) {
            set_egress(CPU_PORT);
            hdr.cpu.setValid();
            hdr.cpu.port = ig_intr_md.ingress_port;
            hdr.cpu.timestamp = ig_intr_md.ingress_mac_tstamp;
        } else {
            forwarding.apply();
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
        transition select(eg_intr_md.egress_port) {
            CPU_PORT : parse_cpu;
            default : parse_ethernet;
        }
    }

    state parse_cpu {
        pkt.extract(hdr.cpu);
        transition parse_ethernet;
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

    action enable_ts7() {
        eg_intr_md_for_oport.capture_tstamp_on_tx = 1;
    }

    table process_ptp {
        key = {
            hdr.ptp.messageType : exact;
        }
        actions = {
            enable_ts7;
            @defaultonly NoAction;
        }
        const default_action = NoAction();
        const entries = {
            0 : enable_ts7();
            1 : enable_ts7();
            2 : enable_ts7();
            3 : enable_ts7();
        }
    }

    apply {
        if (! hdr.cpu.isValid() && hdr.ptp.isValid()) process_ptp.apply();
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
