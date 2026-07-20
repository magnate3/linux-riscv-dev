// typedef bit<9> PortId_t defined by default

#include <core.p4>
#include <tna.p4>

// typedefs and #defines
typedef bit<48> mac_addr_t;
typedef bit<32> ip4Addr_t;
#define ETH_TYPE_IPV4   0x800
#define ETH_TYPE_ARP    0x0806
#define ETH_TYPE_RCLT   0x1000

#define IP_PROTOCOLS_ICMP 1
#define IP_PROTOCOLS_TCP 6
#define IP_PROTOCOLS_UDP 17

#define IP_ID_RCLT      1000

//######## Header definations #############
header ethernet_h {
    mac_addr_t dst_addr;
    mac_addr_t src_addr;
    bit<16> ether_type;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header rclt_t {
    bit<16> rclt_count;
    // bit<16> ether_type;
    // bit<48> mac_src_addr;
    bit<16> dport;
    bit<16> ipv4_id;
}

struct header_t {
    ethernet_h ethernet;
    ipv4_t     ipv4;
    rclt_t     rclt;
}

struct metadata_t { }
struct empty_header_t {}
struct empty_metadata_t {}

// Ingress Parser
parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

	// TNA specific code
	state start {
		pkt.extract(ig_intr_md);
		pkt.advance(PORT_METADATA_SIZE);
		transition parse_ethernet;
	}

	state parse_ethernet {
		pkt.extract(hdr.ethernet);
		transition select(hdr.ethernet.ether_type){
            ETH_TYPE_IPV4: parse_ipv4;
            default: accept;
        }
	}

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.identification){
            IP_ID_RCLT: parse_rclt;
            default: accept;
        }
    }

    state parse_rclt{
        pkt.extract(hdr.rclt);
        transition accept;
    }
}

// Ingress Deparser
control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {

	apply {
		pkt.emit(hdr);
	}
}

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // Counter to count recirculate and normal packet
    // Index 0 for new packets, index 1 for total number of recircultation
    // Counter<bit<32>, bit<8>>(2, CounterType_t.PACKETS_AND_BYTES) c_rclt;
    DirectCounter<bit<32>>(CounterType_t.PACKETS_AND_BYTES) c_rclt_forward;
    
    action nop() {
        c_rclt_forward.count();
    }

    action act_set_rclt (bit<16> dport, PortId_t iface_rclt){
        // c_rclt.count(0);
        c_rclt_forward.count();
        ig_tm_md.ucast_egress_port = iface_rclt;
        hdr.rclt.rclt_count = hdr.rclt.rclt_count + 1;
        hdr.rclt.dport = dport;
        // hdr.rclt.ether_type = hdr.ethernet.ether_type;
        // hdr.rclt.mac_src_addr = hdr.ethernet.src_addr;
        hdr.rclt.ipv4_id = hdr.ipv4.identification;
        // hdr.ethernet.ether_type = ETHER_TYPE_RCLT;
        // hdr.ethernet.src_addr = ETHER_SRC_ADDR_RCLT;
        hdr.ipv4.identification = IP_ID_RCLT;
    }

    action act_continue_rclt (PortId_t iface_rclt){
        // c_rclt.count(1);
        c_rclt_forward.count();
        ig_tm_md.ucast_egress_port = iface_rclt;
        hdr.rclt.rclt_count = hdr.rclt.rclt_count + 1;
    }

    action act_clear_rclt(){
        // c_rclt.count(1);
        c_rclt_forward.count();
        ig_tm_md.ucast_egress_port = (PortId_t)hdr.rclt.dport;
        // hdr.ethernet.ether_type = hdr.rclt.ether_type;
        // hdr.ethernet.src_addr = hdr.rclt.mac_src_addr;
        hdr.ipv4.identification = hdr.rclt.ipv4_id;
        hdr.rclt.setInvalid();
    }

    table tbl_rclt_forward {
        key = {
            ig_intr_md.ingress_port : ternary;
            hdr.rclt.rclt_count : ternary;
        }

        actions = {
            act_set_rclt;
            act_continue_rclt;
            act_clear_rclt;
            nop;
        }

        const default_action = nop();
        size = 16;
        counters = c_rclt_forward;
    }

    action act_wire(PortId_t dport){
        ig_tm_md.ucast_egress_port = dport;
    }

    table tbl_wire {
        key = {
            ig_intr_md.ingress_port : exact;
        }

        actions = {
            act_wire;
        }

        size = 16;
    }

	apply {
        if (hdr.ipv4.isValid()){
            if (!hdr.rclt.isValid()){
                // Initialize rclt header if it is not valid already
                hdr.rclt.setValid();
                hdr.rclt.rclt_count = 0;
            }

            // Apply the tbl_rclt_forward table
            tbl_rclt_forward.apply();

		}
        else{
            // This will handle arp and garbage non-ipv4 packets
            tbl_wire.apply();
        }
	}
}


// Empty egress parser/control blocks
parser EmptyEgressParser(
        packet_in pkt,
        out empty_header_t hdr,
        out empty_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

control EmptyEgressDeparser(
        packet_out pkt,
        inout empty_header_t hdr,
        in empty_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md) {
    apply {
        pkt.emit(hdr);
    }
}

control EmptyEgress(
        inout empty_header_t hdr,
        inout empty_metadata_t eg_md,
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
