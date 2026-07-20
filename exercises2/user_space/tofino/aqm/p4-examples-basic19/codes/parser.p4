
/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/
#include "headers.p4"

const bit<16> TYPE_IPV4 = 0x800;


parser MyParser(packet_in packet,
                out headers hdr,
                inout local_metadata_t metadata,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            IP_PROTO_UDP: parse_udp;
            IP_PROTO_TCP: parse_tcp;
            default: accept;
        }
    }
    state parse_udp {
    packet.extract(hdr.udp);
    transition select(hdr.ipv4.dscp) {
        DSCP_INT &&& DSCP_MASK: parse_shim;
        default:  accept;
    }
}

state parse_tcp {
    packet.extract(hdr.tcp);
    transition select(hdr.ipv4.dscp) {
        DSCP_INT &&& DSCP_MASK: parse_shim;
        default:  accept;
    }
}

state parse_shim {
    packet.extract(hdr.intl4_shim);
    transition parse_int_hdr;
}

state parse_int_hdr {
    packet.extract(hdr.int_header);
    transition accept;
}

}
