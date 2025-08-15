#include <core.p4>
#include <tna.p4>

/* Header Stuff */
enum bit<16> ether_type_t {
    TPID = 0x8100,
    IPV4 = 0x0800,
    ARP  = 0x0806,
    IPV6 = 0x86DD,
    MPLS = 0x8847
}

enum bit<8> ip_protocol_t {
    ICMP = 1,
    IGMP = 2,
    TCP  = 6,
    UDP  = 17
}

enum bit<16> arp_opcode_t {
    REQUEST = 1,
    REPLY   = 2
}


enum bit<8> icmp_type_t {
    ECHO_REPLY   = 0,
    ECHO_REQUEST = 8
}

typedef bit<48> mac_addr_t;
typedef bit<32> ipv4_addr_t;

/* Metadata and Table Stuff */
const int IPV4_HOST_SIZE = 65536;
const int IPV4_LPM_SIZE  = 12288;

#define NEXTHOP_ID_WIDTH 14
typedef bit<NEXTHOP_ID_WIDTH> nexthop_id_t;
const int NEXTHOP_SIZE = 1 << NEXTHOP_ID_WIDTH;

/* Standard ethernet header */
header ethernet_h {
    mac_addr_t    dst_addr;
    mac_addr_t    src_addr;
    ether_type_t  ether_type;
}

header vlan_tag_h {
    bit<3>        pcp;
    bit<1>        cfi;
    bit<12>       vid;
    ether_type_t  ether_type;
}

header ipv4_h {
    bit<4>          version;
    bit<4>          ihl;
    bit<8>          diffserv;
    bit<16>         total_len;
    bit<16>         identification;
    bit<3>          flags;
    bit<13>         frag_offset;
    bit<8>          ttl;
    ip_protocol_t   protocol;
    bit<16>         hdr_checksum;
    ipv4_addr_t     src_addr;
    ipv4_addr_t     dst_addr;
}

header ipv4_options_h {
    varbit<320> data;
}

header icmp_h {
    // TODO: finish ICMP header
}

header arp_h {
    // TODO: finish ARP header
} 

header arp_ipv4_h {
    mac_addr_t   src_hw_addr;
    ipv4_addr_t  src_proto_addr;
    mac_addr_t   dst_hw_addr;
    ipv4_addr_t  dst_proto_addr;
}

struct my_ingress_headers_t {
    // TODO: finish packet header stack
}

struct my_ingress_metadata_t {
    ipv4_addr_t   dst_ipv4;
    bit<1>        ipv4_csum_err;
}

struct my_egress_headers_t {
}

struct my_egress_metadata_t {
}

// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------

parser SwitchIngressParser(packet_in  pkt,
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    out ingress_intrinsic_metadata_t  ig_intr_md)
{
    Checksum() ipv4_checksum;
    
    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition meta_init;
    }

    state meta_init {
        meta.ipv4_csum_err = 0;
        meta.dst_ipv4      = 0;
        transition parse_ethernet;
    }
    
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ether_type_t.TPID :  parse_vlan_tag;
            ether_type_t.IPV4 :  parse_ipv4;
            // TODO: finish parser flow
            default:  accept;
        }
    }

    state parse_vlan_tag {
        pkt.extract(hdr.vlan_tag.next);
        transition select(hdr.vlan_tag.last.ether_type) {
            ether_type_t.TPID :  parse_vlan_tag;
            ether_type_t.IPV4 :  parse_ipv4;
            // TODO: finish parser flow
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        meta.dst_ipv4 = hdr.ipv4.dst_addr;
        
        ipv4_checksum.add(hdr.ipv4);
        meta.ipv4_csum_err = (bit<1>)ipv4_checksum.verify();
        
        // TODO: finish parser flow
        transition accept;
    }   
}

// ---------------------------------------------------------------------------
// Ingress
// ---------------------------------------------------------------------------

control SwitchIngress(
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{
    nexthop_id_t    nexthop_id  = 0;
    mac_addr_t      mac_da      = 0;
    mac_addr_t      mac_sa      = 0;
    PortId_t        egress_port = 511; /* Non-existent port */
    bit<8>          ttl_dec     = 0;

    action set_nexthop(nexthop_id_t nexthop) {
        nexthop_id = nexthop;
    }
    
    table ipv4_host {
        key     = {
            meta.dst_ipv4 : exact @name("dst_ip");
        }
        actions = {
            set_nexthop;
            NoAction;
        }
        size    = IPV4_HOST_SIZE;
        const default_action = NoAction();
    }

    table ipv4_lpm {
        key            = {
            meta.dst_ipv4 : lpm @name("dst_ip");
        }
        actions        = {
            set_nexthop;
        }
        
        default_action = set_nexthop(0);
        size           = IPV4_LPM_SIZE;
    }

    action send(PortId_t port) {
        mac_da      = hdr.ethernet.dst_addr;
        mac_sa      = hdr.ethernet.src_addr;
        egress_port = port;
        ttl_dec     = 0;
    }

    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }

    action l3_switch(PortId_t port, bit<48> new_mac_da, bit<48> new_mac_sa) {
        mac_da      = new_mac_da;
        mac_sa      = new_mac_sa;
        egress_port = port;
        ttl_dec     = 1;        
    }

    table nexthop {
        key            = {
            nexthop_id : exact @name("nh_id");
        }
        actions        = {
            send;
            drop;
            l3_switch;
        }
        size           = NEXTHOP_SIZE;
        default_action = drop();
    }

    action send_back() {
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
    }

    action forward_ipv4() {
        hdr.ethernet.dst_addr      = mac_da;
        hdr.ethernet.src_addr      = mac_sa;
        hdr.ipv4.ttl               = hdr.ipv4.ttl |-| ttl_dec;
        ig_tm_md.ucast_egress_port = egress_port;
    }

    action send_arp_reply() {
        // TODO: implement for generate ARP reply packet

        send_back();
    }

    action send_icmp_echo_reply() {
        // TODO: implement for generate ICMP echo reply packet

        send_back();
    }

    table forward_or_respond {
        key = {
            hdr.arp.isValid()       : exact;
            hdr.arp_ipv4.isValid()  : exact;
            hdr.ipv4.isValid()      : exact;
            hdr.icmp.isValid()      : exact;
            hdr.arp.opcode          : ternary @name("arp_opcode");
            hdr.icmp.msg_type       : ternary @name("icmp_msg_type");
        }
        actions = {
            forward_ipv4;
            send_arp_reply;
            send_icmp_echo_reply;
            drop;
        }
        const entries = {
            (false, false,  true,   false,  _,                      _):
            forward_ipv4();
            // TODO: add const entry for ARP request and ICMP request
            (false, false,  true,   true,   _,                      _):
            forward_ipv4();
        }
        default_action = drop();
    }
    
    /* The algorithm */
    apply {
        if (meta.ipv4_csum_err == 0) {         /* No checksum error for ARP! */
            if (!ipv4_host.apply().hit) {
                ipv4_lpm.apply();
            }
        }

        nexthop.apply();
        forward_or_respond.apply();
    }
}

// ---------------------------------------------------------------------------
// Ingress Parser
// ---------------------------------------------------------------------------
control SwitchIngressDeparser(packet_out                pkt,
    inout my_ingress_headers_t                          hdr,
    in    my_ingress_metadata_t                         meta,
    in    ingress_intrinsic_metadata_for_deparser_t     ig_dprsr_md)
{
    Checksum() ipv4_checksum;
    
    apply {
        hdr.ipv4.hdr_checksum = ipv4_checksum.update({
                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.diffserv,
                hdr.ipv4.total_len,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.frag_offset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.src_addr,
                hdr.ipv4.dst_addr
                /* Adding hdr.ipv4_options.data results in an error */
            });
        pkt.emit(hdr);
    }
}

// ---------------------------------------------------------------------------
// Egress Parser
// ---------------------------------------------------------------------------
parser SwitchEgressParser(packet_in  pkt,
    out my_egress_headers_t          hdr,
    out my_egress_metadata_t         meta,
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

// ---------------------------------------------------------------------------
// Egress 
// ---------------------------------------------------------------------------

control SwitchEgress(
    inout my_egress_headers_t                          hdr,
    inout my_egress_metadata_t                         meta,
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
    apply {
    }
}

// ---------------------------------------------------------------------------
// Egress Deparser
// ---------------------------------------------------------------------------

control SwitchEgressDeparser(packet_out             pkt,
    inout my_egress_headers_t                       hdr,
    in    my_egress_metadata_t                      meta,
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}

Pipeline(
    SwitchIngressParser(),
    SwitchIngress(),
    SwitchIngressDeparser(),
    SwitchEgressParser(),
    SwitchEgress(),
    SwitchEgressDeparser()
) pipe;

Switch(pipe) main;
