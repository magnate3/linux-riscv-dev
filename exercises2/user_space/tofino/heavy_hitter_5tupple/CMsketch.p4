/* This is a simple wire example that forwards packets 
   out of the same port that they came in on. */

#include <core.p4>
#include <tna.p4>


/*=============================================
=            Headers and metadata.            =
=============================================*/
typedef bit<48> mac_addr_t;
header ethernet_h {
    mac_addr_t dst_addr;
    mac_addr_t src_addr;
    bit<16> ether_type;
}

typedef bit<32> ipv4_addr_t;
header ipv4_h {
    bit<4> version;
    bit<4> ihl;
    bit<8> tos;
    bit<16> total_len;
    bit<16> identification;
    bit<3> flags;
    bit<13> frag_offset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdr_checksum;
    ipv4_addr_t src_addr;
    ipv4_addr_t dst_addr;
}

header tcp_h {
	bit<16> sport;
	bit<16> dport;
	bit<32> seqNo;
	bit<32> ackNo;
	bit<4> dataOffset;
	bit<4> res;
	bit<8> flags;
	bit<16> window;
	bit<16> checksum;
	bit<16> urgentPtr;
}

// Global headers and metadata
struct header_t {
    ethernet_h ethernet;
    ipv4_h ip;
    tcp_h tcp;
}
struct metadata_t {
    bit<16> qid;
    bit<32> src_ip_addr;
    bit<32> dst_ip_addr;
    bit<8> ip_protocol;
    bit<8> src_tcp_port;
    bit<8> dst_tcp_port;
    bit<16> tuple_register_index;
    bit<16> tuple_register_index_2;
}
    
/*===============================
=            Parsing            =
===============================*/
// Parser for tofino-specific metadata.
parser TofinoIngressParser(
        packet_in pkt,        
        out ingress_intrinsic_metadata_t ig_intr_md,
        out header_t hdr,
        out metadata_t md) {
    state start {
        pkt.extract(ig_intr_md);
        transition select(ig_intr_md.resubmit_flag) {
            1 : parse_resubmit;
            0 : parse_port_metadata;
        }
    }
    state parse_resubmit {
        // Parse resubmitted packet here.
        transition reject;
    }
    state parse_port_metadata {
        pkt.advance(64); // skip this.
        transition accept;
    }
}


const bit<16> ETHERTYPE_IPV4 = 16w0x0800;
parser EthIpParser(packet_in pkt, out header_t hdr, out metadata_t md){
    state start {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ip;
            default : accept;
        }
    }
    state parse_ip {
        pkt.extract(hdr.ip);
        transition select(hdr.ip.protocol) {
            6 : parse_tcp;
            default : reject; //accepting tcp packets only
        }
    }
    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }
}


parser TofinoEgressParser(
        packet_in pkt,
        out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

/*========================================
=            Ingress parsing             =
========================================*/

parser IngressParser(
        packet_in pkt,
        out header_t hdr, 
        out metadata_t md,
        out ingress_intrinsic_metadata_t ig_intr_md)
{
    state start {
        TofinoIngressParser.apply(pkt, ig_intr_md, hdr, md);
        EthIpParser.apply(pkt, hdr, md);
        transition accept;
    }
}


control CiL2Fwd(
        in ingress_intrinsic_metadata_t ig_intr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {
    apply {
        //Please place your ports here 132 is the ingress port and 146 is th monitoing system port. 
        if(ig_intr_md.ingress_port == 132){
            ig_tm_md.ucast_egress_port = 164;
            ig_tm_md.mcast_grp_a = 1;
        }else if(ig_intr_md.ingress_port == 164){
            ig_tm_md.ucast_egress_port = 132;
        }else{
            ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        }
    }
}

/*===========================================
=            ingress match-action             =
===========================================*/
control Ingress(
        inout header_t hdr, 
        inout metadata_t md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    CiL2Fwd() ciL2Fwd;
    apply {
        ciL2Fwd.apply(ig_intr_md, ig_tm_md);
    }
}

control IngressDeparser(
        packet_out pkt, 
        inout header_t hdr, 
        in metadata_t md,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
        pkt.emit(hdr);
    }
}

/*======================================
=            Egress parsing            =
======================================*/
parser EgressParser(
        packet_in pkt,
        out header_t hdr, 
        out metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    TofinoEgressParser() tofino_parser;
    EthIpParser() eth_ip_parser; 
    state start {
        tofino_parser.apply(pkt, eg_intr_md);
        transition parse_packet;
    }
    state parse_packet {
        eth_ip_parser.apply(pkt, hdr, eg_md);
        transition accept;        
    }
}

/*=========================================
=            Egress match-action            =
=========================================*/
control Egress(
        inout header_t hdr, 
        inout metadata_t eg_mg,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
        inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_oport_md){


    const bit<32> table_size = 1 << 16;

    action do_init_metadata_32(){
        eg_mg.qid = 10032;
        eg_mg.src_ip_addr = hdr.ip.src_addr;
        eg_mg.dst_ip_addr = hdr.ip.src_addr;
        eg_mg.ip_protocol = hdr.ip.protocol;
        eg_mg.src_tcp_port = hdr.tcp.sport;
        eg_mg.dst_tcp_port = hdr.tcp.dport;
    }

    table tab_init_metadata_32 {
        actions = {
            do_init_metadata_32;
        }
        size = 1;
    }

    action do_slash_of_ip_32(){
        // If we want to do it at different mask levels. 
        eg_mg.ip_addr = eg_mg.ip_addr & 0xffffffff ;
    }

    table tab_slash_of_ip_32 {
        actions = {
            do_slash_of_ip_32;
        }
        size = 1;
    }

    /*** Code for one hash function****/
    
    Hash<bit<16>>(HashAlgorithm_t.CRC32) hash32_tuple;

    action do_get_index_32_tuple(){
        eg_mg.tuple_register_index = hash32_tuple.get({
            eg_mg.src_ip_addr,
            eg_mg.dst_ip_addr,
            eg_mg.ip_protocol,
            eg_mg.src_tcp_port,
            eg_mg.dst_tcp_port
            });
    }

    table tab_get_index_32_tuple {
        actions = {
            do_get_index_32_tuple;
        }
        size = 1;
        const default_action = do_get_index_32_tuple;
    }


    Register<bit<8>, bit<16>>(table_size, 0) one_hash_register_table_32;
    // A simple one-bit register action that returns if that index has seen the value
    // stored in the register table.
    RegisterAction<bit<8>, bit<16>, bit<8>>(one_hash_register_table_32) one_hash_register_table_action_32 = {
        void apply(inout bit<8> val, out bit<8> rv) {
            rv = val;
            val = val + 1;
        }
    };

    /*** Code end for one hash function****/


    /*** Code for one hash function****/
    
    Hash<bit<16>>(HashAlgorithm_t.CRC32) hash32_tuple_2;

    action do_get_index_32_tuple_2(){
        eg_mg.tuple_register_index_2 = hash32_tuple_2.get({
            eg_mg.src_ip_addr,
            eg_mg.dst_ip_addr,
            eg_mg.ip_protocol,
            eg_mg.src_tcp_port,
            eg_mg.dst_tcp_port
            });
    }

    table tab_get_index_32_tuple_2 {
        actions = {
            do_get_index_32_tuple_2;
        }
        size = 1;
        const default_action = do_get_index_32_tuple_2;
    }


    Register<bit<8>, bit<16>>(table_size, 0) one_hash_register_table_32_2;
    // A simple one-bit register action that returns if that index has seen the value
    // stored in the register table.
    RegisterAction<bit<8>, bit<16>, bit<8>>(one_hash_register_table_32_2) one_hash_register_table_action_32_2 = {
        void apply(inout bit<8> val, out bit<8> rv) {
            rv = val;
            val = val + 1;
        }
    };

    /*** Code end for one hash function****/


    
    
    apply {
        bit<8> drop_pkt = 0;
        bit<8> ip_ctr = 0;
        bit<8> ip_ctr_2 = 0;
        bit<8> threshold = 0;
        bit<1> egress_global_timestamp = eg_intr_md.enq_tstamp[9:9];
        if(eg_intr_md.egress_port == 148){
            if(hdr.tcp.isValid()){
                tab_init_metadata_32.apply();
                tab_slash_of_ip_32.apply();
                tab_get_index_32_tuple.apply();
                tab_get_index_32_tuple_2.apply();
                ip_ctr = one_hash_register_table_action_32.execute(eg_mg.tuple_register_index);
                ip_ctr_2 = one_hash_register_table_action_32_2.execute(eg_mg.tuple_register_index_2);
                if(ip_ctr<ip_ctr_2){
                    threshold = ip_ctr;
                }else{
                    threshold = ip_ctr_2;
                }
                if(threshold < 40){
                    // we did not touch the threshold
                    eg_dprsr_md.drop_ctl = 1;
                }else{
                    // we touched the threshold and we do not drp the pkt.
                    eg_dprsr_md.drop_ctl = 0;
                }
            }else{
                eg_dprsr_md.drop_ctl = 1;
            }
        }        
    }
}


control EgressDeparser(
        packet_out pkt,
        inout header_t hdr, 
        in metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
    apply {
        pkt.emit(hdr);
    }
}
/*==============================================
=            The switch's pipeline             =
==============================================*/
Pipeline(
    IngressParser(), Ingress(), IngressDeparser(),
    EgressParser(), Egress(), EgressDeparser()) pipe;

Switch(pipe) main;