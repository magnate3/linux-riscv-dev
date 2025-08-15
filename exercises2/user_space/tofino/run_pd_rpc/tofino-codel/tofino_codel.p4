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

#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif
#include "CoDel.p4"
#include "common/headers.p4"
//#include "common/util.p4"

/*************************************************************************
 ************* C O N S T A N T S    A N D   T Y P E S  *******************
*************************************************************************/
enum bit<16> ether_type_t {
    TPID = 0x8100,
    IPV4 = 0x0800,
    IPV6 = 0x86DD
}

enum bit<8>  ip_proto_t {
    ICMP  = 1,
    IGMP  = 2,
    TCP   = 6,
    UDP   = 17
}

typedef bit<48>   mac_addr_t;
typedef bit<32>   ipv4_addr_t;
typedef bit<128>  ipv6_addr_t;
typedef bit<96>  ipv6_prefix_t;

const bit<9> TO_SLOW_DP_PORT=56;
const bit<16> IPV4_HDR_SIZE=20;

//const bit<128> ipv6_prefix=0x20080000000000000000000000000000;
const bit<64> ipv6_prefix = 0x2008000000000000;

/*************************************************************************
 ***********************  H E A D E R S  *********************************
 *************************************************************************/
/*  Define all the headers the program will recognize             */
/*  The actual sets of headers processed by each gress can differ */

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
    bit<4>       version;
    bit<4>       ihl;
    bit<8>       diffserv;
    bit<16>      total_len;
    bit<16>      identification;
    bit<3>       flags;
    bit<13>      frag_offset;
    bit<8>       ttl;
    ip_proto_t   protocol;
    bit<16>      hdr_checksum;
    ipv4_addr_t  src_addr;
    ipv4_addr_t  dst_addr;
}

header option_word_h { /* Works for both IPv4 and TCP options */
    bit<32> data;
}

header ipv6_h {
    bit<4>       version;
    bit<8>       traffic_class;
    bit<20>      flow_label;
    bit<16>      payload_len;
    ip_proto_t   next_hdr;
    bit<8>       hop_limit;
    ipv6_addr_t  src_addr;
    ipv6_addr_t  dst_addr;
}

header icmp_h {
    bit<16>  type_code;
    bit<16>  checksum;
}

header igmp_h {
    bit<16>  type_code;
    bit<16>  checksum;
}

header tcp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<32>  seq_no;
    bit<32>  ack_no;
    bit<4>   data_offset;
    bit<4>   res;
    bit<8>   flags;
    bit<16>  window;
}

header tcp_checksum_h {
    bit<16>  checksum;
    bit<16>  urgent_ptr;
}

header udp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<16>  len;
}

header udp_checksum_h {
    bit<16>  checksum;
}


header bridged_metadata_t {
	bit<48> ingress_tstamp;
    bool  is_valid;
    bit<7>   pad;
    
}

struct metadata_t {
	bridged_metadata_t bridged_metadata;
    bit<1>        first_frag;
    bool        checksum_err_ipv4;
    bit<16>  l4_payload_checksum;
    bit<10> index;
    bit<1> ctr_exceeded;
}

struct header_t{
    ethernet_h         ethernet;
    vlan_tag_h[2]      vlan_tag;
    ipv4_h             ipv4;
    ipv6_h             ipv6;
}

// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------
parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t meta,
        out ingress_intrinsic_metadata_t ig_intr_md) {
    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }
     /* Packet parsing */
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ether_type_t.TPID :  parse_vlan_tag;
            ether_type_t.IPV4 :  parse_ipv4;
            ether_type_t.IPV6 :  parse_ipv6;
            default :  accept;
        }
    }

    state parse_vlan_tag {
        pkt.extract(hdr.vlan_tag.next);
        transition select(hdr.vlan_tag.last.ether_type) {
            ether_type_t.TPID :  parse_vlan_tag;
            ether_type_t.IPV4 :  parse_ipv4;
            ether_type_t.IPV4 :  parse_ipv6;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }

	state parse_ipv6 {
        pkt.extract(hdr.ipv6);
        transition accept;
    }
    
}

// ---------------------------------------------------------------------------
// Ingress
// ---------------------------------------------------------------------------
control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

       action l2_forward(bit<9> port){    
               ig_tm_md.ucast_egress_port  = port;
       }
       table l2_forwarding {
        key = {
             ig_intr_md.ingress_port: exact;
        }
        actions = {
            l2_forward;
            @defaultonly NoAction;
        }
        size = 1024;
        const default_action = NoAction();
       
    }
       action ipv4_forward(bit<9> port){    
               ig_tm_md.ucast_egress_port  = port;
       }
       action ipv6_forward(bit<9> port){    
               ig_tm_md.ucast_egress_port  = port;
       }
    table ipv4_lpm {
        key = {
            hdr.ipv4.dst_addr: exact;
        }
        actions = {
            ipv4_forward;
             @defaultonly NoAction;
        }
        size = 1024;
        const default_action = NoAction();
    }
        table ipv6_lpm {
        key = {
            hdr.ipv6.dst_addr: exact;
        }
        actions = {
            ipv6_forward;
            @defaultonly NoAction;
        }
        size = 1024;
        const default_action = NoAction();
    }
    apply {
        ig_md.bridged_metadata.setValid();
		ig_md.bridged_metadata.ingress_tstamp = ig_prsr_md.global_tstamp;
        if(TO_SLOW_DP_PORT == ig_intr_md.ingress_port)
        {
             if (hdr.ipv4.isValid()){
                 ipv4_lpm.apply();
             }
             else if (hdr.ipv6.isValid()){
                 ipv6_lpm.apply();
             }
             ig_md.bridged_metadata.is_valid = false;
        }
        else
        {
          l2_forwarding.apply();
          ig_md.bridged_metadata.is_valid = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------
control SwitchIngressDeparser(packet_out pkt,
                              inout header_t hdr,
                              in metadata_t meta,
                              in ingress_intrinsic_metadata_for_deparser_t 
                                ig_intr_dprsr_md
                              ) {
       apply {
        pkt.emit(meta.bridged_metadata);
        pkt.emit(hdr);
    }
}


/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/


parser EgressParser(packet_in      pkt,
    /* User */
    out header_t hdr,
    out metadata_t         meta,
    /* Intrinsic */
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(eg_intr_md);
        pkt.extract(meta.bridged_metadata);
        pkt.extract(hdr.ethernet);
        transition accept;
        //transition parse_ethernet;

    }
     /* Packet parsing */
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ether_type_t.TPID :  parse_vlan_tag;
            ether_type_t.IPV4 :  parse_ipv4;
            ether_type_t.IPV6 :  parse_ipv6;
            default :  accept;
        }
    }

    state parse_vlan_tag {
        pkt.extract(hdr.vlan_tag.next);
        transition select(hdr.vlan_tag.last.ether_type) {
            ether_type_t.TPID :  parse_vlan_tag;
            ether_type_t.IPV4 :  parse_ipv4;
            ether_type_t.IPV4 :  parse_ipv6;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }

	state parse_ipv6 {
        pkt.extract(hdr.ipv6);
        transition accept;
    }
}

    /***************** M A T C H - A C T I O N  *********************/

control Egress(
    /* User */
    inout header_t hdr,
    inout metadata_t         meta,
    /* Intrinsic */    
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
     CoDelEgress() codel_egress;

	apply{
    if(meta.bridged_metadata.is_valid)
	codel_egress.apply(meta.bridged_metadata.ingress_tstamp, 
						eg_prsr_md.global_tstamp,
						eg_intr_md.egress_port,
						eg_dprsr_md);
	}
}

    /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
    /* User */
    inout header_t hdr,
    in metadata_t         meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}
Pipeline(SwitchIngressParser(),
       SwitchIngress(),
       SwitchIngressDeparser(),
       EgressParser(),
       Egress(),
       EgressDeparser()) pipe;
Switch(pipe) main;
