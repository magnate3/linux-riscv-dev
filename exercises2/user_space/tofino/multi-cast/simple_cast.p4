/* -*- P4_16 -*- */
#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

typedef bit<9> egress_spec_t;
typedef bit<48> mac_addr_t;

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

header ethernet_t {
    mac_addr_t dst_addr;
    mac_addr_t src_addr;
    bit<16> ether_type;
}

header ip_t {
    bit<8> ver_hl; // 4 + 4
    bit<8> dscp_ecn; // tos, 6 + 2
    bit<16> length;
    bit<16> id;
    bit<16> flag_offset; // flag and offset of fragment, 3 + 13
    bit<8> ttl;
    bit<8> protocol;
    bit<16> checksum;
    bit<32> sip;
    bit<32> dip;
}

struct headers {
    ethernet_t ethernet;
    ip_t ip;
}

struct port_metadata_t {
    bit<16> unused; 
}

struct metadata {
    port_metadata_t port_metadata;
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser IngressParser(packet_in packet,
               out headers hdr,
               out metadata meta,
               out ingress_intrinsic_metadata_t ig_md) {

    state start {
        packet.extract(ig_md);
        transition select(ig_md.resubmit_flag) {
            0 : parse_port_metadata;
        }
    }

    state parse_port_metadata {
        meta.port_metadata = port_metadata_unpack<port_metadata_t>(packet);
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition accept;
    }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control Ingress(
        inout headers hdr,
        inout metadata meta,
        in ingress_intrinsic_metadata_t ig_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprs_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {
    
    action drop() {
        ig_dprs_md.drop_ctl = 0x1;
    }

    action l2_forward(PortId_t port,bit<7> qid,bit<32> credit ) {// 9 bit
        ig_dprs_md.drop_ctl = 0;
         ig_tm_md.qid =  qid;
        ig_tm_md.ucast_egress_port = port;
        //ig_dprs_md.adv_flow_ctl = 2785050624 +  credit;
        ig_dprs_md.adv_flow_ctl =  credit;
        // In doc of TNA, 192 is CPU PCIE port and 64~67 is CPU Ethernet ports for 2-pipe TF1
    }

    action l2_multicast(MulticastGroupId_t group) {// 16 bit
        ig_dprs_md.drop_ctl = 0;
        ig_tm_md.mcast_grp_a = group;
    }

    // action l2_forward_copy_to_cpu(bit<9> port) { // useless
    //     ig_dprs_md.drop_ctl = 0;
    //     ig_tm_md.ucast_egress_port = port;
    //     ig_tm_md.copy_to_cpu = 1;
    // }

    table l2_forward_table{
        key = {
            hdr.ethernet.dst_addr: exact;
        }
        actions = {
            l2_forward;
            l2_multicast;
            // l2_forward_copy_to_cpu;
            drop;
        }
        size = 32;
        default_action = drop();
    }

    apply {
        l2_forward_table.apply();
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control IngressDeparser(
        packet_out packet,
        inout headers hdr,
        in metadata meta,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {

    apply{
        packet.emit(hdr);
    }
}

parser EgressParser(packet_in packet,
               out headers hdr,
               out metadata meta,
               out egress_intrinsic_metadata_t eg_md) {
    state start {
        packet.extract(eg_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            0x0800: parse_ip;
            default: accept;
        }
    }

    state parse_ip{
        packet.extract(hdr.ip);
        transition accept;
    }
}

control dcqcn(
    inout headers hdr,
    in egress_intrinsic_metadata_t eg_md) {

    Wred<bit<19>, bit<32>>(32w1, 8w1, 8w0) wred;
    apply {
        if(hdr.ip.isValid()) {
            if(hdr.ip.dscp_ecn[1:0] == 0) { // Using "!=" and "&&" sometimes causes BUG
            }
            else {
                bit<8> drop_flag = wred.execute(eg_md.deq_qdepth, 0);
                if(drop_flag == 1) hdr.ip.dscp_ecn[1:0] = 3;
            }
        }
    }
}

control Egress(
        inout headers hdr,
        inout metadata meta,
        in egress_intrinsic_metadata_t eg_md,
        in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
        inout egress_intrinsic_metadata_for_deparser_t eg_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_oport_md) {
    action nop(){}
    action set_adv_flow_ctl(bit<32> credit ) {
           eg_dprs_md.adv_flow_ctl = credit; 
    }
    table flow_ctl{

        key     = { eg_md.egress_port: exact; }
        actions = {nop;  set_adv_flow_ctl; }
        const default_action = nop(); 
        size           = 64;
    }
    apply { 
        
        dcqcn.apply(hdr, eg_md);
        flow_ctl.apply();
        
    }
}

control EgressChecksum(inout headers hdr) {
    Checksum() csum;
    apply{
        hdr.ip.checksum = csum.update({
            hdr.ip.ver_hl,
            hdr.ip.dscp_ecn,
            hdr.ip.length,
            hdr.ip.id,
            hdr.ip.flag_offset,
            hdr.ip.ttl,
            hdr.ip.protocol,
            hdr.ip.sip,
            hdr.ip.dip
        });
    }
}

control EgressDeparser(packet_out packet,
                  inout headers hdr,
                  in metadata meta,
                  in egress_intrinsic_metadata_for_deparser_t ig_dprs_md) {
    
    apply { 
        EgressChecksum.apply(hdr);
        packet.emit(hdr);
    }
}

Pipeline(IngressParser(), Ingress(), IngressDeparser(), EgressParser(), Egress(), EgressDeparser()) pipe;

Switch(pipe) main;
