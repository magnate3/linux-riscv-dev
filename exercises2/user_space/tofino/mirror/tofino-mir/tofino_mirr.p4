#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

typedef bit<48> mac_t;
typedef bit<32> ip_address_t;
typedef bit<16> l4_port_t;
// const bit<48> MIRROR_TARGET_MAC = 0x08c0eb333174;   //188 1.3
#define ETH_TYPE_IPV4 0x0800
#define IP_VERSION_4 4w4
#define MAX_PORTS 511
#define IPV4_IHL_MIN 4w5
const bit<16> TYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_UDP = 0x11;
const bit<8>  IP_PROTO_TCP = 0x6;
const bit<16> INT_PORT = 5000; 
const bit<48> MIRROR_TARGET_MAC = 0x0090fb792055;   //184 2.1
const bit<16> INT_DATA_LEN = 4; 
const bit<8> REPORT_HDR_TTL = 64;
header ethernet_h{
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<6>  dscp;
    bit<2>  ecn;
    bit<16> len;
    bit<16> identification;
    bit<3>  flags;
    bit<13> frag_offset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdr_checksum;
    bit<32> src_addr;
    bit<32> dst_addr;
}

const bit<8> IPV4_MIN_HEAD_LEN = 20;


header udp_t {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> length_;
    bit<16> checksum;
}
header int_t {
    bit<32> val;
}
const bit<8> UDP_HEADER_LEN = 8;
struct ingress_headers_t {
    ethernet_h ethernet;
    // INT Report Encapsulation
    //ethernet_h                  report_ethernet;
    //ipv4_t                      report_ipv4;
    //udp_t                       report_udp;
    //int_t                     report_data;
}

struct ingress_metadata_t{
    MirrorId_t mirror_session;
    bit<8> mir_hdr_type;
}

parser IngressParser(
    packet_in pkt,
    out ingress_headers_t hdr,
    out ingress_metadata_t md,
    out ingress_intrinsic_metadata_t ig_intr_md
){
    state start{
        transition init_meta;
    }
    state init_meta{
        md.mirror_session=0;
        md.mir_hdr_type=0;
        transition parse_start;
    }
    state parse_start{
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }
    state parse_ethernet{
        pkt.extract(hdr.ethernet);
        transition accept;
    }
}

//const bit<4> ING_MIRROR = 7;
typedef bit<4> mirror_type_t;

const mirror_type_t MIRROR_TYPE_I2E = 1;
const mirror_type_t MIRROR_TYPE_E2E = 2;
const bit<8> MIRROR_HEADER_TYPE = 0xA2;

header mirror_h{
    bit<8> header_type;
}

control Ingress(
    inout ingress_headers_t hdr,
    inout ingress_metadata_t metadata,
    in ingress_intrinsic_metadata_t ig_intr_md,
    in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md
){
    action sendto(PortId_t port){
        ig_tm_md.ucast_egress_port=port;
    }
    action drop(){
        ig_dprsr_md.drop_ctl=1;
    }
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
    action ing_mirror(MirrorId_t mirs){
        ig_dprsr_md.mirror_type=MIRROR_TYPE_I2E;
        metadata.mirror_session=mirs;
        metadata.mir_hdr_type=MIRROR_HEADER_TYPE;
    }
     table ing_mirror_table{
        key={
            ig_intr_md.ingress_port: exact;
        }
        actions={
            ing_mirror;
            NoAction;
        }
        size=16;
        default_action=ing_mirror(8);
    }    
    apply{
          ing_mirror_table.apply();
          l2_forwarding.apply();
    }
}

control IngressDeparser(
    packet_out pkt,
    inout ingress_headers_t hdr,
    in ingress_metadata_t md,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md
){
    Mirror() ing_mirror;
    apply{
        if(ig_dprsr_md.mirror_type==MIRROR_TYPE_I2E){
            ing_mirror.emit<mirror_h>(
                md.mirror_session,
                {
                    md.mir_hdr_type
                }
            );
        }
        pkt.emit(hdr.ethernet);
    }
}

struct egress_headers_t{
    ethernet_h eth_hdr;
    ethernet_h                  report_ethernet;
    ipv4_t                      report_ipv4;
    udp_t                       report_udp;
    int_t                     report_data;
    
}

struct egress_metadata_t{
    bit<1> is_mirror;
}

parser EgressParser(
    packet_in pkt,
    out egress_headers_t hdr,
    out egress_metadata_t md,
    out egress_intrinsic_metadata_t eg_intr_md
){
    mirror_h tmp_mir_hdr;
    state start{
        transition init_meta;
    }
    state init_meta{
        md.is_mirror=0;
        transition parse_start;
    }
    state parse_start{
        pkt.extract(eg_intr_md);
        tmp_mir_hdr=pkt.lookahead<mirror_h>();
        transition select(tmp_mir_hdr.header_type){
            MIRROR_HEADER_TYPE:
                parse_mirror;
            default:
                parse_ethernet;
        }
    }
    state parse_mirror{
        pkt.extract(tmp_mir_hdr);
        md.is_mirror=1;
        tmp_mir_hdr.setInvalid();
        transition parse_ethernet;
    }
    state parse_ethernet{
        pkt.extract(hdr.eth_hdr);
        transition accept;
    }
}

control Egress(
    inout egress_headers_t hdr,
    inout egress_metadata_t metadata,
    in egress_intrinsic_metadata_t eg_intr_md,
    in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t eg_op_md
){
        action do_report_encapsulation(mac_t src_mac, mac_t mon_mac, ip_address_t src_ip, ip_address_t mon_ip, l4_port_t mon_port) {
        //Report Ethernet Header
        hdr.report_ethernet.setValid();
        hdr.report_ethernet.dst_addr = mon_mac;
        hdr.report_ethernet.src_addr = src_mac;
        hdr.report_ethernet.ether_type = ETH_TYPE_IPV4;

        //Report IPV4 Header
        hdr.report_ipv4.setValid();
        hdr.report_ipv4.version = IP_VERSION_4;
        hdr.report_ipv4.ihl = IPV4_IHL_MIN;
        hdr.report_ipv4.dscp = 6w0;
        hdr.report_ipv4.ecn = 2w0;

        // 20 + 8 + 16 + 14 + 20 + 8 + / 4 + 8 + (36 * #hops) / 84
        hdr.report_ipv4.len = (bit<16>) IPV4_MIN_HEAD_LEN + (bit<16>) UDP_HEADER_LEN +  INT_DATA_LEN;

        hdr.report_ipv4.identification = 0;
        hdr.report_ipv4.flags = 0;
        hdr.report_ipv4.frag_offset = 0;
        hdr.report_ipv4.ttl = REPORT_HDR_TTL;
        hdr.report_ipv4.protocol = IP_PROTO_UDP;
        hdr.report_ipv4.src_addr = src_ip;
        hdr.report_ipv4.dst_addr = mon_ip;
        ///Report UDP Header
        hdr.report_udp.setValid();
        hdr.report_udp.src_port = 1234;
        hdr.report_udp.dst_port = mon_port;
        hdr.report_udp.length_ = (bit<16>) UDP_HEADER_LEN + INT_DATA_LEN;
        hdr.report_data.val = 128;
        }
        table tb_generate_report {
         key = {
        }
        actions = {
            do_report_encapsulation;
            NoAction();
        }
        size = 1024;
        //default_action = do_report_encapsulation();
        //default_action = NoAction();
        default_action =  do_report_encapsulation(0, 0, 0, 0, 0);
       }
    apply{

        if(metadata.is_mirror==1){
             tb_generate_report.apply();
             hdr.eth_hdr.setInvalid();    
        }
    }
}

control EgressDeparser(
    packet_out pkt,
    inout egress_headers_t hdr,
    in egress_metadata_t md,
    in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md
){
    Checksum() ipv4Checksum;
    Checksum() udp_checksum;
    apply{
           hdr.report_ipv4.hdr_checksum = ipv4Checksum.update(
             {
                hdr.report_ipv4.version,
                hdr.report_ipv4.ihl,
                hdr.report_ipv4.dscp,
                hdr.report_ipv4.ecn,
                hdr.report_ipv4.len,
                hdr.report_ipv4.identification,
                hdr.report_ipv4.flags,
                hdr.report_ipv4.frag_offset,
                hdr.report_ipv4.ttl,
                hdr.report_ipv4.protocol,
                hdr.report_ipv4.src_addr,
                hdr.report_ipv4.dst_addr
             }
            );
           hdr.report_udp.checksum= udp_checksum.update(
             {
                hdr.report_ipv4.src_addr,
                hdr.report_ipv4.dst_addr,
                hdr.report_ipv4.protocol,
                hdr.report_udp.src_port,
                hdr.report_udp.dst_port,
                hdr.report_udp.length_
             }
            );
        pkt.emit(hdr.eth_hdr);
        pkt.emit(hdr.report_ethernet);
        pkt.emit(hdr.report_ipv4);
        pkt.emit(hdr.report_udp);
        pkt.emit(hdr.report_data);
    }
}

Pipeline(
    IngressParser(),Ingress(),IngressDeparser(),
    EgressParser(),Egress(),EgressDeparser()
) pipe;

Switch(pipe) main;
