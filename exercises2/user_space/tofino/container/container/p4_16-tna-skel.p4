/* -*- P4_16 -*- */

#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

/*************************************************************************
 ************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/

enum bit<8>  ip_proto_t {
    ICMP  = 1,
    IGMP  = 2,
    TCP   = 6,
    UDP   = 17
}
enum bit<16> ether_type_t {
    TPID = 0x8100,
    IPV4 = 0x0800, 
    IPV6 = 0x86DD
} 
 

typedef bit<128>  ipv6_addr_t;
typedef bit<32>   ipv4_addr_t;
const bit<16> IPV4_HDR_SIZE=20;
/*************************************************************************
 ***********************  H E A D E R S  *********************************
 *************************************************************************/

/*  Define all the headers the program will recognize             */
/*  The actual sets of headers processed by each gress can differ */

/* Standard ethernet header */
header ethernet_h {
    bit<48>   dst_addr;
    bit<48>   src_addr;
    bit<16>   ether_type;
}

header ipv6_h {
    bit<4>       version;
    bit<8>       traffic_class;
    bit<20>      flow_label;
#if 0
    bit<8>       traffic_class1;
    bit<8>       traffic_class2;
    bit<8>       traffic_class3;
    bit<8>       traffic_class4;
    bit<8>       traffic_class5;
    bit<8>       traffic_class6;
    bit<8>       traffic_class7;
    bit<8>       traffic_class8;
    bit<8>       traffic_class9;
    bit<8>       traffic_class10;
    bit<8>       traffic_class11;
    bit<8>       traffic_class12;
    bit<8>       traffic_class13;
    bit<8>       traffic_class14;
    bit<8>       traffic_class15;
    bit<8>       traffic_class16;
    bit<8>       traffic_class17;
    bit<8>       traffic_class18;
    bit<8>       traffic_class19;
    bit<8>       traffic_class20;
    bit<8>       traffic_class21;
    bit<8>       traffic_class22;
    bit<8>       traffic_class23;
#endif
    bit<16>      payload_len;
    ip_proto_t   next_hdr;
    bit<8>       hop_limit;
    ipv6_addr_t  src_addr;
    ipv6_addr_t  dst_addr;
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
/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  H E A D E R S  ************************/

struct my_ingress_headers_t {
    ethernet_h   ethernet;
    ipv6_h       ip6;
    ipv4_h       ip4;
}
//#if __TARGET_TOFINO__ == 2
#if 1
@pa_container_type ("ingress", "hdr.ip6.traffic_class", "normal")
@pa_container_size("ingress", "hdr.ip6.traffic_class", 16)
 // pa_container_size =8 will has compling bug
//@pa_container_size("ingress", "hdr.ip6.traffic_class", 8)
@pa_container_type ("ingress", "hdr.ip6.hop_limit", "normal")
//@pa_container_type ("ingress", "hdr.ip6.hop_limit", "tagalong")
@pa_container_size("ingress", "hdr.ip6.hop_limit", 8)
@pa_container_size("ingress", "hdr.ip4.ttl", 8)
#else
@pa_container_size("ingress", "hdr.ip6.traffic_class", 8)
@pa_mutually_exclusive("ingress", "hdr.ip6.traffic_class", "hdr.ip6.hop_limit")
#endif

    /******  G L O B A L   I N G R E S S   M E T A D A T A  *********/

struct my_ingress_metadata_t {
}

    /***********************  P A R S E R  **************************/
parser MyIngressParser(packet_in        pkt,
    /* User */
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    /* Intrinsic */
    out ingress_intrinsic_metadata_t  ig_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
     state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
       pkt.extract(hdr.ethernet);
       transition select(hdr.ethernet.ether_type) {
            ether_type_t.IPV4 :  parse_ipv4;
            ether_type_t.IPV6 :  parse_ipv6;
            default :  accept;
        }
    }
    state parse_ipv4 {
        pkt.extract(hdr.ip4);
        transition accept;
    }
    state parse_ipv6 {
        pkt.extract(hdr.ip6);
        transition accept;
    }
}

    /***************** M A T C H - A C T I O N  *********************/

control MyIngress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{
    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }

   
    action ipv4_forward(bit<48> dstAddr,bit<9> port){
        //set the src mac address as the previous dst
        hdr.ethernet.src_addr = hdr.ethernet.dst_addr;

       //set the destination mac address that we got from the match in the table
        hdr.ethernet.dst_addr = dstAddr;

        //set the output port that we also get from the table
        /* standard_metadata.egress_spec = port; */
        ig_tm_md.ucast_egress_port = port;

        hdr.ip6.version =   6;
        hdr.ip6.traffic_class = 64;
        hdr.ip6.flow_label = 0;
        hdr.ip6.payload_len = hdr.ip4.total_len -  IPV4_HDR_SIZE;
        hdr.ip6.next_hdr = hdr.ip4.protocol;
        hdr.ip6.hop_limit = hdr.ip4.ttl + 64;
        hdr.ip6.hop_limit = hdr.ip4.ttl *2;
    }
    action ipv6_forward(bit<48> dstAddr,bit<9> port){
        //set the src mac address as the previous dst
        hdr.ethernet.src_addr = hdr.ethernet.dst_addr;

       //set the destination mac address that we got from the match in the table
        hdr.ethernet.dst_addr = dstAddr;

        //set the output port that we also get from the table
        /* standard_metadata.egress_spec = port; */
        ig_tm_md.ucast_egress_port = port;

        hdr.ip4.version = 4;
        hdr.ip4.ihl = 5;
        hdr.ip4.diffserv = 0;
        hdr.ip4.identification = 1;
        hdr.ip4.protocol = hdr.ip6.next_hdr;
        hdr.ip4.ttl = hdr.ip6.hop_limit;
        hdr.ip4.total_len= hdr.ip6.payload_len + IPV4_HDR_SIZE;
    }
        table ipv4_lpm {
        key = {
            hdr.ip4.dst_addr: lpm;
        }
        actions = {
            ipv4_forward();
            drop();
        }
        default_action = drop();
    }
        table ipv6_lpm {
        key = {
            hdr.ip6.dst_addr: lpm;
        }
        actions = {
            ipv6_forward();
            drop();
        }
        default_action = drop();
    }
    apply {
          ipv4_lpm.apply();
    }
}

    /*********************  D E P A R S E R  ************************/

control MyIngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ip6);
    }
}


/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  H E A D E R S  ************************/

struct my_egress_headers_t {
}

    /********  G L O B A L   E G R E S S   M E T A D A T A  *********/

struct my_egress_metadata_t {
}

    /***********************  P A R S E R  **************************/

parser MyEgressParser(packet_in        pkt,
    /* User */
    out my_egress_headers_t          hdr,
    out my_egress_metadata_t         meta,
    /* Intrinsic */
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

    /***************** M A T C H - A C T I O N  *********************/

control MyEgress(
    /* User */
    inout my_egress_headers_t                          hdr,
    inout my_egress_metadata_t                         meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
    apply {
    }
}

    /*********************  D E P A R S E R  ************************/

control MyEgressDeparser(packet_out pkt,
    /* User */
    inout my_egress_headers_t                       hdr,
    in    my_egress_metadata_t                      meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}


/************ F I N A L   P A C K A G E ******************************/
Pipeline(
    MyIngressParser(),
    MyIngress(),
    MyIngressDeparser(),
    MyEgressParser(),
    MyEgress(),
    MyEgressDeparser()
) pipe;
Switch(pipe) main;

