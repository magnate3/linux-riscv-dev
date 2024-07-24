/* -*- P4_16 -*- */

#include <core.p4>
#include <tna.p4>

/*************************************************************************
 ************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/
const bit<16> ETHERTYPE_TPID = 0x8100;
const bit<16> ETHERTYPE_IPV4 = 0x0800;

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

header vlan_tag_h {
    bit<3>   pcp;
    bit<1>   cfi;
    bit<12>  vid;
    bit<16>  ether_type;
}

header ipv4_h {
    bit<4>   version;
    bit<4>   ihl;
    bit<8>   diffserv;
    bit<16>  total_len;
    bit<16>  identification;
    bit<3>   flags;
    bit<13>  frag_offset;
    bit<8>   ttl;
    bit<8>   protocol;
    bit<16>  hdr_checksum;
    bit<32>  src_addr;
    bit<32>  dst_addr;
}
@pa_container_size("ingress", "hdr.ipv4.dst_addr", 32)
/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/
 
    /***********************  H E A D E R S  ************************/

struct my_ingress_headers_t {
    ethernet_h   ethernet;
    vlan_tag_h   vlan_tag;
    ipv4_h       ipv4;
}

    /******  G L O B A L   I N G R E S S   M E T A D A T A  *********/

struct my_ingress_metadata_t {
    bit<16>  item6;
    bit<16>  item7;
    bit<16>  item8;
    bit<16>  item9;
    bit<16>  item10;
    bit<16>  item11;
    bit<16>  item12;
    bit<16>  item13;
}

    /***********************  P A R S E R  **************************/
parser IngressParser(packet_in        pkt,
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
            ETHERTYPE_TPID:  parse_vlan_tag;
            ETHERTYPE_IPV4:  parse_ipv4;
            default: accept;
        }
    }

    state parse_vlan_tag {
        pkt.extract(hdr.vlan_tag);
        transition select(hdr.vlan_tag.ether_type) {
            ETHERTYPE_IPV4:  parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }

}

    /***************** M A T C H - A C T I O N  *********************/

control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{
    /* The template type reflects the total width of the counter pair */
    DirectCounter<bit<64>>(CounterType_t.PACKETS_AND_BYTES) ipv4_host_stats;

    action action1(PortId_t port) {
      ig_tm_md.ucast_egress_port = port;
    }

      table tab1 {
      key = {
          ig_intr_md.ingress_port : exact;
      }
      actions = {
          action1;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 256;
      }    

    action action2(bit<12>  vid) {
       hdr.vlan_tag.vid= vid;
    }
    
      table tab2 {
      key = {
          ig_tm_md.ucast_egress_port: exact;
          hdr.vlan_tag.vid:  exact;
      }
      actions = {
          action2;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 1024;
      }    

    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }
    action action3(PortId_t port,bit<32>  dst_addr) {
        ig_tm_md.ucast_egress_port = port;
#       hdr.ipv4.dst_addr = dst_addr;
#ifdef BYPASS_EGRESS
        ig_tm_md.bypass_egress = 1;
#endif
   }
    table tab3{
        key     = { hdr.ipv4.dst_addr : lpm;hdr.vlan_tag.vid:  exact;}
        actions = { action3; @defaultonly NoAction; }
        
        const default_action = NoAction();
        size           = 12288;
    }
    
    action action4(bit<32>  src_addr) {
       hdr.ipv4.src_addr = src_addr;
   }
      table tab4 {
      key = {
          hdr.vlan_tag.vid:  exact;
          hdr.ipv4.dst_addr : lpm;
      }
      actions = {
          action4;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 1024;
      }    

    action action5() {
       hdr.ipv4.ttl= 64;
   }
      table tab5 {
      key = {
          hdr.ipv4.src_addr : lpm;
      }
      actions = {
          action5;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 1024;
      }    
    action action6() {
       meta.item6 = 64;
   }
      table tab6 {
      key = {
          hdr.ipv4.ttl: exact;
      }
      actions = {
          action6;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
    action action7() {
       meta.item7 = 64;
   }
      table tab7 {
      key = {
          meta.item6: exact;
      }
      actions = {
          action7;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
    action action8() {
       meta.item8 = 64;
   }
      table tab8 {
      key = {
          meta.item7: exact;
      }
      actions = {
          action8;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
    action action9() {
       meta.item9 = 64;
   }
      table tab9 {
      key = {
          meta.item8: exact;
      }
      actions = {
          action9;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
    action action10() {
       meta.item10 = 64;
   }
      table tab10 {
      key = {
          meta.item9: exact;
      }
      actions = {
          action10;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
    action action11() {
       meta.item11 = 64;
   }
      table tab11 {
      key = {
          meta.item10: exact;
      }
      actions = {
          action11;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
    action action12() {
       meta.item12 = 64;
   }
      table tab12 {
      key = {
          meta.item11: exact;
      }
      actions = {
          action12;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
#if 0
    action action13() {
       meta.item13 = 64;
   }
      table tab13 {
      key = {
          meta.item12: exact;
      }
      actions = {
          action13;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
#endif
    apply {
                tab1.apply();
                tab2.apply();
                tab3.apply();
                tab4.apply();
                tab5.apply();
                tab6.apply();
                tab7.apply();
                tab8.apply();
                tab9.apply();
                tab10.apply();
                tab11.apply();
                tab12.apply();
                //tab13.apply();
    }
}

    /*********************  D E P A R S E R  ************************/

control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    apply {
        pkt.emit(hdr);
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
    bit<16>  out_item1;
    bit<16>  out_item2;
}

    /***********************  P A R S E R  **************************/

parser EgressParser(packet_in        pkt,
    /* User */
    out my_ingress_headers_t          hdr,
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

control Egress(
    /* User */
    inout my_ingress_headers_t                          hdr,
    inout my_egress_metadata_t                         meta,
    /* Intrinsic */    
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
#if 1
    action action14() {
       //meta.out_item1 = 64;
       hdr.ipv4.ttl= 64;
   }
      table tab14 {
      key     = { hdr.ipv4.dst_addr : lpm;hdr.vlan_tag.vid:  exact;}
      actions = {
          action14;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }    
#endif
    apply {
        tab14.apply();
    }
}

    /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                      hdr,
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
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;
