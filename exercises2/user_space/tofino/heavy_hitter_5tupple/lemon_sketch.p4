#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif
#include "common/headers.p4"
#include "common/util.p4"

#define T1 16384
#define T2 8192
#define T3 1024
#define T4 256
#define TH 256

#define layer1_size_bit1 4194304
#define layer2_size_bit1 2097152
#define layer3_size_bit1 524288
#define layer4_size_bit1 131072
#define layer5_size_bit1 524288

#define counter_size 131072
#define heavy_keeper 8192

#define l1_bitmap 32w8
#define l2_bitmap 32w32
#define l3_bitmap 32w64
#define l4_bitmap 32w64
#define l5_bitmap 32w512

#define TOPK_REG_SIZE 16
#define TOPK_REG_INDEX_WIDTH 4
//{hdr.ipv4.src_addr,hdr.ipv4.dst_addr,hdr.tcp.src_port,hdr.tcp.dst_port,hdr.ipv4.protocol}
#define FLOW_KEY {hdr.ipv4.src_addr}
#define PKG_KEY  {hdr.tcp.seq_no,hdr.udp.payload,hdr.tcp.checksum,hdr.udp.checksum,hdr.ipv4.src_addr,hdr.ipv4.dst_addr}
#define THRESHOLD  100


struct lemon_metadata_t {
    bit<32> threshold;
    bit<8> heavy_flag;

    bit<16> dhash;
    bit<32> shash;
    bit<16> bhash;

    bit<32> c_slot;
    bit<32> l1_slot;
    bit<32> l2_slot;
    bit<32> l3_slot;
    bit<32> l4_slot;
    bit<32> l5_slot;
    bit<32> heavy_slot;
    bit<8> tag;
}
struct metadata_t {
    lemon_metadata_t lemon;
}

// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------
parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {
    TofinoIngressParser() tofino_parser;

    state start {
        tofino_parser.apply(pkt, ig_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type){
            ETHERTYPE_IPV4 : parse_ipv4;
            ETHERTYPE_ARP  : parse_arp;
            ETHERTYPE_VLAN : parse_vlan;
            default : accept;
        }
    }

    state parse_vlan {
        pkt.extract(hdr.vlan_tag);
        transition select(hdr.vlan_tag.ether_type){
            ETHERTYPE_IPV4 : parse_ipv4;
            ETHERTYPE_ARP  : parse_arp;
            default : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol){
            IP_PROTOCOLS_TCP : parse_tcp;
            IP_PROTOCOLS_UDP : parse_udp;
            IP_PROTOCOLS_ICMP : parse_icmp;
            default : accept;
        }
    }

    state parse_arp {
        pkt.extract(hdr.arp);
        transition accept;
    }
    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }
    state parse_udp {
        pkt.extract(hdr.udp);
        transition accept;
    }
    state parse_icmp {
        pkt.extract(hdr.icmp);
        transition accept;
    }

}

// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------
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

    Register<bit<32>,_>(counter_size,0) counter_1;
    RegisterAction<bit<16>,_,bit<16>>(counter_1) counter_1_op = {
        void apply(inout bit<16> val, out bit<16> rv) {
            val = val + 1;
            rv = val;
        }
    };
    
    Register<bit<1>,_>(layer1_size_bit1,0) Layer1;
    RegisterAction<bit<1>,_,bit<1>>(Layer1) Layer1_op = {
        void apply(inout bit<1> val, out bit<1> rv) {
            val = 1;
        }
    };

    Register<bit<1>,_>(layer2_size_bit1,0) Layer2;
    RegisterAction<bit<1>,_,bit<1>>(Layer2) Layer2_op = {
        void apply(inout bit<1> val, out bit<1> rv) {
            val = 1;
        }
    };

    Register<bit<1>,_>(layer3_size_bit1,0) Layer3;
    RegisterAction<bit<1>,_,bit<1>>(Layer3) Layer3_op = {
        void apply(inout bit<1> val, out bit<1> rv) {
            val = 1;
        }
    };

    Register<bit<1>,_>(layer4_size_bit1,0) Layer4;
    RegisterAction<bit<1>,_,bit<1>>(Layer4) Layer4_op = {
        void apply(inout bit<1> val, out bit<1> rv) {
            val = 1;
        }
    };

    Register<bit<1>,_>(layer5_size_bit1,0) Layer5;
    RegisterAction<bit<1>,_,bit<1>>(Layer5) Layer5_op = {
        void apply(inout bit<1> val, out bit<1> rv) {
            val = 1;
        }
    };

    Register<bit<32>,_>(heavy_keeper,0) Heavy_sip;
    RegisterAction<bit<32>,_,bit<32>>(Heavy_sip) Heavy_sip_op = {
        void apply(inout bit<32> val, out bit<32> rv) {
            if(val == 0)
                val = hdr.ipv4.src_addr;
        }
    };


    //general hash
    //HashAlgorithm: IDENTITY, RANDOM, CRC8, CRC16, CRC32, CRC64, CUSTOM
    CRCPolynomial<bit<32>>(32w0x142ab0c9, // polynomial
                        true,          // reversed
                        false,         // use msb?
                        false,         // extended?
                        32w0xFFFFFFFF, // initial shift register value
                        32w0xFFFFFFFF  // result xor
                        ) polyl0;
    Hash<bit<16>>(HashAlgorithm_t.CUSTOM,polyl0) hash_deep; //13

    //Hash<bit<16>>(HashAlgorithm_t.CRC16) hash_l2; //13
    CRCPolynomial<bit<32>>(32w0x04a21Dc4, // polynomial
                           true,          // reversed
                           false,         // use msb?
                           false,         // extended?
                           32w0xFFFFFFFF, // initial shift register value
                           32w0xFFFFFFFF  // result xor
                           ) polyl1;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, polyl1) hash_slot;


    //Hash<bit<16>>(HashAlgorithm_t.CRC16) hash_l2; //13
    CRCPolynomial<bit<32>>(32w0x04ce1D56, // polynomial
                           true,          // reversed
                           false,         // use msb?
                           false,         // extended?
                           32w0xFFFFFFFF, // initial shift register value
                           32w0xFFFFFFFF  // result xor
                           ) polyl3;
    Hash<bit<16>>(HashAlgorithm_t.CUSTOM, polyl3) hash_bitmap;


    action return_tag1(){
        ig_md.lemon.tag = 1;
    }
    action return_tag2(){
        ig_md.lemon.tag = 2;
    }
    action return_tag3(){
        ig_md.lemon.tag = 1;
    }
    action return_tag4(){
        ig_md.lemon.tag = 1;
    }
    action return_tag5(){
        ig_md.lemon.tag = 1;
    }

    table lemon_match {
        key = {
            ig_md.lemon.dhash : range;
        }
        actions = {
            return_tag1;
            return_tag2;
            return_tag3;
            return_tag4;
            return_tag5;

        }
        const default_action = return_tag1;
        size = 8;
    }

    action compute_hash1(){
        // output SKETCH_INDEX_WIDTH bit hash result to index 
        ig_md.lemon.dhash = hash_deep.get(PKG_KEY);
        //ig_md.lemon.bhash = hash_bitmap.get(PKG_KEY);
        //ig_md.lemon.shash = hash_slot.get(FLOW_KEY);
    }
    action compute_hash2(){
        // output SKETCH_INDEX_WIDTH bit hash result to index 
        //ig_md.lemon.dhash = hash_deep.get(PKG_KEY);
        ig_md.lemon.bhash = hash_bitmap.get(PKG_KEY);
        //ig_md.lemon.shash = hash_slot.get(FLOW_KEY);
    }
    action compute_hash3(){
        // output SKETCH_INDEX_WIDTH bit hash result to index 
        //ig_md.lemon.dhash = hash_deep.get(PKG_KEY);
        //ig_md.lemon.bhash = hash_bitmap.get(PKG_KEY);
        ig_md.lemon.shash = hash_slot.get(FLOW_KEY);
    }

    apply {
        //port-forward
        if(ig_intr_md.ingress_port == 142){
            ig_tm_md.ucast_egress_port = 141;
            ig_tm_md.bypass_egress = 1;
        }
        if(ig_intr_md.ingress_port == 141){
            ig_tm_md.ucast_egress_port = 142;
            ig_tm_md.bypass_egress = 1;
        }
        compute_hash1();
        compute_hash2();
        compute_hash3();

        //ig_md.lemon.dhash = hash_deep.get(PKG_KEY);
        //ig_md.lemon.bhash = hash_bitmap.get(PKG_KEY);
        //ig_md.lemon.shash = hash_slot.get(FLOW_KEY);

        ig_md.lemon.c_slot[16:0] = ig_md.lemon.shash[16:0];
        ig_md.lemon.heavy_slot[12:0] = ig_md.lemon.shash[12:0];

        ig_md.lemon.l1_slot[21:3] = ig_md.lemon.shash[18:0];
        ig_md.lemon.l1_slot[2:0] = ig_md.lemon.bhash[2:0];

        ig_md.lemon.l2_slot[20:5] = ig_md.lemon.shash[15:0];
        ig_md.lemon.l2_slot[4:0] = ig_md.lemon.bhash[4:0];

        ig_md.lemon.l3_slot[17:5] = ig_md.lemon.shash[12:0];
        ig_md.lemon.l3_slot[4:0] = ig_md.lemon.bhash[4:0];

        ig_md.lemon.l4_slot[15:5] = ig_md.lemon.shash[10:0];
        ig_md.lemon.l4_slot[4:0] = ig_md.lemon.bhash[4:0];

        ig_md.lemon.l5_slot[18:9] = ig_md.lemon.shash[9:0];
        ig_md.lemon.l5_slot[8:0] = ig_md.lemon.bhash[8:0];

        counter_1_op.execute(ig_md.lemon.c_slot);
        lemon_match.apply();

        if(ig_md.lemon.tag == 1){
            Layer1_op.execute(ig_md.lemon.l1_slot);
        }
        if(ig_md.lemon.tag == 2){
            Layer2_op.execute(ig_md.lemon.l2_slot);
        }
        if(ig_md.lemon.tag == 3){
            Layer3_op.execute(ig_md.lemon.l3_slot);
        }
        if(ig_md.lemon.tag == 4){
            Layer4_op.execute(ig_md.lemon.l4_slot);
        }
        if(ig_md.lemon.tag == 5){
            Layer5_op.execute(ig_md.lemon.l5_slot);
            Heavy_sip_op.execute(ig_md.lemon.heavy_slot);
        }

    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;
