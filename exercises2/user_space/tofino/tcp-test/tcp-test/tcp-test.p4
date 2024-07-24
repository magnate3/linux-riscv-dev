/* -*- P4_16 -*- */
//need to handle ARP
#include <core.p4>
#include <tna.p4>

#define SWITCH_ID 1
#define OUTPUT_PORT 16
#define RECIRCULATE_PORT 196
#define NEW_FLOW_THRESH 48000
#define MAXRTT 12000
#define LINKSPEED 1

/*************************************************************************
 ************* C O N S T A N T S    A N D   T Y P E S  *******************
*************************************************************************/
enum bit<16> ether_type_t {
    TPID       = 0x8100,
    IPV4       = 0x0800,
    ARP        = 0x0806,
    TRANS      = 0x2222
}

enum bit<8>  ip_proto_t {
    ICMP  = 1,
    IGMP  = 2,
    TCP   = 6,
    UDP   = 17
}


type bit<48> mac_addr_t;

/*************************************************************************
 ***********************  H E A D E R S  *********************************
 *************************************************************************/
/*  Define all the headers the program will recognize             */
/*  The actual sets of headers processed by each gress can differ */

/* Standard ethernet header */
header ethernet_h {
    bit<48>    dst_addr;
    bit<48>    src_addr;
    ether_type_t  ether_type;
}



header vlan_tag_h {
    bit<3>        pcp;
    bit<1>        cfi;
    bit<12>       vid;
    ether_type_t  ether_type;
}

header arp_h {
    bit<16>       htype;
    bit<16>       ptype;
    bit<8>        hlen;
    bit<8>        plen;
    bit<16>       opcode;
    bit<48>    hw_src_addr;
    bit<32>       proto_src_addr;
    bit<48>    hw_dst_addr;
    bit<32>       proto_dst_addr;
}

header ipv4_h {
    bit<4>       version;
    bit<4>       ihl;
    bit<7>       diffserv;
    bit<1>       res;
    bit<16>      total_len;
    bit<16>      identification;
    bit<3>       flags;
    bit<13>      frag_offset;
    bit<8>       ttl;
    bit<8>   protocol;
    bit<16>      hdr_checksum;
    bit<32>  src_addr;
    bit<32>  dst_addr;
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
    bit<4>   flags_before;
    bit<1>   flags_ack;
    bit<3>   flags_after;
    bit<16>  window;
    bit<16>  checksum;
    bit<16>  urgent_ptr;
}

header udp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<16>  len;
    bit<16>  checksum;
}

header INT_h {
    bit<32>  qlen;
    bit<32>  bandwidth;
    bit<32>  txBytes;
    bit<32>  tstamp;
    
}
header bridge_h {
    bit<32>  first_arrival_time;
    bit<32>  last_arrival_time;
    bit<32>  cache_length;
    bit<32>  base_RTT;
    bit<32> pkt_len;
    bit<16>  last_in_coco;
    bit<16>  port;
    

    
}
/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/
 
    /***********************  H E A D E R S  ************************/

struct my_ingress_headers_t{
    ethernet_h         ethernet;
    ipv4_h             ipv4;
    tcp_h              tcp;
    bridge_h           bridge;
}

struct tstamp_offset{
    bit<32> tstamp;
    bit<32> offset;
}
    /******  G L O B A L   I N G R E S S   M E T A D A T A  *********/


struct my_ingress_metadata_t {
    bit<32> flow_ID;
    bit<32> pkt_length;
    bit<16> coco_index;
    bit<16> timecm_port_offset;
    bit<16> coco_port_offset;
    bit<32> coco_time_offset;
    bit<32> cache_length;
    bit<32> cache_ID;
    bit<16> timecm_index_1;
    bit<16> timecm_index_2;
    bit<16> timecm_index_3;
    bit<32> lt_1;
    bit<32> lt_2;
    bit<32> lt_3;
    bit<32> ft_1;
    bit<32> ft_2;
    bit<32> ft_3;
    bit<32> arrival_tstamp;
    bit<32> last_tstamp;
    bit<1> flag_new_flow;
    bit<2> rtt_flag;
    bit<32> rand;
    bit<32> prob;
    PortId_t port;
    bit<4> pkt_length_mantissa;
    bit<8> pkt_length_exp;
    bit<4> rand_mantissa;
    bit<1> if_rep;
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
        transition meta_init;
    }

    state meta_init {
        
        transition parse_ethernet;
    }
    
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        /* 
         * The explicit cast allows us to use ternary matching on
         * serializable enum
         */        
        transition select((bit<16>)hdr.ethernet.ether_type) {
            (bit<16>)ether_type_t.IPV4            :  parse_ipv4;
            default :  accept;
        }
    }



    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        meta.pkt_length[15:0] = hdr.ipv4.total_len;
        transition select(hdr.ipv4.protocol) {
            6 : parse_tcp;
            default : accept;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        hdr.bridge.setValid();
        transition accept;
    }


}

 @pa_no_overlay("egress","meta.cache_length")
 @pa_solitary("egress","meta.cache_length")
@pa_container_size("egress","meta.cache_length",32)
 @pa_no_overlay("egress","meta.sum_cache_len")
 @pa_solitary("egress","meta.sum_cache_len")
@pa_container_size("egress","meta.sum_cache_len",32)
 @pa_no_overlay("egress","meta.read_index")
 @pa_solitary("egress","meta.read_index")
@pa_container_size("egress","meta.read_index",16)
 @pa_no_overlay("egress","meta.update_index")
 @pa_solitary("egress","meta.update_index")
@pa_container_size("egress","meta.update_index",16)

control Ingress(/* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{

    CRCPolynomial<bit<32>>(0x04C11DB7,false,false,false,32w0xFFFFFFFF,32w0xFFFFFFFF) crc32a;
    CRCPolynomial<bit<32>>(0x741B8CD7,false,false,false,32w0xFFFFFFFF,32w0xFFFFFFFF) crc32b;
    CRCPolynomial<bit<32>>(0xDB710641,false,false,false,32w0xFFFFFFFF,32w0xFFFFFFFF) crc32c;
    CRCPolynomial<bit<32>>(0x12345678,false,false,false,32w0xFFFFFFFF,32w0xFFFFFFFF) crc32d;
    CRCPolynomial<bit<32>>(0x87654321,false,false,false,32w0xFFFFFFFF,32w0xFFFFFFFF) crc32e;

    Hash<bit<32>>(HashAlgorithm_t.CUSTOM,crc32a) hash_ID;
    Hash<bit<16>>(HashAlgorithm_t.CUSTOM,crc32b) hash_t_1;
    Hash<bit<16>>(HashAlgorithm_t.CUSTOM,crc32c) hash_t_2;
    Hash<bit<16>>(HashAlgorithm_t.CUSTOM,crc32d) hash_t_3;
    Hash<bit<16>>(HashAlgorithm_t.CUSTOM,crc32e) hash_coco;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM,crc32e) hash_rand;
    

    Register<tstamp_offset, bit<32>>(1) tstamp_indicator_reg;
    RegisterAction<tstamp_offset, bit<32>, bit<32>>(tstamp_indicator_reg) operate_tstamp_indicator_reg =
    {
        void apply(inout tstamp_offset register_data, out bit<32> result) {
	        if (ig_intr_md.ingress_mac_tstamp[31:0] < register_data.tstamp)
            {
                register_data.tstamp = ig_intr_md.ingress_mac_tstamp[31:0];
                register_data.offset = register_data.offset + 1;
            }
            else if (ig_intr_md.ingress_mac_tstamp[31:0] > register_data.tstamp + MAXRTT)
            {
                register_data.tstamp = register_data.tstamp + MAXRTT;
                register_data.offset = register_data.offset + 1;
            }
            result = register_data.offset;
        }
    };


    action server_select_data(PortId_t port, bit<2> rtt_flag)
    {
        ig_tm_md.ucast_egress_port = port;
        meta.port = port;
        meta.arrival_tstamp = ig_intr_md.ingress_mac_tstamp[31:0];
        meta.rtt_flag = rtt_flag;
    }
    action server_select_ack(PortId_t port, bit<2> rtt_flag)
    {
        ig_tm_md.ucast_egress_port = port;
        meta.port = ig_intr_md.ingress_port;
        meta.arrival_tstamp = ig_intr_md.ingress_mac_tstamp[31:0];
    }
    @stage(0)  table server_select_t
    {   
        key={hdr.ipv4.dst_addr:exact;}
        actions={server_select_data;server_select_ack;}
        default_action=server_select_data(0,0);
    }

    action cal_flow_ID_a() {
        meta.flow_ID = hash_ID.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.ipv4.protocol,hdr.tcp.src_port,hdr.tcp.dst_port});
    }
    @stage(0) table cal_flow_ID_t {
        actions = { cal_flow_ID_a; }
        default_action = cal_flow_ID_a();
    }

    action get_random_a() {
        meta.rand = hash_rand.get({ig_intr_md.ingress_mac_tstamp});
    }
    @stage(0) table get_random_t {
        actions = { get_random_a; }
        default_action = get_random_a();
    }

    

    action get_pkt_length_a(bit<4> mantissa, bit<8> exp)
    {
        meta.pkt_length = meta.pkt_length + 14;
        meta.pkt_length_mantissa = mantissa;
        meta.pkt_length_exp = exp;
    }
    @stage(0)  table get_pkt_length_t
    {   
        key = {meta.pkt_length:exact;}
        actions={get_pkt_length_a;}
        default_action=get_pkt_length_a(0,0);
    }

    action split_R_a(bit<4> mantissa, bit<8> exp)
    {
        meta.rand_mantissa  = mantissa;
        meta.pkt_length_exp = meta.pkt_length_exp + exp;
        
    }
    @stage(1)  table split_R_t
    {   
        key = {meta.rand:ternary;}
        actions={split_R_a;}
        default_action=split_R_a(0,0);
    }

    action get_port_offset_a(bit<16> timecm_offset, bit<16> coco_offset)
    {
       meta.timecm_port_offset = timecm_offset;
       meta.coco_port_offset = coco_offset;
    }
    @stage(1)  table get_port_offset_t
    {   
        key={meta.port:exact;meta.coco_time_offset:exact; hdr.tcp.flags_ack:exact;}
        actions={get_port_offset_a;}
        default_action=get_port_offset_a(0,0);
    }

    action get_time_offset_a()
    {
       meta.coco_time_offset = operate_tstamp_indicator_reg.execute(0);
    }
    @stage(0)  table get_time_offset_t
    {   
        actions={get_time_offset_a;}
        default_action=get_time_offset_a;
        size=64;
    }

    action cal_prob_a(bit<32> prob)
    {
       meta.prob = prob |-| (bit<32>)meta.pkt_length;
    }
    @stage(2)  table cal_prob_t
    {   
        key = {meta.rand_mantissa:exact; meta.pkt_length_mantissa:exact; meta.pkt_length_exp:exact;}
        actions={cal_prob_a;}
        default_action=cal_prob_a(0);
    }
    
    action set_flow_ID_a() {
        meta.flow_ID[1:0] = meta.rtt_flag;
    }
    @stage(2) table set_flow_ID_t {
        actions = { set_flow_ID_a; }
        default_action = set_flow_ID_a();
    }

    
    action cal_c_index_a()
    {
       meta.coco_index = hash_coco.get({meta.flow_ID[31:2], meta.rtt_flag});
    }
    @stage(1)  table cal_c_index_t
    {   
        actions={cal_c_index_a;}
        default_action=cal_c_index_a();
    }

    // action merge_offset_a(bit<16> coco_time_offset)
    // {
    //    meta.coco_port_offset = meta.coco_port_offset + coco_time_offset;
    // }
    // @stage(2)  table merge_offset_t
    // {   
    //     key = {meta.coco_time_offset:exact; hdr.tcp.flags_ack:exact;}
    //     actions={merge_offset_a;}
    //     default_action=merge_offset_a(0);
    //}

    Register<bit<16>, PortId_t>(256) coco_index_reg;
    RegisterAction<bit<16>, PortId_t, bit<16>>(coco_index_reg) operate_coco_index_reg =
    {
        void apply(inout bit<16> register_data, out bit<16> result) {
            
            if (register_data < meta.coco_port_offset)
            {
                register_data = register_data + 1;
            }
            else if (register_data >= 24576)
            {
                register_data = register_data - 24576;
            }
            result = register_data;

        }
    };
    
    action read_coco_index_a()
    {   
        meta.coco_index = operate_coco_index_reg.execute(meta.port);
    }
    @stage(2)  table read_coco_index_t
    {   
        actions={read_coco_index_a;}
        default_action=read_coco_index_a();
    }
    action cal_t_index_1_a()
    {
       meta.timecm_index_1 = hash_t_1.get({meta.cache_ID});
    }
    @stage(5)  table cal_t_index_1_t
    {   
        actions={cal_t_index_1_a;}
        default_action=cal_t_index_1_a();
    }

    action cal_t_index_2_a()
    {
       meta.timecm_index_2 = hash_t_2.get({meta.cache_ID});
    }
    @stage(5)  table cal_t_index_2_t
    {   
        actions={cal_t_index_2_a;}
        default_action=cal_t_index_2_a();
    }
    
    action cal_t_index_3_a()
    {
       meta.timecm_index_3 = hash_t_3.get({meta.cache_ID});
    }
    @stage(5)  table cal_t_index_3_t
    {   
        actions={cal_t_index_3_a;}
        default_action=cal_t_index_3_a();
    }
    
    action get_t_index_a()
    {   
        meta.timecm_index_1 = meta.timecm_index_1 + meta.timecm_port_offset;
        meta.timecm_index_2 = meta.timecm_index_2 + meta.timecm_port_offset;
         meta.timecm_index_3 = meta.timecm_index_3 + meta.timecm_port_offset;
    }
    @stage(6)  table get_t_index_t
    {   
        actions={get_t_index_a;}
        default_action=get_t_index_a();
    }

    action get_c_index_a()
    {   
        meta.coco_index = meta.coco_index + meta.coco_port_offset;
    }
    @stage(2)  table get_c_index_t
    {   
        actions={get_c_index_a;}
        default_action=get_c_index_a();
    
    }
    


    Register<bit<32>, bit<16>>(32768) coco_length_reg;
    RegisterAction<bit<32>, bit<16>, bit<1>>(coco_length_reg) insert_coco_length_reg =
    {
        void apply(inout bit<32> register_data, out bit<1> result) {
            if (meta.prob >= register_data)
            {
                result = 1;
            }
            else
            {
                result = 0;
            }
            
            register_data = register_data + (bit<32>) meta.pkt_length;

        }
    };
    RegisterAction<bit<32>, bit<16>, bit<32>>(coco_length_reg) read_coco_length_reg =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
            
            result = register_data;
            register_data = 0;

        }
    };

     action insert_coco_length_a()
    {   
        meta.if_rep = insert_coco_length_reg.execute(meta.coco_index);
    }
    @stage(3)  table insert_coco_length_t
    {   
        actions={insert_coco_length_a;}
        default_action=insert_coco_length_a();
    }


    action read_coco_length_a()
    {   
        meta.cache_length = read_coco_length_reg.execute(meta.coco_index);
        meta.flow_ID = 0;
    }
    @stage(3)  table read_coco_length_t
    {   
        actions={read_coco_length_a;}
        default_action=read_coco_length_a();
    }



    Register<bit<32>, bit<16>>(32768) coco_ID_reg;
    RegisterAction<bit<32>, bit<16>, bit<32>>(coco_ID_reg) operate_coco_ID_reg =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
            
        
                result = register_data;
                register_data = meta.flow_ID;

        }
    };

    action insert_coco_ID_a()
    {   
        meta.cache_ID = operate_coco_ID_reg.execute(meta.coco_index);
    }
    @stage(4)  table insert_coco_ID_t
    {   
        actions={insert_coco_ID_a;}
        default_action=insert_coco_ID_a();
    }
     Register<bit<32>, bit<16>>(32768) timecm_last_reg_1;
    RegisterAction<bit<32>, bit<16>, bit<32>>(timecm_last_reg_1) operate_timecm_last_reg_1 =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
            if (hdr.tcp.flags_ack == 0 && meta.arrival_tstamp - register_data > NEW_FLOW_THRESH)
	        result = 1;
            else if (hdr.tcp.flags_ack != 0)
            result = register_data - 1;
            if (hdr.tcp.flags_ack == 0)
            register_data = meta.arrival_tstamp;
        }
    };

    Register<bit<32>, bit<16>>(32768) timecm_last_reg_2;
    RegisterAction<bit<32>, bit<16>, bit<32>>(timecm_last_reg_2) operate_timecm_last_reg_2 =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
	        if (hdr.tcp.flags_ack == 0 && meta.arrival_tstamp - register_data > NEW_FLOW_THRESH)
	        result = 1;
            else if (hdr.tcp.flags_ack != 0)
            result = register_data - 1;
            if (hdr.tcp.flags_ack == 0)
            register_data = meta.arrival_tstamp;
        }
    };

    Register<bit<32>, bit<16>>(32768) timecm_last_reg_3;
    RegisterAction<bit<32>, bit<16>, bit<32>>(timecm_last_reg_3) operate_timecm_last_reg_3 =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
	        if (hdr.tcp.flags_ack == 0 && meta.arrival_tstamp - register_data > NEW_FLOW_THRESH)
	        result = 1;
            else if (hdr.tcp.flags_ack != 0)
            result = register_data - 1;
            if (hdr.tcp.flags_ack == 0)
            register_data = meta.arrival_tstamp;
        }
    };

    action insert_timecm_last_1_a()
    {   
        meta.lt_1 = operate_timecm_last_reg_1.execute(meta.timecm_index_1);
    }
    @stage(7)  table insert_timecm_last_1_t
    {   
        actions={insert_timecm_last_1_a;}
        default_action=insert_timecm_last_1_a();
    }

    action insert_timecm_last_2_a()
    {   
        meta.lt_2 = operate_timecm_last_reg_2.execute(meta.timecm_index_2);
    }
    @stage(7)  table insert_timecm_last_2_t
    {   
        actions={insert_timecm_last_2_a;}
        default_action=insert_timecm_last_2_a();
    }
    action insert_timecm_last_3_a()
    {   
        meta.lt_3 = operate_timecm_last_reg_3.execute(meta.timecm_index_3);
    }
    @stage(7)  table insert_timecm_last_3_t
    {   
        actions={insert_timecm_last_3_a;}
        default_action=insert_timecm_last_3_a();
    }

    action set_new_flow_a()
    {   
        meta.flag_new_flow = 1;
    }
    @stage(8)  table set_new_flow_t
    {   
        actions={set_new_flow_a;}
        default_action=set_new_flow_a();
    }


   


    Register<bit<32>, bit<16>>(32768) timecm_first_reg_1;
    RegisterAction<bit<32>, bit<16>, bit<32>>(timecm_first_reg_1) operate_timecm_first_reg_1 =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
	        if (meta.flag_new_flow == 1)
            {
                register_data = meta.arrival_tstamp;
            }
            result = register_data;
        }
    };
    Register<bit<32>, bit<16>>(32768) timecm_first_reg_2;
    RegisterAction<bit<32>, bit<16>, bit<32>>(timecm_first_reg_2) operate_timecm_first_reg_2 =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
	        if (meta.flag_new_flow == 1)
            {
                register_data = meta.arrival_tstamp;
            }
            result = register_data;
        }
    };
    Register<bit<32>, bit<16>>(32768) timecm_first_reg_3;
    RegisterAction<bit<32>, bit<16>, bit<32>>(timecm_first_reg_3) operate_timecm_first_reg_3 =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
	        if (meta.flag_new_flow == 1)
            {
                register_data = meta.arrival_tstamp;
            }
            result = register_data;

        }
    };

    action insert_timecm_first_1_a()
    {   
        meta.ft_1 = operate_timecm_first_reg_1.execute(meta.timecm_index_1);
    }
    @stage(9)  table insert_timecm_first_1_t
    {   
        actions={insert_timecm_first_1_a;}
        default_action=insert_timecm_first_1_a();
    }

    action insert_timecm_first_2_a()
    {   
        meta.ft_2 = operate_timecm_first_reg_2.execute(meta.timecm_index_2);
    }
    @stage(9)  table insert_timecm_first_2_t
    {   
        actions={insert_timecm_first_2_a;}
        default_action=insert_timecm_first_2_a();
    }
    action insert_timecm_first_3_a()
    {   
        meta.ft_3 = operate_timecm_first_reg_3.execute(meta.timecm_index_3);
    }
    @stage(9)  table insert_timecm_first_3_t
    {   
        actions={insert_timecm_first_3_a;}
        default_action=insert_timecm_first_3_a();
    }


    action set_bridge_a(bit<32> RTT_offset, bit<16> final_flag)
    {   
        hdr.bridge.base_RTT = RTT_offset;
        hdr.bridge.last_in_coco = final_flag;
        hdr.bridge.pkt_len = meta.pkt_length;
        hdr.bridge.port = (bit<16>)ig_intr_md.ingress_port;

    }
    @stage(6)  table set_bridge_t
    {   
        key = {meta.coco_index:exact; meta.cache_ID[1:0]:exact;}
        actions={set_bridge_a;}
        default_action=set_bridge_a(0,0);
    }

    apply 
    {
        if (hdr.tcp.isValid())
        {   
            //stage 0
            get_pkt_length_t.apply();
            server_select_t.apply();
            cal_flow_ID_t.apply();
            get_random_t.apply();
            get_time_offset_t.apply();
            

            //stage 1
            get_port_offset_t.apply();
            cal_c_index_t.apply();
            split_R_t.apply();

            //stage 2
            cal_prob_t.apply();
            set_flow_ID_t.apply();
            if (hdr.tcp.flags_ack == 0)
            {
                //stage 2
                get_c_index_t.apply();
            }
            else
            {
                //stage 2
                read_coco_index_t.apply();
            }

            //stage 3
            if (hdr.tcp.flags_ack == 0)
                insert_coco_length_t.apply();
            else
                read_coco_length_t.apply();

            //stage 4
            if ((hdr.tcp.flags_ack == 0 && meta.if_rep == 1) || hdr.tcp.flags_ack == 1)
            {
                insert_coco_ID_t.apply();
            }
            else if (hdr.tcp.flags_ack == 0)
            {
                meta.cache_ID = meta.flow_ID;
            }

            //stage 5 6
            cal_t_index_1_t.apply();
            cal_t_index_2_t.apply();
            cal_t_index_3_t.apply();
            get_t_index_t.apply();    
            set_bridge_t.apply();

                if (meta.cache_ID != 0)
                {
                    insert_timecm_last_1_t.apply();
                    insert_timecm_last_2_t.apply();
                    insert_timecm_last_3_t.apply();
                    if (hdr.tcp.flags_ack == 1)
                    {
                        
                        hdr.bridge.last_arrival_time = min(meta.lt_1, meta.lt_2);
                        hdr.bridge.last_arrival_time = min(hdr.bridge.last_arrival_time, meta.lt_3);
                    }

                    if (hdr.tcp.flags_ack == 0 && (meta.lt_1[0:0] != 0 || meta.lt_2[0:0]!= 0 ||meta.lt_3[0:0] != 0))
                    {
                        set_new_flow_t.apply();
                    }
                    
                    insert_timecm_first_1_t.apply();
                    insert_timecm_first_2_t.apply();
                    insert_timecm_first_3_t.apply();
                    hdr.bridge.cache_length = meta.cache_length;
                    hdr.bridge.first_arrival_time = min(meta.ft_1, meta.ft_2);
                    hdr.bridge.last_arrival_time = min(hdr.bridge.first_arrival_time, meta.ft_3);
                }
                    
                
                
            // else
            // {

            // }
            

        }        
        
        
        

    }
}
control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
        // Checksum() ipv4_checksum;
    
    

    Checksum() ipv4_checksum;
    apply {
        if (hdr.ipv4.isValid()) {
            hdr.ipv4.hdr_checksum = ipv4_checksum.update({
                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.diffserv,
                hdr.ipv4.res,
                hdr.ipv4.total_len,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.frag_offset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.src_addr,
                hdr.ipv4.dst_addr
            });  
        }
    
        pkt.emit(hdr);
        
    }
}
/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  H E A D E R S  ************************/






    struct my_egress_headers_t{
    ethernet_h         ethernet;
    ipv4_h             ipv4;
    tcp_h              tcp;
    bridge_h           bridge;
    INT_h              INT;
    }



    /********  G L O B A L   E G R E S S   M E T A D A T A  *********/
struct sumlen_flag
{
    bit<32> sumlen;
    bit<32> flag;
}
struct my_egress_metadata_t {
    bit<32> flow_duration;
    bit<32> sum_cache_len;
    bit<32> rate_dt;
    bit<32> cache_length;
    bit<16> last_in_coco;
    bit<16> port;
    bit<32> past_rate;
    bit<16> update_index;
    bit<16> read_index;
    bit<16> counter_offset;
    bit<32> time_offset;
    bit<1> odd_or_even;
    bit<32> arrival_tstamp;
    //PowerTCP
    bit<32> QL_g;
    bit<32> QL_g_neg;
    bit<32> QL_g_flag;
    bit<32> pktlen_mod;
    bit<4> pktlen_mantissa;
    bit<8> pktlen_exp;
    bit<4> QL_g_mantissa;
    bit<8> QL_g_exp;
    bit<32> past_QL;
    bit<4> factor_1_mantissa; //R/(B-R)
    bit<8> factor_1_exp; //R/(B-R)
}

    /***********************  P A R S E R  **************************/

parser EgressParser(packet_in        pkt,
    /* User */
    out my_egress_headers_t          hdr,
    out my_egress_metadata_t         meta,
    /* Intrinsic */
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(eg_intr_md);
        transition parse_ethernet;
    }
    
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        /* 
         * The explicit cast allows us to use ternary matching on
         * serializable enum
         */        
        transition select((bit<16>)hdr.ethernet.ether_type) {
            (bit<16>)ether_type_t.IPV4            :  parse_ipv4;
            default :  accept;
        }
    }



    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            6 : parse_tcp;
            default : accept;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        hdr.INT.setValid();
        transition parse_bridge;
    }
    state parse_bridge {
        pkt.extract(hdr.bridge);
        meta.cache_length = hdr.bridge.cache_length;
        meta.last_in_coco = hdr.bridge.last_in_coco;
        meta.port = hdr.bridge.port;
        meta.pktlen_mod = hdr.bridge.pkt_len;
        transition accept;
    }
}

    /***************** M A T C H - A C T I O N  *********************/

control Egress(
    /* User */
    inout my_egress_headers_t                          hdr,
    inout my_egress_metadata_t                         meta,
    /* Intrinsic */    
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
    Hash<bit<16>>(HashAlgorithm_t.IDENTITY) hash_id1;
    Hash<bit<16>>(HashAlgorithm_t.IDENTITY) hash_id2;
    action get_update_index_a()
    {   
        meta.update_index = hash_id1.get({eg_prsr_md.global_tstamp[13:6]});
        meta.odd_or_even = eg_prsr_md.global_tstamp[5:5];
        meta.arrival_tstamp = eg_prsr_md.global_tstamp[31:0];
    }
    @stage(0)  table get_update_index_t
    {   
        actions={get_update_index_a;}
        default_action=get_update_index_a();
    }

    action get_offset_a(bit<16> counter_offset, bit<32> RTT_offset)
    {   
        meta.counter_offset = counter_offset;
        meta.time_offset = RTT_offset;
    }
    @stage(0)  table get_offset_t
    {   
        key = {eg_intr_md.egress_port:exact;}
        actions={get_offset_a;}
        default_action=get_offset_a(0, 0);
    }

    action get_past_tstamp_a()
    {   
        meta.arrival_tstamp = meta.arrival_tstamp - meta.time_offset;
    }
    @stage(1)  table get_past_tstamp_t
    {   
        actions={get_past_tstamp_a;}
        default_action=get_past_tstamp_a();
    }


    action get_read_index_a()
    {   
        meta.read_index = hash_id2.get({meta.arrival_tstamp[13:6]});
    }
    @stage(2)  table get_read_index_t
    {   
        actions={get_read_index_a;}
        default_action=get_read_index_a();
    }

    action get_index_a()
    {   
        meta.update_index = meta.update_index + meta.counter_offset;
        meta.read_index = meta.read_index + meta.counter_offset;
    }
    @stage(3)  table get_index_t
    {   
        actions={get_index_a;}
        default_action=get_index_a();
    }

    action cal_flow_duration_a()
    {   
        meta.flow_duration = hdr.bridge.last_arrival_time - hdr.bridge.first_arrival_time;
    }
    @stage(0)  table cal_flow_duration_t
    {   
        actions={cal_flow_duration_a;}
        default_action=cal_flow_duration_a();
    }

    action identify_dt_a()
    {   
        meta.flow_duration = meta.flow_duration |-| hdr.bridge.base_RTT;
    }
    @stage(1)  table identify_dt_t
    {   
        actions={identify_dt_a;}
        default_action=identify_dt_a();
    }


    Register<sumlen_flag, bit<16>>(256) sum_cached_packet_len_reg;
    RegisterAction<sumlen_flag, bit<16>, bit<32>>(sum_cached_packet_len_reg) operate_sum_cached_packet_len_reg =
    {
        void apply(inout sumlen_flag register_data, out bit<32> result) {
            if (register_data.flag == 1)
            result = register_data.sumlen;
            if (register_data.flag == 1)
            {
                register_data.flag = 0;
            }
            else if (meta.last_in_coco == 1)
            {
                register_data.flag = 1;
            }
            if (register_data.flag == 1)
            {
                register_data.sumlen = 1;
            }
            else
            {
                register_data.sumlen = register_data.sumlen + meta.cache_length;
            }

        }
    };

    
    action sum_cache_len_a()
    {   
        meta.sum_cache_len = operate_sum_cached_packet_len_reg.execute(meta.port);
    }
    @stage(3)  table sum_cache_len_t
    {   
        actions={sum_cache_len_a;}
        default_action=sum_cache_len_a();
    }

     action past_rate_cal_a(bit<32> past_rate)
    {   
        meta.past_rate = past_rate;
    }

    Register<bit<32>, bit<16>>(4096) QD_odd_reg;
    RegisterAction<bit<32>, bit<16>, bit<32>>(QD_odd_reg) operate_QL_odd_reg =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
            result = register_data;
            if (meta.odd_or_even == 1)
            {
                register_data = (bit<32>) eg_intr_md.deq_qdepth;
            }

        }
    };

    Register<bit<32>, bit<16>>(4096) QD_even_reg;
    RegisterAction<bit<32>, bit<16>, bit<32>>(QD_even_reg) operate_QL_even_reg =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
            result = register_data;
            if (meta.odd_or_even == 0)
            {
                register_data = (bit<32>) eg_intr_md.deq_qdepth;
            }

        }
    };

    @stage(4)  table past_rate_cal_t
    {   
        key = {meta.sum_cache_len:ternary;}
        actions={past_rate_cal_a;}
        default_action=past_rate_cal_a(0);
    }




    Register<bit<32>, bit<16>>(256) rate_dt_reg;

    MathUnit<bit<32>>(true,0,25,{68,73,78,85,93,102,113,128,0,0,0,0,0,0,0,0}) prog_decay_mu;
	 RegisterAction<bit<32>, bit<16>, bit<32>>(rate_dt_reg) decay_rate_dt_reg =
     {
		void apply(inout bit<32> register_data, out bit<32> result){
            result = register_data;
			register_data = prog_decay_mu.execute(register_data);
		}
	};
    RegisterAction<bit<32>, bit<16>, bit<32>>(rate_dt_reg) update_rate_dt_reg =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {

            register_data = register_data + meta.past_rate;
            result = register_data;
        }
    };

    action decay_rate_dt_a()
    {   
        meta.rate_dt = decay_rate_dt_reg.execute(meta.port);
    }
    @stage(6)  table decay_rate_dt_t
    {   
        actions={decay_rate_dt_a;}
        default_action=decay_rate_dt_a();
    }

     action update_rate_dt_a()
    {   
        meta.rate_dt = update_rate_dt_reg.execute(meta.port);
    }
    @stage(6)  table update_rate_dt_t
    {   
        actions={update_rate_dt_a;}
        default_action=update_rate_dt_a();
    }

    action update_QL_even_a()
    {   
        operate_QL_even_reg.execute(meta.update_index);
    }
    @stage(4)  table update_QL_even_t
    {   
        actions={update_QL_even_a;}
        default_action=update_QL_even_a();
    }

    action update_QL_odd_a()
    {   
        operate_QL_odd_reg.execute(meta.update_index);
    }
    @stage(4)  table update_QL_odd_t
    {   
        actions={update_QL_odd_a;}
        default_action=update_QL_odd_a();
    }

    action read_QL_even_a()
    {   
        meta.past_QL = operate_QL_even_reg.execute(meta.read_index);
    }
    @stage(4)  table read_QL_even_t
    {   
        actions={read_QL_even_a;}
        default_action=read_QL_even_a();
    }

    action read_QL_odd_a()
    {   
        meta.past_QL = operate_QL_odd_reg.execute(meta.read_index);
    }
    @stage(4)  table read_QL_odd_t
    {   
        actions={read_QL_odd_a;}
        default_action=read_QL_odd_a();
    }

    Register<bit<32>, bit<16>>(256) txBytes_reg;
    RegisterAction<bit<32>, bit<16>, bit<32>>(txBytes_reg) operate_txBytes_reg =
    {
        void apply(inout bit<32> register_data, out bit<32> result) {
            result = register_data;
            register_data = register_data + meta.pktlen_mod;

        }
    };
    

    action PUSH_INT_a()
    {   
        
        hdr.INT.txBytes = operate_txBytes_reg.execute((bit<16>)eg_intr_md.egress_port);
        hdr.INT.bandwidth = LINKSPEED;
        hdr.INT.qlen =(bit<32>)eg_intr_md.deq_qdepth;
        hdr.INT.tstamp = eg_prsr_md.global_tstamp[31:0];
        hdr.bridge.setInvalid();
    }
    @stage(11)  table PUSH_INT_t
    {   
        actions={PUSH_INT_a;}
        default_action=PUSH_INT_a();
    }


    action split_pktlen_a(bit<4> mantissa, bit<8> exp)
    {   
        meta.pktlen_mantissa = mantissa;
        meta.pktlen_exp = exp;
    }
    @stage(5)  table split_pktlen_t
    {   
        key = {meta.pktlen_mod:ternary;}
        actions={split_pktlen_a;}
        default_action=split_pktlen_a(0,0);
    }
    
    action cal_QL_g_a()
    {   
        meta.QL_g = (bit<32>)eg_intr_md.deq_qdepth - meta.past_QL;
        meta.QL_g_neg = meta.past_QL - (bit<32>)eg_intr_md.deq_qdepth;
        meta.QL_g_flag = (bit<32>)eg_intr_md.deq_qdepth |-| meta.past_QL;
    }
    @stage(5)  table cal_QL_g_t
    {   
        actions={cal_QL_g_a;}
        default_action=cal_QL_g_a;
    }

   


    action cal_step_1_a(bit<4> mantissa_1,bit<8> exp_1)
    {   
        meta.factor_1_mantissa = mantissa_1; 
        meta.factor_1_exp = exp_1;
    }
    @stage(7)  table cal_step_1_t
    {   
        key = {meta.rate_dt:ternary;}
        actions={cal_step_1_a;}
        default_action=cal_step_1_a(0,0);
    }


    action cal_for_QL_g_a(bit<4> mantissa, bit<8> exp)
    {   
        meta.QL_g_mantissa = mantissa;
        meta.QL_g_exp = exp;

    }
    @stage(7)  table cal_for_QL_g_t
    {   
        key = {meta.QL_g:ternary;}
        actions={cal_for_QL_g_a;}
        default_action=cal_for_QL_g_a(0,0);
    }

    action cal_for_pktlen_1_a(bit<4> mantissa, bit<8> exp)
    {   
        meta.QL_g_mantissa = mantissa;
        meta.QL_g_exp = meta.QL_g_exp + meta.factor_1_exp;
        meta.pktlen_exp = meta.pktlen_exp + exp;

    }
    @stage(8)  table cal_for_pktlen_1_t
    {   
        key = {meta.QL_g_mantissa:exact;meta.factor_1_mantissa:exact;}
        actions={cal_for_pktlen_1_a;}
        default_action=cal_for_pktlen_1_a(0,0);
    }


    action cal_for_pktlen_2_a(bit<4> mantissa, bit<8> exp)
    {   
        meta.pktlen_mantissa = mantissa;
        meta.pktlen_exp = meta.pktlen_exp + exp;

    }
    @stage(9)  table cal_for_pktlen_2_t
    {   
        key = {meta.pktlen_mantissa:exact;meta.QL_g_mantissa:exact;meta.QL_g_exp:exact;}
        actions={cal_for_pktlen_2_a;}
        default_action=cal_for_pktlen_2_a(0,0);
    }

    action reform_pktlen_mod_a(bit<32> pktlen_mod)
    {   
        meta.pktlen_mod = meta.pktlen_mod + pktlen_mod;
    }
    @stage(10)  table reform_pktlen_mod_t
    {   
        key = {meta.pktlen_mantissa:exact;meta.pktlen_exp:exact;meta.QL_g_flag:exact;}
        actions={reform_pktlen_mod_a;}
        default_action=reform_pktlen_mod_a(0);
    }


    apply 
    {
        if (hdr.tcp.isValid())
        {   
            
            
            
            //stage 0
            if (hdr.tcp.flags_ack == 0)
            {
                meta.port = (bit<16>) eg_intr_md.egress_port;
            }
            get_update_index_t.apply();
            get_offset_t.apply();
            cal_flow_duration_t.apply();

            //stage 1
            get_past_tstamp_t.apply();
            identify_dt_t.apply();

            //stage 2
            get_read_index_t.apply();
            get_index_t.apply();
            
            
            if (meta.flow_duration == 0)
            {
                meta.cache_length = 0;
            }
                    
            //stage 3
            sum_cache_len_t.apply();

            //stage 4
            if (hdr.tcp.flags_ack == 0)
            {
                if (meta.odd_or_even == 0)
                {
                    read_QL_odd_t.apply();
                    update_QL_even_t.apply();
                }
                else
                {
                    update_QL_odd_t.apply();
                    read_QL_even_t.apply();
                }
            }

            
            past_rate_cal_t.apply();
            
            // stage 5
            split_pktlen_t.apply();
            cal_QL_g_t.apply();

            //stage 6
            if (meta.QL_g_flag == 0)
            meta.QL_g = meta.QL_g_neg;
            
            if (meta.last_in_coco == 1)
            {
                decay_rate_dt_t.apply();
            }
            else
            {
                update_rate_dt_t.apply();
            }

            //stage 7
            if (eg_intr_md.deq_qdepth != 0)
            {
                cal_step_1_t.apply();
                cal_for_QL_g_t.apply();
                
                //stage 8
                
                cal_for_pktlen_1_t.apply();

                //stage 9
                cal_for_pktlen_2_t.apply();
                //stage 10
                reform_pktlen_mod_t.apply();
            }
            if (hdr.tcp.flags_ack == 0)
            PUSH_INT_t.apply();
            
        }
    }
}



    /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
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
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;

