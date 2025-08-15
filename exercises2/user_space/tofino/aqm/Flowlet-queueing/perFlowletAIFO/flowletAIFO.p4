/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>
/*************************************************************************
************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/
const bit<16> BufferSize=25000;
const PortId_t OutputPort = 156;
const bit<16>  IndexLimit = 500;
const bit<32> round_add=1;

#define WORKER_PORT 9001
#define rank_range_threshold 150

typedef bit<8> ip_protocol_t;
const ip_protocol_t IP_PROTOCOLS_TCP = 6;
const ip_protocol_t IP_PROTOCOLS_UDP = 17;
const bit<16> ETHERTYPE_TPID = 0x8100;
const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<16> ETHERTYPE_WORKER= 0x4321;

/* Table Sizes */

#ifdef USE_ALPM
const int IPV4_LPM_SIZE  = 400*1024;
#else
const int IPV4_LPM_SIZE  = 12288;
#endif

#ifdef USE_STAGE
#define STAGE(n) @stage(n)
#else
#define STAGE(n)
#endif

typedef bit<48>   mac_addr_t;
typedef bit<32>   ipv4_addr_t;

/*************************************************************************
***********************  H E A D E R S  *********************************
*************************************************************************/

/*  Define all the headers the program will recognize             */
/*  The actual sets of headers processed by each gress can differ */

/* Standard ethernet header */
header ethernet_h {
   mac_addr_t    dst_addr;
   mac_addr_t    src_addr;
   bit<16>  ether_type;
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
   bit<8>       protocol;
   bit<16>      hdr_checksum;
   ipv4_addr_t  src_addr;
   ipv4_addr_t  dst_addr;
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
   bit<16>  checksum;
   bit<16>  urgent_ptr;
}

header udp_h {
   bit<16> src_port;
   bit<16> dst_port;
   bit<16> hdr_length;
   bit<16> checksum;
}

header worker_h {
    bit<16>     qlength;    // Queue occupancy in cells
    bit<32>     qid;
    bit<32>     round;      //virtual_time
    bit<32>     round_index;
}



/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/


   /***********************  H E A D E R S  ************************/

struct headers_t {
   ethernet_h          ethernet;
   ipv4_h              ipv4;
   tcp_h               tcp;
   udp_h               udp;
   worker_h            worker;
}

   /******  G L O B A L   I N G R E S S   M E T A D A T A  *********/

struct my_ingress_metadata_t {
   bit<32>      pkt_rank;
   bit<32>      flow_index;
   bit<32>      round;

   bit<32>       count_1;
   bit<32>       count_2;
   bit<32>       count_3;
   bit<32>       count_4;
   bit<32>       count_5;
   bit<32>       count_6;
   bit<32>       count_7;
   bit<32>       count_8;
   bit<32>       count_9;
   bit<32>       count_10;
   bit<32>       count_11;
   bit<32>       count_12;
   bit<32>       count_13;
   bit<32>       count_14;
   bit<32>       count_15;
   bit<32>       count_16;
   bit<8>      tail;

   bit<32>       count_1_2;
   bit<32>       count_3_4;
   bit<32>       count_5_6;
   bit<32>       count_7_8;
   bit<32>       count_9_10;
   bit<32>       count_11_12;
   bit<32>       count_13_14;
   bit<32>       count_15_16;
 
   bit<32>       count_1_2_3_4;
   bit<32>       count_5_6_7_8;
   bit<32>       count_9_10_11_12;
   bit<32>       count_13_14_15_16;

   bit<32>      count_1_to_8;
   bit<32>      count_9_to_16;

   bit<32>      count_all;
   bit<16>      queue_length;
   bit<16>      available_queue; //B-l
   bit<16>      left_side;
   bit<16>      aifo_admission;
   bit<32>      finish_time;
}

    /***********************  P A R S E R  **************************/

parser TofinoIngressParser(
       packet_in pkt,
       inout my_ingress_metadata_t meta,
       out ingress_intrinsic_metadata_t ig_intr_md) {
   state start {
       pkt.extract(ig_intr_md);
       transition select(ig_intr_md.resubmit_flag) {
           1 : parse_resubmit;
           0 : parse_port_metadata;
       }
   }

   state parse_resubmit {
       // Parse resubmitted packet here.
       pkt.advance(64);
       transition accept;
   }

   state parse_port_metadata {
       pkt.advance(64);  //tofino 1 port metadata size
       transition accept;
   }
}


parser EtherIPTCPUDPParser(packet_in        pkt,
   /* User */
   out headers_t          hdr
   )
{
   state start {
       transition parse_ethernet;
   }

   state parse_ethernet {
       pkt.extract(hdr.ethernet);
       transition select(hdr.ethernet.ether_type) {
           ETHERTYPE_IPV4 :  parse_ipv4;
           default :  accept;
       }
   }

   state parse_ipv4 {
       pkt.extract(hdr.ipv4);
       transition select(hdr.ipv4.protocol) {
           IP_PROTOCOLS_TCP : parse_tcp;
           IP_PROTOCOLS_UDP : parse_udp;
           default : accept;
       }
   }

   state parse_tcp {
       pkt.extract(hdr.tcp);
       transition accept;
   }

   state parse_udp {
       pkt.extract(hdr.udp);
       transition select(hdr.udp.dst_port) {
           WORKER_PORT: parse_worker;
           default: accept;
       }
   }

   state parse_worker {
       pkt.extract(hdr.worker);
       transition accept;
   }
}

parser SwitchIngressParser(
       packet_in pkt,
       out headers_t hdr,
       out my_ingress_metadata_t meta,
       out ingress_intrinsic_metadata_t ig_intr_md) {

   TofinoIngressParser() tofino_parser;
   EtherIPTCPUDPParser() layer4_parser;

   state start {
       tofino_parser.apply(pkt, meta, ig_intr_md);
       layer4_parser.apply(pkt, hdr);
       transition accept;
   }
}
   /***************** M A T C H - A C T I O N  *********************/

control Ingress(
   /* User */
   inout headers_t                       hdr,
   inout my_ingress_metadata_t                      meta,
   /* Intrinsic */
   in    ingress_intrinsic_metadata_t               ig_intr_md,
   in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
   inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
   inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{
    //table forward
    action forward(PortId_t port){
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
        ig_tm_md.ucast_egress_port = port;
        ig_tm_md.qid = 0;
    }
    action drop(){
        ig_dprsr_md.drop_ctl = 0x1;
    }

    table table_forward{
        key = {
            hdr.ipv4.dst_addr: exact;
            // hdr.wfq_t.isValid(): exact;
            // hdr.worker.isValid(): exact;
        }

        actions = {
            forward;
            drop;
        }
        const default_action = drop();   
        size = 512;
    }

    action get_weightindex_TCP(bit<32> flow_idx){
        meta.flow_index = flow_idx;      //flow_index
    }
    //table getweightTCP
    table get_weightindex_TCP_table{
        key = {
            hdr.ipv4.src_addr: exact;
            hdr.tcp.dst_port : exact;
        }
        actions = {
            get_weightindex_TCP;
        }   
        size = 512;
    }
    //counter meta.tail
    Register<bit<16>,_>(32w1) IndexReg;
    RegisterAction<bit<16>,_,bit<16>>(IndexReg) set_and_get_tail_reg = {
        void apply(inout bit<16> value,out bit<16> result){
            if(value == IndexLimit){
                result = 0;
                value = 0;
            }
            else{
                result = value;
                value = value + 1;
            }
        }
    };

    //timestamp:
    Register<bit<32>,bit<32>> (512) TimestampReg;
    RegisterAction<bit<32>,bit<32>,bit<32>> (TimestampReg) get_and_update_time = {
        void apply(inout bit<32> value,out bit<32> result){
            if (ig_prsr_md.global_tstamp[31:0]-value>500000){
                result = 1;
            }
            else{
                result = 0;
            }
            value = ig_prsr_md.global_tstamp[31:0]; //update timestamp always
        }
    };   

    //rank
    Register<bit<32>,bit<32>> (512) RankReg;
    RegisterAction<bit<32>,bit<32>,bit<32>> (RankReg) read_rank_reg = {
        void apply(inout bit<32> value,out bit<32> result){
            result=value;
        }
    };
    RegisterAction<bit<32>,bit<32>,bit<32>> (RankReg) write_rank_reg = {
        void apply(inout bit<32> value,out bit<32> result){
            value=meta.pkt_rank;
        }
    };

    //ingress round register
    Register<bit<32>,bit<5>> (32,0) Ingress_Round_Reg;
    RegisterAction<bit<32>,bit<5>,bit<32>> (Ingress_Round_Reg) set_ig_round_reg = {
        void apply(inout bit<32> value){
            value = hdr.worker.round;
        }
    };

    RegisterAction<bit<32>,bit<5>,bit<32>> (Ingress_Round_Reg) get_ig_round_reg = {
        void apply(inout bit<32> value,out bit<32> result){
            result = value;
        }
    };


    //f.finish_time register
    Register<bit<32>,bit<32>> (32w500,0) Packet_Sent_Reg;
    RegisterAction<bit<32>,bit<32>,bit<32>> (Packet_Sent_Reg) regact_update_and_get_f_finish_time2 = {
        void apply(inout bit<32> value,out bit<32> result){ 
            if(value >= meta.round){
                value = value + 2;
            }
            else{
                value = meta.round + 2; 
            }
            result = value;  
        }
    };
    
    RegisterAction<bit<32>,bit<32>,bit<32>> (Packet_Sent_Reg) regact_update_and_get_f_finish_time4 = {
        void apply(inout bit<32> value,out bit<32> result){ 
            if(value >= meta.round){
                value = value + 4;
            }
            else{
                value = meta.round + 4;
            }
            result = value;  
        }
    };

    RegisterAction<bit<32>,bit<32>,bit<32>> (Packet_Sent_Reg) regact_update_and_get_f_finish_time8 = {
        void apply(inout bit<32> value,out bit<32> result){ 
            if(value >= meta.round){
                value = value + 8;
            }
            else{
                value = meta.round + 8; 
            }
            result = value;  
        }
    };

    RegisterAction<bit<32>,bit<32>,bit<32>> (Packet_Sent_Reg) regact_update_and_get_f_finish_time16 = {
        void apply(inout bit<32> value,out bit<32> result){ 
            if(value >= meta.round){
                value = value + 16;
            }
            else{
                value = meta.round + 16; 
            }
            result = value;  
        }
    };


   // register to store the queue length (l)
    Register<bit<16>, bit<5>> (32,0) ig_queue_length_reg;
    RegisterAction<bit<16>, bit<5>, bit<16>>(ig_queue_length_reg) ig_queue_length_reg_write = {
       void apply(inout bit<16> value, out bit<16> read_value){
            value=hdr.worker.qlength;
            read_value = value;
       }
   };
   RegisterAction<bit<16>, bit<5>, bit<16>>(ig_queue_length_reg) ig_queue_length_reg_read = {
       void apply(inout bit<16> value, out bit<16> read_value){
               read_value = value;
       }
   };  

    action update_and_get_f_finish_time2(bit<32> flow_index) {
        meta.finish_time = regact_update_and_get_f_finish_time2.execute(flow_index);
    }

    action update_and_get_f_finish_time4(bit<32> flow_index) {
        meta.finish_time = regact_update_and_get_f_finish_time4.execute(flow_index);
    }

    action update_and_get_f_finish_time8(bit<32> flow_index) {
        meta.finish_time = regact_update_and_get_f_finish_time8.execute(flow_index);
    }

    action update_and_get_f_finish_time16(bit<32> flow_index) {
        meta.finish_time = regact_update_and_get_f_finish_time16.execute(flow_index);
    }

    table update_and_get_f_finish_time {
        key = {
            meta.flow_index: exact;
        }
        actions = {
            update_and_get_f_finish_time2;
            update_and_get_f_finish_time4;
            update_and_get_f_finish_time8;
            update_and_get_f_finish_time16;
        }
        size = 128;
    }

   action action_subtract_queueLength() {
           meta.available_queue=(bit<16>) (BufferSize - meta.queue_length);
       }

    table subtract_queueLength{
       actions = { action_subtract_queueLength;}
       default_action = action_subtract_queueLength();
       size=1;
   }

   action action_calculate_left_side(){
    meta.left_side = ((bit<16>)meta.count_all) << 10; //(rp-Min)*((1-k)*B)   (1-k)*B---2^14
   }

   table calculate_left_side{
       actions = { action_calculate_left_side;}
       default_action = action_calculate_left_side();
       size=1;
   }

   action action_do_AIFO_admission(){
       meta.aifo_admission = max(meta.left_side,meta.available_queue);
    }
    table AIFO_admission{
       actions = { action_do_AIFO_admission;}
       default_action = action_do_AIFO_admission();
       size=1;
   }

   action recirculation(bit<9> port){
       ig_tm_md.ucast_egress_port = port;
   }

   action worker_recirculate(){
       //packet routing: for now we simply bounce back the packet.
       //any routing match-action logic should be added here.
       ig_tm_md.ucast_egress_port=164;
   }

   action action_get_ig_queue_length(){
        meta.queue_length=ig_queue_length_reg_read.execute(0);

   }

   table get_ig_queue_length {
       actions = {
           action_get_ig_queue_length();
       }
       default_action =action_get_ig_queue_length();
       size = 1;
   }

   action action_set_ig_queue_length(){
        ig_queue_length_reg_write.execute(0);
   }

   table set_ig_queue_length {
       actions = {
           action_set_ig_queue_length();
       }
       default_action =action_set_ig_queue_length();
       size = 1;
   }

    //count
    Register<bit<32>,_> (32w1) count_test_1;
    RegisterAction<bit<32>,_,bit<32>> (count_test_1) check_win_reg1 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 0){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_2;
    RegisterAction<bit<32>,_,bit<32>> (count_test_2) check_win_reg2 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 1){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_3;
    RegisterAction<bit<32>,_,bit<32>> (count_test_3) check_win_reg3 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 2){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_4;
    RegisterAction<bit<32>,_,bit<32>> (count_test_4) check_win_reg4 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 3){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_5;
    RegisterAction<bit<32>,_,bit<32>> (count_test_5) check_win_reg5 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 4){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_6;
    RegisterAction<bit<32>,_,bit<32>> (count_test_6) check_win_reg6 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 5){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_7;
    RegisterAction<bit<32>,_,bit<32>> (count_test_7) check_win_reg7 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 6){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_8;
    RegisterAction<bit<32>,_,bit<32>> (count_test_8) check_win_reg8 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 7){
                value = meta.pkt_rank;
            }

        }
    };

    Register<bit<32>,_> (32w1) count_test_9;
    RegisterAction<bit<32>,_,bit<32>> (count_test_9) check_win_reg9 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 8){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_10;
    RegisterAction<bit<32>,_,bit<32>> (count_test_10) check_win_reg10 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 9){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_11;
    RegisterAction<bit<32>,_,bit<32>> (count_test_11) check_win_reg11 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 10){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_12;
    RegisterAction<bit<32>,_,bit<32>> (count_test_12) check_win_reg12 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 11){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_13;
    RegisterAction<bit<32>,_,bit<32>> (count_test_13) check_win_reg13 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 12){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_14;
    RegisterAction<bit<32>,_,bit<32>> (count_test_14) check_win_reg14 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 13){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_15;
    RegisterAction<bit<32>,_,bit<32>> (count_test_15) check_win_reg15 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 14){
                value = meta.pkt_rank;
            }

        }
    };
    Register<bit<32>,_> (32w1) count_test_16;
    RegisterAction<bit<32>,_,bit<32>> (count_test_16) check_win_reg16 = {
        void apply(inout bit<32> value,out bit<32> result){
            if(meta.pkt_rank > value){
                result = 1;
            }
            else{
                result = 0;
            }
            if(meta.tail == 15){
                value = meta.pkt_rank;
            }

        }
    }; 

    action add_count_1_2(){
        meta.count_1_2 = meta.count_1 + meta.count_2;
    }
    table add_count_1_2_table{
        actions = {
            add_count_1_2();
        }
        default_action = add_count_1_2();
        size = 1;
    }
    action add_count_3_4(){
        meta.count_3_4 = meta.count_3 + meta.count_4;
    }
    table add_count_3_4_table{
        actions = {
            add_count_3_4();
        }
        default_action = add_count_3_4();
        size = 1;
    }
    action add_count_5_6() {
            meta.count_5_6 = meta.count_5 + meta.count_6;
        }
        table add_count_5_6_table {
            actions = {
                add_count_5_6();
            }
            default_action = add_count_5_6();
            size = 1;
        }
    action add_count_7_8() {
            meta.count_7_8 = meta.count_7 + meta.count_8;
        }
        table add_count_7_8_table {
            actions = {
                add_count_7_8();
            }
            default_action = add_count_7_8();
            size = 1;
        }
    action add_count_9_10() {
            meta.count_9_10 = meta.count_9 + meta.count_10;
        }
        table add_count_9_10_table {
            actions = {
                add_count_9_10();
            }
            default_action = add_count_9_10();
            size = 1;
        }
    action add_count_11_12() {
            meta.count_11_12 = meta.count_11 + meta.count_12;
        }
        table add_count_11_12_table {
            actions = {
                add_count_11_12();
            }
            default_action = add_count_11_12();
            size = 1;
        }
    action add_count_13_14() {
            meta.count_13_14 = meta.count_13 + meta.count_14;
        }
        table add_count_13_14_table {
            actions = {
                add_count_13_14();
            }
            default_action = add_count_13_14();
            size = 1;
        }
    action add_count_15_16() {
            meta.count_15_16 = meta.count_15 + meta.count_16;
        }
        table add_count_15_16_table {
            actions = {
                add_count_15_16();
            }
            default_action = add_count_15_16();
            size = 1;
        }
    action add_count_1_2_3_4() {
            @in_hash {
                meta.count_1_2_3_4 = meta.count_1_2 + meta.count_3_4;
            }
        }
        table add_count_1_2_3_4_table {
            actions = {
                add_count_1_2_3_4();
            }
            default_action = add_count_1_2_3_4();
            size = 1;
        }
    action add_count_5_6_7_8() {
            @in_hash {
                meta.count_5_6_7_8 = meta.count_5_6 + meta.count_7_8;
            }
        }
        table add_count_5_6_7_8_table {
            actions = {
                add_count_5_6_7_8();
            }
            default_action = add_count_5_6_7_8();
            size = 1;
        }

    action add_count_9_10_11_12() {
            @in_hash {
                meta.count_9_10_11_12 = meta.count_9_10 + meta.count_11_12;
            }
        }
        table add_count_9_10_11_12_table {
            actions = {
                add_count_9_10_11_12();
            }
            default_action = add_count_9_10_11_12();
            size = 1;
        }

    action add_count_13_14_15_16() {
            @in_hash {
                meta.count_13_14_15_16 = meta.count_13_14 + meta.count_15_16;
            }
        }
        table add_count_13_14_15_16_table {
            actions = {
                add_count_13_14_15_16();
            }
            default_action = add_count_13_14_15_16();
            size = 1;
        }

    action add_count_1_to_8() {
            @in_hash {
                meta.count_1_to_8 = meta.count_1_2_3_4 + meta.count_5_6_7_8;
            }
        }
        table add_count_1_to_8_table {
            actions = {
                add_count_1_to_8();
            }
            default_action = add_count_1_to_8();
            size = 1;
        }
    action add_count_9_to_16() {
            @in_hash {
                meta.count_9_to_16 = meta.count_9_10_11_12 + meta.count_13_14_15_16;
            }
        }
        table add_count_9_to_16_table {
            actions = {
                add_count_9_to_16();
            }
            default_action = add_count_9_to_16();
            size = 1;
        }
    action add_count_all() {
            @in_hash {
                meta.count_all = meta.count_1_2_3_4;
            }
        }
        table add_count_all_table {
            actions = {
                add_count_all();
            }
            default_action = add_count_all();
            size = 1;
        }
        


   apply {
        if (hdr.ipv4.isValid()) {

            // do routing to get the egress port and qid
            table_forward.apply();
            meta.count_all = 0;
            if(hdr.worker.isValid()){
                    set_ig_queue_length.apply();
                    set_ig_round_reg.execute(0);
                    ig_dprsr_md.drop_ctl = 0;
                    worker_recirculate();
            }
            else if(hdr.tcp.isValid()){
                //get flow_index
                get_weightindex_TCP_table.apply();
                //get meta.tail
                meta.tail = (bit<8>)set_and_get_tail_reg.execute(0);
                //get round
                meta.round = get_ig_round_reg.execute(0);   
                //get rank
                update_and_get_f_finish_time.apply();
                bit<32> tmp= get_and_update_time.execute(meta.flow_index);
                if(tmp==1){
                    meta.pkt_rank=meta.finish_time;
                    write_rank_reg.execute(meta.flow_index);
                }
                else{
                    meta.pkt_rank=read_rank_reg.execute(meta.flow_index);
                }

                get_ig_queue_length.apply();
                //compute the actual queue length (B-L)
                subtract_queueLength.apply();

                //get quantile   meta.count_all num_wins that rank<pkt_rank 
                meta.count_1=check_win_reg1.execute(0);
                meta.count_2=check_win_reg2.execute(0);
                meta.count_3=check_win_reg3.execute(0);
                meta.count_4=check_win_reg4.execute(0);
                // meta.count_5=check_win_reg5.execute(0);
                // meta.count_6=check_win_reg6.execute(0);
                // meta.count_7=check_win_reg7.execute(0);
                // meta.count_8=check_win_reg8.execute(0);
                // meta.count_9=check_win_reg9.execute(0);
                // meta.count_10=check_win_reg10.execute(0);
                // meta.count_11=check_win_reg11.execute(0);
                // meta.count_12=check_win_reg12.execute(0);
                // meta.count_13=check_win_reg13.execute(0);
                // meta.count_14=check_win_reg14.execute(0);
                // meta.count_15=check_win_reg15.execute(0);
                // meta.count_16=check_win_reg16.execute(0); 

                add_count_1_2_table.apply();
                add_count_3_4_table.apply();
                // add_count_5_6_table.apply();
                // add_count_7_8_table.apply();
                // add_count_9_10_table.apply();
                // add_count_11_12_table.apply();
                // add_count_13_14_table.apply();
                // add_count_15_16_table.apply();

                add_count_1_2_3_4_table.apply();
                // add_count_5_6_7_8_table.apply();
                // add_count_9_10_11_12_table.apply();
                // add_count_13_14_15_16_table.apply();

                // add_count_1_to_8_table.apply();
                // add_count_9_to_16_table.apply();
                
                add_count_all_table.apply();

                //get left_side B*(1-k)/n*q
                calculate_left_side.apply();

                /* check AIFO admision condition */
                AIFO_admission.apply();

                // one condition for all
                // B*(1-k)/n*q >= B - l

                if (meta.aifo_admission == meta.left_side ) {
                    /* Drop this packet */
                    ig_dprsr_md.drop_ctl = 0x1;
                }
            }

       }
    }
}

   /*********************  D E P A R S E R  ************************/


control IngressDeparser(packet_out pkt,
   /* User */
   inout headers_t                       hdr,
   in    my_ingress_metadata_t                      meta,
   /* Intrinsic */
   in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    Checksum() ipv4_checksum;
   apply {

        if(hdr.ipv4.isValid()){
            // update the IPv4 checksum
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
            });
        }
        pkt.emit(hdr);
   }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

   /********  G L O B A L   E G R E S S   M E T A D A T A  *********/
struct my_egress_metadata_t {

}
   /***********************  P A R S E R  **************************/

parser EgressParser(packet_in        pkt,
   /* User */
   out headers_t          hdr,
   out my_egress_metadata_t         meta_eg,
   /* Intrinsic */
   out egress_intrinsic_metadata_t  eg_intr_md)
{
   /* This is a mandatory state, required by Tofino Architecture */
    EtherIPTCPUDPParser() layer4_parser;
    state start {
       pkt.extract(eg_intr_md);
       layer4_parser.apply(pkt,hdr);
       transition accept;
   }
}

   /***************** M A T C H - A C T I O N  *********************/

control Egress(
   /* User */
   inout headers_t                          hdr,
   inout my_egress_metadata_t                         meta_eg,
   /* Intrinsic */
   in    egress_intrinsic_metadata_t                  eg_intr_md,
   in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
   inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
   inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{

     //Engress round register
    Register<bit<32>,bit<5>> (32,0) Egress_Round_Reg;
    RegisterAction<bit<32>,bit<5>,bit<32>> (Egress_Round_Reg) set_eg_round_reg = {
        void apply(inout bit<32> value,out bit<32> result){
            value = value + round_add;
            result = value;
        }
    };

    RegisterAction<bit<32>,bit<5>,bit<32>> (Egress_Round_Reg) get_eg_round_reg = {
        void apply(inout bit<32> value,out bit<32> result){
            result = value;
        }
    };

    Register<bit<16>, _>(32w1) eg_queue_length_reg;
    RegisterAction<bit<16>, _, bit<16>>(eg_queue_length_reg) eg_queue_length_reg_write = {
       void apply(inout bit<16> value, out bit<16> read_value){
            value = eg_intr_md.deq_qdepth[15:0];
            read_value = value;
       }
   };
   RegisterAction<bit<16>, _, bit<16>>(eg_queue_length_reg) eg_queue_length_reg_read = {
       void apply(inout bit<16> value, out bit<16> read_value){
            read_value = value;
       }
   };

   apply {
	   if(hdr.worker.isValid()){
            hdr.worker.qlength = eg_queue_length_reg_read.execute(0);
            hdr.worker.round = get_eg_round_reg.execute(0);
  	    }
        else if (!hdr.worker.isValid() && eg_intr_md.egress_port == OutputPort){
            eg_queue_length_reg_write.execute(0);
            set_eg_round_reg.execute(0);
        }
   }
}
   /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
   /* User */
   inout headers_t                       hdr,
   in    my_egress_metadata_t                      meta_eg,
   /* Intrinsic */
   in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
   apply {
        pkt.emit(hdr);
        // pkt.emit(hdr.ethernet);
        // pkt.emit(hdr.ipv4);
        // pkt.emit(hdr.udp);
        // pkt.emit(hdr.rifoWorker);
   }
}

/************ F I N A L   P A C K A G E ******************************/
Pipeline(
   SwitchIngressParser(),
   Ingress(),
   IngressDeparser(),
   EgressParser(),
   Egress(),
   EgressDeparser()
) pipe;
Switch(pipe) main;
