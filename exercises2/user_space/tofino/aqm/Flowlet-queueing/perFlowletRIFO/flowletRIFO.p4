/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>
/*************************************************************************
************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/
const bit<20> BufferSize=524289;
const PortId_t OutputPort = 156;
const bit<16>  CounterLimit = 50;
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
    bit<32>     qlength;    // Queue occupancy in cells
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
//    bit<32>      weight;
   bit<32>      round;
   bit<32>      finish_time;

   bit<20>      queue_length;
   bit<20>      available_queue; //B-l
   bit<32>      min_pkt_rank;
   bit<32>      max_pkt_rank;
   bit<16>      dividend; //rp-Min
   bit<16>      divisor;  //Max-Min
   bit<24>      left_side; //(rp-Min)*(1-k)*B
   bit<24>      right_side; //(B-l)*(Max-Min)
   bit<24>      rifo_admission;
   bit<32>      rank_range;
   bit<16>      max_min;
   bit<8>       max_min_exponent; //Max-Min
   bit<8>       buffer_exponent; //B-l
   bit<8>       dividend_exponent; //rp-Min

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

    //   //table for round index:
    // action get_workerround_index_action(bit<32> round_index){
    //     hdr.worker.round_index = round_index;
    // }
    // table get_workerround_index_table{
    //     key = {
    //         hdr.worker.egress_port: exact;
    //         hdr.worker.qid: exact;
    //     }
    //     actions = {
    //         get_workerround_index_action;
    //     }
    //     size = 128;
    // }
    // //table for get round index (not worker):
    // action get_round_index_action(bit<32> flow_round_index){
    //     meta.flow_round_index = flow_round_index; 
    // }
    // table get_flow_round_index_table{
    //     key = {
    //         ig_tm_md.ucast_egress_port:exact;
    //         ig_tm_md.qid:exact;
    //     }
    //     actions = {
    //         get_round_index_action;
    //     }
    //     size = 128;
    // }

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


    // //table getweightUDP
    // action get_weightindex_UDP(bit<32> flow_idx){
    //     meta.flow_index = flow_idx;      //flow_index
    // }
    // table get_weightindex_UDP_table{
    //     key = {
    //         hdr.ipv4.src_addr: exact;
    //         hdr.udp.dst_port : exact;
    //     }
    //     actions = {
    //         get_weightindex_UDP;
    //     }   
    //     size = 512;
    // }


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

    // //Get weight ,1/wf
    // action get_weight_action(bit<32> weight) {
    //     meta.weight = weight;      
    // }
    // table get_weight_table {
    //     key = {
    //         meta.flow_index:exact;
    //     }
    //     actions = {
    //         get_weight_action;
    //     }  
    //     size = 512;
    // }

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
    RegisterAction<bit<32>,bit<32>,bit<32>> (Packet_Sent_Reg) regact_update_and_get_f_finish_time32 = {
        void apply(inout bit<32> value,out bit<32> result){ 
            if(value >= meta.round){
                value = value + 32;
            }
            else{
                value = meta.round + 32; 
            }
            result = value;  
        }
    };

    //timestamp:
    Register<bit<32>,bit<32>> (512,0) TimestampReg;
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

    //counter
    Register<bit<16>,_>(32w1) countReg;
    RegisterAction<bit<16>,_,bit<16>>(countReg) set_counter_reg = {
        void apply(inout bit<16> value,out bit<16> result){
            if(value == CounterLimit){
                result = 1;
                value = 1;
            }
            else{
                result = 0;
                value = value + 1;
            }
        }
    };

    //rank
    Register<bit<32>,bit<32>> (512,0) RankReg;
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

   // register to store the queue length (l)
    Register<bit<32>, bit<5>> (32,0) ig_queue_length_reg;
    RegisterAction<bit<32>, bit<5>, bit<32>>(ig_queue_length_reg) ig_queue_length_reg_write = {
       void apply(inout bit<32> value, out bit<32> read_value){
            value=hdr.worker.qlength;
            read_value = value;
       }
   };
   RegisterAction<bit<32>, bit<5>, bit<32>>(ig_queue_length_reg) ig_queue_length_reg_read = {
       void apply(inout bit<32> value, out bit<32> read_value){
               read_value = value;
       }
   };
   /* registers to track min and max values of ranks*/
   Register<bit<32>, bit<5>> (32,0) min_rank_reg;
   RegisterAction<bit<32>, bit<5>, bit<32>>(min_rank_reg) min_rank_reg_write_action = {
       void apply(inout bit<32> value, out bit<32> read_value){
           if (value == 0x0){
               value = meta.pkt_rank;
           }
           else if(meta.pkt_rank < value){
               value = meta.pkt_rank;
           }
           read_value = value;
       }
   };
   RegisterAction<bit<32>, bit<5>, bit<32>>(min_rank_reg) min_rank_reg_reset_action = {
       void apply(inout bit<32> value, out bit<32> read_value){
           value = meta.pkt_rank;
       }
   };

   Register<bit<32>, bit<5>> (32,0) max_rank_reg;
   RegisterAction<bit<32>, bit<5>, bit<32>>(max_rank_reg) max_rank_reg_write_action = {
       void apply(inout bit<32> value, out bit<32> read_value){
           if(meta.pkt_rank > value){
                   value = meta.pkt_rank;
           }
           read_value=value;
       }
   };
   RegisterAction<bit<32>, bit<5>, bit<32>>(max_rank_reg) max_rank_reg_reset_action = {
       void apply(inout bit<32> value, out bit<32> read_value){
           value = meta.pkt_rank;
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

    action update_and_get_f_finish_time32(bit<32> flow_index) {
        meta.finish_time = regact_update_and_get_f_finish_time32.execute(flow_index);
    }

    table update_and_get_f_finish_time {
        key = {
            meta.flow_index: exact;
        }
        actions = {
            update_and_get_f_finish_time2;
            // update_and_get_f_finish_time4;
            update_and_get_f_finish_time8;
            update_and_get_f_finish_time16;
            update_and_get_f_finish_time32;
        }
        size = 128;
    }

   action action_subtract_queueLength() {
           meta.available_queue=BufferSize - meta.queue_length;
       }

    table subtract_queueLength{
       actions = { action_subtract_queueLength;}
       default_action = action_subtract_queueLength();
       size=1;
   }
   action action_compute_dividend(){
       meta.dividend = (bit<16>)(meta.pkt_rank - meta.min_pkt_rank);
   }
   table compute_dividend{
       actions = { action_compute_dividend;}
       default_action = action_compute_dividend();
       size=1;
   }
   action action_compute_divisor(){
       meta.divisor = (bit<16>)(meta.max_pkt_rank - meta.min_pkt_rank);
       meta.max_min = meta.divisor;
    }
   table compute_divisor{
       actions = { action_compute_divisor;}
       default_action = action_compute_divisor();
       size=1;
   }

   action action_calculate_left_side(){
    meta.left_side =(bit<24>) meta.dividend_exponent + 18; //(rp-Min)*((1-k)*B)   (1-k)*B---2^14
   }

   table calculate_left_side{
       actions = { action_calculate_left_side;}
       default_action = action_calculate_left_side();
       size=1;
   }

   action action_do_RIFO_admission(){
       meta.rifo_admission = max( meta.left_side, meta.right_side);
    }
    table RIFO_admission{
       actions = { action_do_RIFO_admission;}
       default_action = action_do_RIFO_admission();
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
//    action set_rank(){
//        hdr.rifo.rank =(bit<16>) hdr.ipv4.src_addr[7:0];
//     }
   action set_exponent_buffer(bit<8> exponent_value){
       meta.buffer_exponent = exponent_value ;
    }
   table queue_length_lookup {
       key = {
           meta.available_queue: range;
       }
       actions = {
           set_exponent_buffer;
       }
       size = 512;
   }

    action set_exponent_dividend(bit<8> exponent_value){
       meta.dividend_exponent= exponent_value ;
    }
   table dividend_lookup {
       key = {
           meta.dividend: range;
       }
       actions = {
           set_exponent_dividend;
       }
       size = 512;
   }

   action set_exponent_max_min(bit<8> exponent_value){
       meta.max_min_exponent= exponent_value ;
    }
   table max_min_lookup {
       key = {
           meta.max_min: range;
       }
       actions = {
           set_exponent_max_min;
       }
       size = 512;
   }

   action action_get_ig_queue_length(){
        bit<32> tmp=ig_queue_length_reg_read.execute(0);
        meta.queue_length=tmp[19:0];

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


   action calculate_max_min_buffer_mul(bit<24> mul){
       meta.right_side= mul ;
    }
   table max_min_buffer_lookup {
       key = {
           meta.max_min_exponent: exact;
           meta.buffer_exponent: exact;
       }
       actions = {
           calculate_max_min_buffer_mul;
       }
       size = 512;
   }
   apply {
        if (hdr.ipv4.isValid()) {

            // do routing to get the egress port and qid
            table_forward.apply();
            if(hdr.worker.isValid()){
                    set_ig_queue_length.apply();
                    set_ig_round_reg.execute(0);
                    ig_dprsr_md.drop_ctl = 0;
                    worker_recirculate();
            }
            else if(hdr.tcp.isValid()){
                // get flow_index
                // if(hdr.udp.isValid()){
                //     get_weightindex_UDP_table.apply();
                // }
                // else{             
                //     get_weightindex_TCP_table.apply();
                // }
                get_weightindex_TCP_table.apply();
                //get round
                meta.round = get_ig_round_reg.execute(0);
                // // Get weight
                // get_weight_table.apply();
                //get finish_time
                update_and_get_f_finish_time.apply();
                bit<32> tmp= get_and_update_time.execute(meta.flow_index);
                if(tmp==1){
                    meta.pkt_rank=meta.finish_time;
                    write_rank_reg.execute(meta.flow_index);
                }
                else{
                    meta.pkt_rank=read_rank_reg.execute(meta.flow_index);
                }
                //get counter
                bit<16> count_reset = set_counter_reg.execute(0);
                if(count_reset == 1){
                    max_rank_reg_reset_action.execute(0);
                    min_rank_reg_reset_action.execute(0);
                }
                else{
                    //Get Max and Min ranks
                    meta.min_pkt_rank = min_rank_reg_write_action.execute(0);
                    meta.max_pkt_rank = max_rank_reg_write_action.execute(0);
                }

                /*compute dividend (Max-rank) and divisor (Max-Min)*/
                compute_divisor.apply();
                compute_dividend.apply();

                get_ig_queue_length.apply();
                //compute the actual queue length (B-L)
                subtract_queueLength.apply();
                //find exponent of the remaining queue length (B-L) using lookup tables

                queue_length_lookup.apply();
                //find (max - min) exponent using lookup
                max_min_lookup.apply();

                //do multiplication of (max-min) * (B-l) using lookup tables
                max_min_buffer_lookup.apply();

                //get exponent of dividend
                dividend_lookup.apply();

                 calculate_left_side.apply();

                /* check RIFO admision condition */
                RIFO_admission.apply();

                // one condition for all
                // (1-K) * (rank-Min) * B >= (B-l) * (Max-Min)

                if ( meta.max_pkt_rank != meta.min_pkt_rank && meta.rifo_admission == meta.left_side ) {
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

    Register<bit<32>, _>(32w1) eg_queue_length_reg;
    RegisterAction<bit<32>, _, bit<32>>(eg_queue_length_reg) eg_queue_length_reg_write = {
       void apply(inout bit<32> value, out bit<32> read_value){
            value = (bit<32>)eg_intr_md.deq_qdepth;
            read_value = value;
       }
   };
   RegisterAction<bit<32>, _, bit<32>>(eg_queue_length_reg) eg_queue_length_reg_read = {
       void apply(inout bit<32> value, out bit<32> read_value){
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
