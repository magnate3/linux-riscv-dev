#include <core.p4>
#include <t2na.p4>

//THIS VERSION IS BASED ON THE /home/leris/p4code/alireza/iRED_modified/INT_version/L4S_Classic_CG/L4S_Classic_CG.p4
// CHAGING THE EGRESS TO INGRESS PROCESSING

/*** Types of variable ***/
typedef bit<4> header_type_t;
typedef bit<4> header_info_t;
typedef bit<32> number_of_ports_t;
typedef bit<48> mac_addr_t;
typedef bit<32> ipv4_addr_t;
typedef bit<16> ether_type_t;
typedef bit<32> value_t;
typedef bit<10> index_t;
typedef bit<9> ports_t;

//LOW PASS FILTER EWMA TYPE
typedef bit<16> lpf_type_16;
typedef bit<32> lpf_type_32;

/*** INT TYPE ***/
typedef bit<6> switchID_v; //para fechar 232 bits divisivel por 8 = 29bytes
typedef bit<9> ingress_port_v;
typedef bit<9> egress_port_v;
//32
typedef bit<9>  egressSpec_v;
typedef bit<7>  qid_v;

typedef bit<32>  ingress_global_timestamp_v; //era 48, virou 32 pra armazenar
typedef bit<32>  egress_global_timestamp_v; //era 48, virou 32 pra armazenar
typedef bit<32>  enq_timestamp_v;
typedef bit<32> enq_qdepth_v; //the value stored in the metadata is 19
typedef bit<32> deq_timedelta_v;
//typedef bit<3>  priority_v;
typedef bit<32> deq_qdepth_v;


/*** Constants ***/
const number_of_ports_t N_PORTS                = 512;
const header_type_t HEADER_TYPE_NORMAL_PKT     = 0;
const header_type_t HEADER_TYPE_MIRROR_EGRESS  = 1;
const header_type_t HEADER_TYPE_MIRROR_INGRESS  = 2;
const ether_type_t ETHERTYPE_IPV4              = 16w0x0800;
const value_t TARGET_DELAY_L4S                     = 2000000;  // 2 ms
const value_t TARGET_DELAY_L4S_DOUBLE              = 10000000;  // 4ms


//TERMINAR ESSA VERSAO
//const value_t TARGET_DELAY_CG                     = 20000000;  // 2 ms
//const value_t TARGET_DELAY_CG_DOUBLE              = 40000000;  // 4ms


const ports_t recirc_port                        = 6; // recirc port 0 on pipe 1

const ports_t MIRROR_PORT                        = 128; // recirc port 0 on pipe 1



const bit<8> PROTO_INT = 253;
const bit<8> PROTO_UDP = 17;
const bit<8> PROTO_TCP = 6;
const bit<2>  RTP_VERSION = 2;

#define INTERNAL_HEADER         \
    header_type_t header_type;  \
    header_info_t header_info

#define MAX_HOPS 10
#define NUMBER_OF_QUEUES 3 //CLASSIC L4S AND CG

/*** Headers ***/
header ethernet_h {
    mac_addr_t dst_mac_addr;
    mac_addr_t src_mac_addr;
    bit<16> ether_type;
}

header ipv4_h {
    bit<4> version;
    bit<4> ihl;
    bit<6>  dscp;
    //bit<1>  l4s;
    bit<2>  ecn;
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


header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  reserved;
    bit<9>  flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header rtp_t {
    bit<2> version;
    bit<1> padding;
    bit<1> extension;
    bit<4> csrcCounter;
    bit<1> marker;
    bit<7> payloadType;
    bit<16> seqNumber;
    bit<32> timestamp;
    bit<32> ssrcID;
    //bit<16> csrcID;

}

header classification_h {
    bit<8> metadata_classT1;
    bit<8> metadata_classT2;
    bit<8> metadata_classT3;
    bit<8> metadata_classT4;
    bit<8> metadata_classT5;
    bit<8> metadata_final_classification;
}

//  hdr.INT[0].ingress_port = (ingress_port_v)standard_metadata.ingress_port;
//         hdr.INT[0].egress_port = (egress_port_v)standard_metadata.egress_port;
//         hdr.INT[0].egress_spec = (egressSpec_v)standard_metadata.egress_spec;
//         hdr.INT[0].priority = (priority_v)standard_metadata.priority;
//         hdr.INT[0].qid = (qid_v)standard_metadata.qid;
//         hdr.INT[0].ingress_global_timestamp = (ingress_global_timestamp_v)standard_metadata.ingress_global_timestamp;
//         hdr.INT[0].egress_global_timestamp = (egress_global_timestamp_v)standard_metadata.egress_global_timestamp;
//         hdr.INT[0].enq_timestamp = (enq_timestamp_v)standard_metadata.enq_timestamp;


//         hdr.INT[0].enq_qdepth = (enq_qdepth_v)enq_qdepth_avg;
//         hdr.INT[0].deq_timedelta = (deq_timedelta_v)deq_timedelta_avg;
//         hdr.INT[0].deq_qdepth = (deq_qdepth_v)deq_qdepth_avg;
//         hdr.INT[0].processing_time = (deq_timedelta_v)processing_time_avg;

/*** Mirror Header to carry port metadata ***/
header mirror_h {
    INTERNAL_HEADER;
    @flexible PortId_t egress_port;
    @flexible MirrorId_t  mirror_session;
    @flexible bit<32> IPG;
    @flexible bit<8> testing;
    @flexible bit<8> flowid;
    @flexible bit<32> ifg;
    @flexible bit<16> packet_size;
    @flexible bit<16> frame_size;
    @flexible bit<1> rtp_marker;
    @flexible bit<32> queue_delay;
    @flexible bit<8> has_rtp;
    //@flexible bit<48> egress_global_tstamp;
}





//*** Bridge header to carry ingress timestamp from Ingress to Egress ***//
header bridge_h {
    bit<16> bridge_ingress_port; //9bits tem que bater 64...
    bit<48> ingress_global_tstamp;
    bit<16> bridge_qid;
}


header nodeCount_h{
    bit<16>  id;
    bit<16>  count;
}




header InBandNetworkTelemetry_h {
    switchID_v swid;
    ingress_port_v ingress_port;
    egress_port_v egress_port;
    egressSpec_v egress_spec;
    qid_v qid;
    ingress_global_timestamp_v ingress_global_timestamp;
    egress_global_timestamp_v egress_global_timestamp;
    enq_timestamp_v enq_timestamp;
    enq_qdepth_v enq_qdepth;
    deq_timedelta_v deq_timedelta;
    deq_qdepth_v deq_qdepth;
    //priority_v priority;
    deq_timedelta_v processing_time;
    bit<32> number_of_packets_for_average;
    
    
}

// header cute_new_header_h{
//     bit<16> number_of_packets_frame;
//     bit<32> ipi_value;
//     bit<32> ipf_value;
// }

struct headers_t {
    bridge_h            bridge;
    mirror_h            mirror;
    ethernet_h          ethernet;
    ipv4_h              ipv4;
    tcp_t        tcp;
    udp_t        udp;
    rtp_t        rtp;
    // cute_new_header_h cute_new_header;
    nodeCount_h        nodeCount;
    InBandNetworkTelemetry_h[2] INT;
}



struct ingress_metadata_t {
    bit<16>  count;
}

struct parser_metadata_t {
    bit<16>  remaining;
}



struct metadata_t{
    
    classification_h classification; // Alireza 

    bridge_h    bridge;
    mirror_h    mirror;
    MirrorId_t  mirror_session;
    PortId_t    egress_port;
    header_type_t header_type;
    header_info_t header_info;
    //bit<48> egress_global_tstamp;
    //bit<32> queue_delay;
    ingress_metadata_t   ingress_metadata;
    parser_metadata_t   parser_metadata;
    //bit<8> qid; // Alireza
    bit<16> metadata_index;
    bit<32> metadata_enq_qdepth;
    bit<32> metadata_queue_delay;
    bit<32> metadata_totalPkts;
    bit<32> metadata_input_ipg;
    
    bit<32> metadata_ifg;
    bit<32> metadata_ifg_temp;
    bit<32> metadata_ipg;
    bit<32> metadata_ipg_temp;
    bit<32>metadata_queue_delay_new;
    
    bit<16> metadata_frame_size;
    bit<16> metadata_packet_size;
    bit<8> metadata_flowID; //maybe change to 32 if problem
    bit<8> metadata_rtp_marker;
    bit<8> metadata_has_rtp;
    // bit<8> metadata_counter1;
    // bit<8> metadata_counter2;
    // bit<8> metadata_counter3;
    
    //bit<8> metadata_classT1;
    //bit<8> metadata_classT2;
    //bit<8> metadata_classT3;
    //bit<8> metadata_classT4;
    //bit<8> metadata_classT5;
    //bit<8> metadata_final_classification;
    bit<20> metadata_ipg_20lsb;
    bit<20> metadata_ifg_20lsb;
}


// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------
parser SwitchIngressParser(
        packet_in pkt,
        out headers_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
       pkt.extract(ig_intr_md);
        transition select(ig_intr_md.ingress_port){
            (MIRROR_PORT): parse_mirror;
            (_): parse_port_metadata;
            
        }
    }

    /* NORMAL PKTS */
    state parse_port_metadata{
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    /* I2I MIRRORED PKTS */
    state parse_mirror{
        pkt.advance(PORT_METADATA_SIZE); //its ok until here
        pkt.extract(hdr.mirror);
        transition parse_bridge;
    }

        // JUST FOR MIRRPRED PPACKETS
    state parse_bridge{
        pkt.extract(hdr.bridge);
        transition parse_ethernet;
    }

    

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select (hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_UDP: parse_udp;
            PROTO_INT: parse_count;
            PROTO_TCP: parse_tcp;
            //PROTO_ICMP: parse_icmp;
        default: accept;
      }
    }
    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition parse_rtp;
    }

    state parse_rtp {
        pkt.extract(hdr.rtp);
        transition accept;
    }
    
    state parse_count{
        pkt.extract(hdr.nodeCount);
        //transition accept;
        ig_md.parser_metadata.remaining = hdr.nodeCount.count;
        transition select(ig_md.parser_metadata.remaining) {
            2 : parse_two_int;
            1: parse_one_int;
            0: accept;
        }
    }

    state parse_two_int{
        pkt.extract(hdr.INT.next);
        pkt.extract(hdr.INT.next);
        transition accept;
    }

    state parse_one_int{
        pkt.extract(hdr.INT.next);
        transition accept;
    }
}


// SwitchIngress
control SwitchIngress(
        inout headers_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    //Register<bit<32>, _> (0x100) ipg_reg;
    Register<bit<8>, _> (1) flow_id_reg; //after LPF calculates the EWMA
    Register<bit<16>, _> (N_PORTS) packet_size_reg;
    Register<bit<32>, _> (0x100) ingress_timestamp_reg_every_pkt;
    Register<bit<16>, _> (0x100) packet_size_ewma_reg;
    Register<bit<16>, _> (0x100) frame_size_reg; //uses mathunit
    Register<bit<16>, _> (0x100) frame_size_ewma_reg; //after LPF calculates the EWMA
    //Register<bit<16>, _> (0x100) rtp_seq_number;
    Lpf<lpf_type_16, bit<8>>(size=1024) lpf_packet_size;
    lpf_type_16 lpf_input_packet_size;
    lpf_type_16 lpf_output_packet_size;

    Lpf<lpf_type_32, bit<8>>(size=1024) lpf_ipg;
    lpf_type_32 lpf_input_ipg;
    lpf_type_32 lpf_output_ipg;

    Lpf<lpf_type_16, bit<8>>(size=1024) lpf_frame_size;
    lpf_type_16 lpf_input_frame_size;
    lpf_type_16 lpf_output_frame_size;

    Lpf<lpf_type_32, bit<8>>(size=1024) lpf_ifg;
    lpf_type_32 lpf_input_ifg;
    lpf_type_32 lpf_output_ifg;
    

    Hash<bit<8>>(HashAlgorithm_t.CRC8) decider_hash_tcp; //2 to the power of 8 possible flows
    Hash<bit<8>>(HashAlgorithm_t.CRC8) decider_hash_udp; //2 to the power of 8 possible flows


    //MathUnit<bit<32>> (MathOp_t.MUL, 1, 10) division_transmission_rate;

    // RegisterAction<bit<32>, bit<8>, bit<32>>(ipg_reg) update_ipg_reg = {
    //         void apply(inout bit<32> value, out bit<32> result) {
    //             result = value;
    //             value = ig_md.metadata_input_ipg;

    //         }
    // };

    RegisterAction<bit<16>, bit<8>, bit<16>>(packet_size_reg) store_packet_size_reg= {
            void apply(inout bit<16> value) {
                value = ig_md.metadata_packet_size;

            }
    };

    RegisterAction<bit<8>, bit<8>, bit<8>>(flow_id_reg) store_flow_id_reg = {
            void apply(inout bit<8> value) {
                value = ig_md.metadata_flowID;

            }
    };

    RegisterAction<bit<32>, bit<8>, bit<32>>(ingress_timestamp_reg_every_pkt) update_ingress_timestamp_reg_every_pkt = {
            void apply(inout bit<32> value, out bit<32> result) {
                result = value;
                value = ig_intr_prsr_md.global_tstamp[31:0];

            }
    };

    // RegisterAction<bit<16>, bit<8>, bit<16>>(packet_size_ewma_reg) store_packet_size_ewma_reg = {
    //         void apply(inout bit<16> value) {
    //             value = lpf_output_packet_size;
    //         }
    // };
    // RegisterAction<bit<16>, bit<8>, bit<16>>(frame_size_ewma_reg) store_frame_size_ewma_reg = {
    //         void apply(inout bit<16> value) {
    //             ////////////////value = lpf_output_frame_size;
    //             value = hdr.mirror.packet_size;
    //         }
    // };


    // RegisterAction<bit<16>, bit<8>, bit<16>>(rtp_seq_number) update_rtp_seq_number = {
    //         void apply(inout bit<16> value, out bit<16> result) {
    //             result = value;
    //             value = hdr.rtp.seqNumber;
    //         }
    // };

    RegisterAction<bit<16>, bit<8>, bit<16>>(frame_size_reg) store_frame_size_reg = {
            void apply(inout bit<16> value, out bit<16> output_value) {
                //result = value;
                value = ig_md.metadata_frame_size;
            }
    };

    RegisterAction<bit<16>, bit<8>, bit<16>>(frame_size_reg) read_frame_size_reg = {
            void apply(inout bit<16> value, out bit<16> output_value) {
                //result = value;
                value = value;
                output_value = value;
            }
    };




    


    

    bit<32> recirculationTime;
    bool flag;
    Register<bit<32>, _> (1) recircTime;
    //Register<bit<16>, _>(N_PORTS) congest_port;
    // Counter<bit<32>, bit<32>>(N_PORTS, CounterType_t.PACKETS) drop_cloned_pkt;
    // Counter<bit<32>, bit<8>>(N_PORTS, CounterType_t.PACKETS) counter_packets_frame;
    Register<bit<32>, _>(N_PORTS) drop_cloned_pkt;
    Register<bit<32>, _>(N_PORTS) counter_packets_frame;
    //Counter<bit<32>, bit<32>>(N_PORTS, CounterType_t.PACKETS) drop_regular_pkt;

    RegisterAction<bit<32>, bit<8>, bit<32>>(drop_cloned_pkt) count_drop_cloned_pkt = {
            void apply(inout bit<32> value) {
                value = value+1;
            }
    };

    RegisterAction<bit<32>, bit<8>, bit<32>>(counter_packets_frame) count_counter_packets_frame = {
            void apply(inout bit<32> value) {
                value = value+1;
            }
    };
    action drop_regular_pkts(){
        ig_intr_dprsr_md.drop_ctl = 0x1;
    }

    action drop_cloned_pkts(){
        ig_intr_dprsr_md.drop_ctl = 0x1;
    }

    action find_flowID_ipv4_udp(){
        ig_md.metadata_flowID = decider_hash_udp.get({hdr.ipv4.dst_addr, hdr.udp.dstPort, hdr.ipv4.protocol});
    }

    action find_flowID_ipv4_tcp(){
        // bit<1> base = 0;
        // bit<16> max = 0xffff;
        // bit<16> hash_result;
        // //bit<48> IP_Port = hdr.ipv4.dstAddr ++ hdr.udp.dstPort;
        // bit<48> IP_dst_add = hdr.ipv4.dstAddr;
        // bit<48> UDP_dst_port = hdr.udp.dstPort;
        // bit<8> IP_Proto = hdr.ipv4.proto;
        // hash(
        //      hash_result,
        //      HashAlgorithm.crc8,
        //      base,
        //      {
        //         IP_Port, UDP_dst_port, IP_Proto
        //      },
        //      max
        //      );

        //bit<48> concatenated_hash_input = (bit<48>) (hdr.ipv4.dst_addr ++ hdr.udp.dstPort);

        ig_md.metadata_flowID = decider_hash_tcp.get({hdr.ipv4.dst_addr, hdr.tcp.dstPort, hdr.ipv4.protocol});
        //ig_md.metadata_flowID = decider_hash.get({concatenated_hash_input});
        // bit<8> temp_index = 0;
        // store_flow_id_reg.execute(temp_index);
    }


    Register<bit<8>, _> (N_PORTS) rtp_marker_reg;
    RegisterAction<bit<8>, bit<8>, bit<8>>(rtp_marker_reg) store_rtp_marker_reg = {
            void apply(inout bit<8> value) {
                value = ig_md.metadata_rtp_marker;
            }

    };



    Register<bit<32>, _> (N_PORTS) ipg_cloned_packets_reg;
    RegisterAction<bit<32>, bit<32>, bit<32>>(ipg_cloned_packets_reg) store_ipg_cloned_packets_reg = {
            void apply(inout bit<32> value) {
                value = ig_md.metadata_ipg;
            }

    };

    Register<bit<32>, _> (N_PORTS) ifg_reg;
    RegisterAction<bit<32>, bit<8>, bit<32>>(ifg_reg) store_ifg_reg = {
            void apply(inout bit<32> value) {
                value = ig_md.metadata_ifg;
            }

    };
    RegisterAction<bit<32>, bit<8>, bit<32>>(ifg_reg) read_ifg_reg = {
            void apply(inout bit<32> value, out bit<32> output_value) {
                value = value;
                output_value = value;
            }

    };

    Register<bit<32>, _> (1) queue_delay;
    RegisterAction<bit<32>, bit<8>, bit<32>>(queue_delay) sum_queue_delay = {
            void apply(inout bit<32> value) {
                value = value + ig_md.metadata_queue_delay_new;
            }

    };

    Register<bit<8>, _> (N_PORTS) marking_register;
    RegisterAction<bit<8>, bit<8>, bit<8>>(marking_register) read_marking_register = {
            void apply(inout bit<8> value, out bit<8> output_value) {
                value = value;
                output_value = value;
            }

    };

    RegisterAction<bit<8>, bit<8>, bit<8>>(marking_register)  store_marking_register = {
            void apply(inout bit<8> value) {
                value = (bit<8>) ig_md.classification.metadata_final_classification;  //ig_md.metadata_final_classification;
            }

    };
    
    Register<bit<8>, _> (N_PORTS) metadata_classT1;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT1)  store_metadata_classT1 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT1;
            }

    };

    Register<bit<8>, _> (N_PORTS) metadata_classT2;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT2)  store_metadata_classT2 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT2;
            }

    };

    Register<bit<8>, _> (N_PORTS) metadata_classT3;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT3)  store_metadata_classT3 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT3;
            }

    };

    Register<bit<8>, _> (N_PORTS) metadata_classT4;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT4)  store_metadata_classT4 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT4;
            }

    };

    Register<bit<8>, _> (N_PORTS) metadata_classT5;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT5)  store_metadata_classT5 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT5;
            }

    };

    

    //T1
    action classify_T1_FS(bit<8> classify_result){
        ig_md.classification.metadata_classT1 = classify_result;
    }

    action classify_T1_IPG(bit<8> classify_result){
        ig_md.classification.metadata_classT1 = classify_result;
    }

    table table_T1_FS{
        key = {
            ig_md.metadata_frame_size: range;
        }
        actions = {
            classify_T1_FS();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table table_T1_IPG{
        key = {
            ig_md.metadata_ipg_20lsb: range;
        }
        actions = {
            classify_T1_IPG();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    //T2
    action classify_T2_IFG(bit<8> classify_result){
        ig_md.classification.metadata_classT2 = classify_result;
    }

    action classify_T2_IPG(bit<8> classify_result){
        ig_md.classification.metadata_classT2 = classify_result;
    }
    
    table table_T2_IFG{
        key = {
            ig_md.metadata_ifg_20lsb: range;
        }
        actions = {
            classify_T2_IFG();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table table_T2_IPG{
        key = {
            ig_md.metadata_ipg_20lsb: range;
        }
        actions = {
            classify_T2_IPG();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    action classify_T3_FS(bit<8> classify_result){
        ig_md.classification.metadata_classT3 = classify_result;
    }

    action classify_T3_IFG(bit<8> classify_result){
        ig_md.classification.metadata_classT3 = classify_result;
    }

    action classify_T3_IPG(bit<8> classify_result){
        ig_md.classification.metadata_classT3 = classify_result;
    }


    //T3
    table table_T3_FS{
        key = {
            ig_md.metadata_frame_size: range;
        }
        actions = {
            classify_T3_FS();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table table_T3_IFG{
        key = {
            ig_md.metadata_ifg_20lsb: range;
        }
        actions = {
            classify_T3_IFG();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table table_T3_IPG{
        key = {
            ig_md.metadata_ipg_20lsb: range;
        }
        actions = {
            classify_T3_IPG();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }
    //T4

    action classify_T4_IPG(bit<8> classify_result){
        ig_md.classification.metadata_classT4 = classify_result;
    }
    action classify_T4_IFG(bit<8> classify_result){
        ig_md.classification.metadata_classT4 = classify_result;
    }
    action classify_T4_FS(bit<8> classify_result){
        ig_md.classification.metadata_classT4 = classify_result;
    }

    
    table table_T4_IFG{
        key = {
            ig_md.metadata_ifg_20lsb: range;
        }
        actions = {
            classify_T4_IFG();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table table_T4_IPG{
        key = {
            ig_md.metadata_ipg_20lsb: range;
        }
        actions = {
            classify_T4_IPG();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table table_T4_FS{
        key = {
            ig_md.metadata_frame_size: range;
        }
        actions = {
            classify_T4_FS();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    //T5
    action classify_T5_FS(bit<8> classify_result){
        ig_md.classification.metadata_classT5 = classify_result;
    }

    action classify_T5_IPG(bit<8> classify_result){
        ig_md.classification.metadata_classT5 = classify_result;
    }

    table table_T5_FS{
        key = {
            ig_md.metadata_frame_size: range;
        }
        actions = {
            classify_T5_FS();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }
    table table_T5_IPG{
        key = {
            ig_md.metadata_ipg_20lsb: range;
        }
        actions = {
            classify_T5_IPG();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    //majority
    action final_classification(bit<8> majority){
        //ig_md.metadata_final_classification = (bit<8>) majority;
        ig_md.classification.metadata_final_classification = (bit<8>) majority;
    }
    table table_majority{
        key = {
            //ig_md.metadata_classT1: exact;
            //ig_md.metadata_classT2: exact;
            //ig_md.metadata_classT3: exact;
            //ig_md.metadata_classT4: exact;
            //ig_md.metadata_classT5: exact;
            
            ig_md.classification.metadata_classT1: exact;
            ig_md.classification.metadata_classT2: exact;
            ig_md.classification.metadata_classT3: exact;
            ig_md.classification.metadata_classT4: exact;
            ig_md.classification.metadata_classT5: exact;
            
        }
        actions = {
            final_classification();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }
    

    
    


   
    

    apply {
    
    // //* Check for cloned pkts *//
    if (ig_intr_md.ingress_port == MIRROR_PORT){
        //if(hdr.mirror.frame_size != 0){
            /////////////////////lpf_input_frame_size = hdr.ipv4.total_len;
            //lpf_input_frame_size = hdr.mirror.frame_size;
            //////////////////////////lpf_output_frame_size = lpf_frame_size.execute(lpf_input_frame_size,0);
            //store_frame_size_ewma_reg.execute(hdr.mirror.flowid);
        //}
        //* Cloned pkts *//
        //* Turn ON congestion flag. Write '1' in the register index port *//
        // write_congest_port.execute((bit<16>)hdr.mirror.egress_port);
        
        // //* Compute recirculation time from egress to ingress *//
        ig_tm_md.ucast_egress_port = 100; //drop????
        //recirculationTime = (bit<32>)ig_intr_prsr_md.global_tstamp - (bit<32>)hdr.mirror.egress_global_tstamp;
        // recirculationTime = (bit<32>)hdr.mirror.egress_global_tstamp;
        // recirc_action.execute(0);

        //ig_md.metadata_flowID = hdr.mirror.flowid;

        //if (hdr.ipv4.isValid() && hdr.ipv4.protocol == PROTO_UDP) {
        // if(hdr.ipv4.protocol == PROTO_UDP){
        //     find_flowID_ipv4_udp();
        // }
        // else if(hdr.ipv4.protocol == PROTO_TCP){
        //     find_flowID_ipv4_tcp();
        // }


        ig_md.metadata_flowID = hdr.mirror.flowid;
        bit<8> temp_index = 0;
        store_flow_id_reg.execute(temp_index);

        ig_md.metadata_rtp_marker = (bit<8>) hdr.mirror.rtp_marker;
        store_rtp_marker_reg.execute(ig_md.metadata_flowID);

        //drop_cloned_pkt.count((bit<32>)ig_md.metadata_flowID); //maybe flowid?  (bit<32>)hdr.mirror.flowid
        count_drop_cloned_pkt.execute(ig_md.metadata_flowID);

        if(hdr.mirror.rtp_marker == 1){
            //counter_packets_frame.count(ig_md.metadata_flowID);
            count_counter_packets_frame.execute(ig_md.metadata_flowID);
        }

        // //* Drop cloned pkt *//
        // drop_cloned_pkt.count((bit<32>)hdr.mirror.egress_port); //maybe flowid?  (bit<32>)hdr.mirror.flowid
        // drop_cloned_pkts();


        //packet size
        lpf_input_packet_size = hdr.mirror.packet_size;
        lpf_output_packet_size = lpf_packet_size.execute(lpf_input_packet_size,0);
        //ig_md.metadata_packet_size = hdr.mirror.packet_size;
        ig_md.metadata_packet_size = lpf_output_packet_size;
        store_packet_size_reg.execute(ig_md.metadata_flowID);


        //bit<32> shit = 0;
        //ipg
        lpf_input_ipg = hdr.mirror.IPG;
        lpf_output_ipg = lpf_ipg.execute(lpf_input_ipg, 0);
        //ig_md.metadata_input_ipg = hdr.mirror.IPG;
        ig_md.metadata_ipg = lpf_output_ipg;
        store_ipg_cloned_packets_reg.execute((bit<32>)ig_md.metadata_flowID);


        //frame size
        if(hdr.mirror.frame_size != 0){
            lpf_input_frame_size = hdr.mirror.frame_size;
            lpf_output_frame_size = lpf_frame_size.execute(lpf_input_frame_size, 0);
            //ig_md.metadata_frame_size = hdr.mirror.frame_size;
            ig_md.metadata_frame_size = lpf_output_frame_size;
            store_frame_size_reg.execute(ig_md.metadata_flowID);
        }
        else if(hdr.mirror.frame_size == 0){
            ig_md.metadata_frame_size = read_frame_size_reg.execute(ig_md.metadata_flowID);
        }

        //ifg
        if(hdr.mirror.ifg != 0){
            lpf_input_ifg = hdr.mirror.ifg;
            lpf_output_ifg = lpf_ifg.execute(lpf_input_ifg, 0);
            //ig_md.metadata_ifg = hdr.mirror.ifg;
            ig_md.metadata_ifg = lpf_output_ifg;
            store_ifg_reg.execute(ig_md.metadata_flowID);
        }
        else if(hdr.mirror.ifg == 0){
            ig_md.metadata_ifg = read_ifg_reg.execute(ig_md.metadata_flowID);
        }
        




        //queue delay (previous + current. the collect_info python resets when read)
        ig_md.metadata_queue_delay_new = hdr.mirror.queue_delay;
        sum_queue_delay.execute(temp_index);


        //classification
        ig_md.metadata_ifg_20lsb = ig_md.metadata_ifg[19:0];
        ig_md.metadata_ipg_20lsb = ig_md.metadata_ipg[19:0];

        //T1
        table_T1_FS.apply();
        if(ig_md.classification.metadata_classT1 == 0){
            table_T1_IPG.apply();
        }

        //T2
        table_T2_IFG.apply();
        if(ig_md.classification.metadata_classT2 == 0){
            table_T2_IPG.apply();
        }

        //T3
        table_T3_FS.apply();
        if(ig_md.classification.metadata_classT3 == 0){
            table_T3_IFG.apply();
        }
        if(ig_md.classification.metadata_classT3 == 0){
            table_T3_IPG.apply();
        }

        //T4
        table_T4_IFG.apply();
        if(ig_md.classification.metadata_classT4 == 5){
            table_T4_IPG.apply();
        }
        else if(ig_md.classification.metadata_classT4 == 6){
            table_T4_FS.apply();
        }

        //T5
        table_T5_FS.apply();
        if(ig_md.classification.metadata_classT5 == 0){
            table_T5_IPG.apply();
        }

        table_majority.apply();
        store_metadata_classT1.execute(ig_md.metadata_flowID);
        store_metadata_classT2.execute(ig_md.metadata_flowID);
        store_metadata_classT3.execute(ig_md.metadata_flowID);
        store_metadata_classT4.execute(ig_md.metadata_flowID);
        store_metadata_classT5.execute(ig_md.metadata_flowID);

        
        store_marking_register.execute(ig_md.metadata_flowID);





        drop_cloned_pkts();

    }else{
        
        // 1/1 -> 1/0
        if (ig_intr_md.ingress_port == 137) {
            ig_tm_md.ucast_egress_port = 136;
        }
        // 1/0 -> 1/1
        else if (ig_intr_md.ingress_port == 136) {
            // if(hdr.nodeCount.isValid()){
            //     if(hdr.nodeCount.count < 2){
            //         //loopbackport
            //         ig_tm_md.ucast_egress_port = recirc_port;
            //     }
            //     else{
            //         //sendback
            //         ig_tm_md.ucast_egress_port = 136;
            //     }
            // }
            // else{
                //ig_tm_md.ucast_egress_port = recirc_port;
                //ig_tm_md.ucast_egress_port = 137;
            //}
            if(hdr.nodeCount.isValid()){
                if(hdr.nodeCount.count == 0){
                    //recirculate
                    //ig_tm_md.ucast_egress_port = recirc_port;
                    ig_tm_md.ucast_egress_port[8:7] = ig_intr_md.ingress_port[8:7];
                    ig_tm_md.ucast_egress_port[6:0] = recirc_port[6:0]; //n sei pq pega os 7 bits
                }
                else{
                    //sendback talvez esse aqui nem precise mas deixa ai
                    ig_tm_md.ucast_egress_port = 136;
                }
            }
            else{
                ig_tm_md.ucast_egress_port = 137;
            }
        }

        


        // if(ig_intr_md.ingress_port == recirc_port){
        //     ig_tm_md.ucast_egress_port = 136;
        // }

        // 23/0 (448) TF1 -> 1/1(137) Luigi
        else if (ig_intr_md.ingress_port == 448) {
            ig_tm_md.ucast_egress_port = 137;
        }

        // 24/0 (40) TF1 -> 1/1(137) Luigi
        else if (ig_intr_md.ingress_port == 440) {
            ig_tm_md.ucast_egress_port = 137;
        }
        else{
            //RECIRCULACAO
            ig_tm_md.ucast_egress_port = 136;
        }

        


        //extract features FOR EVERY PACKET (per flow down)

        // //timestamp
        // bit<32> ingress_previous_timestamp = update_ingress_timestamp_reg_every_pkt.execute(ig_md.metadata_flowID);


        // //PACKET SIZE!!!
        // //ig_md.metadata_input_ipg = ig_intr_prsr_md.global_tstamp[31:0] - ingress_previous_timestamp;
        // lpf_input_packet_size = hdr.ipv4.total_len;
        // lpf_output_packet_size = lpf_packet_size.execute(lpf_input_packet_size,0);
        // //bit<32>
        // store_packet_size_ewma_reg.execute(ig_md.metadata_flowID);
        


        //FRAME SIZE!!!!
        //hdr.cute_new_header.setValid();
        // if(hdr.rtp.marker == 1){
        //     if(hdr.rtp.version == RTP_VERSION){
        //         ig_md.metadata_frame_size = reset_frame_size_reg.execute(ig_md.metadata_flowID);
                
        //         // bit<16> previous_rtp_seq_number = update_rtp_seq_number.execute(ig_md.metadata_flowID);
        //         // ig_md.metadata_number_packets_frame = hdr.rtp.seqNumber - previous_rtp_seq_number;

        //         //hdr.cute_new_header.rtp_seq_number = update_rtp_seq_number.execute(ig_md.metadata_flowID);

        //         // hdr.cute_new_header.rtp_seq_number = hdr.rtp.seqNumber - previous_rtp_seq_number;
        //         //bit<16> number_packets_frame = hdr.rtp.seqNumber - previous_rtp_seq_number;
        //         //hdr.cute_new_header.number_of_packets_frame = hdr.rtp.seqNumber - previous_rtp_seq_number;
        //         // bit<16> frame_size = number_packets_frame * hdr.udp.length_;

        //         // hdr.cute_new_header.frame_size = (bit<16>)frame_size;
        //         //increase_frame_size_reg.execute(ig_md.metadata_flowID);

        //         // lpf_input_packet_size = hdr.cute_new_header.rtp_seq_number;
        //         // lpf_output_packet_size = lpf_packet_size.execute(lpf_input_packet_size,0);
        //         // store_frame_size_ewma_reg.execute(ig_md.metadata_flowID);

        //         //lpf_input_frame_number_packets = hdr.cute_new_header.number_of_packets_frame;
        //         //lpf_output_frame_number_packets = lpf_frame_size.execute(lpf_input_frame_number_packets,0);
        //     }
        // }
        // //WHEN THE MARKER IS NOT SET, JUST SUM THE PACKET SIZE TO CALCULATE THE FRAME SIZE
        // else{
        //     increase_frame_size_reg.execute(ig_md.metadata_flowID);
        //     ig_md.metadata_frame_size = 0; //just to send the mirror with 0
        // }
        

        //if (hdr.ipv4.l4s == 1){
        // Alireza ****
        //if(hdr.ipv4.ecn == 2){
             //ig_tm_md.qid = 2;
         //}
         //else if(hdr.ipv4.ecn == 1 ){ // IMPORTANT: ECT(1) = 01 (valor 1), ECT(0) = 10 (valor 2)
            //* L4S queue *//
           // ig_tm_md.qid=1; //INT GOING ALSO
        
         //}
         //else{
            //* Classic queue *//
           // ig_tm_md.qid=0; 

            //drop
             //drop_regular_pkts();
        
         //}
            // Alireza ****
        //* Read the output port state from the register*//
        // flag = read_congest_port.execute((bit<16>)ig_tm_md.ucast_egress_port);

        // //* Check if the congestion flag is 1 (Drop ON). *//
        //     if(flag == true){

        //         //if (hdr.ipv4.l4s != 1){ //for L4S not drop.
        //         if(hdr.ipv4.ecn == 0) { 
        //             drop_regular_pkt.count((bit<32>)ig_tm_md.ucast_egress_port);
        //             drop_regular_pkts();

        //         }
        //     }   



        //MARKINGGGG    
        if(hdr.tcp.isValid()){
            find_flowID_ipv4_tcp();
        }
        else if(hdr.udp.isValid()){
            find_flowID_ipv4_udp();
        }
        
        bit<8> marking_decision = read_marking_register.execute(ig_md.metadata_flowID);

        if(hdr.nodeCount.isValid()){
            marking_decision = 0;
        }


        if(marking_decision == 1){
            hdr.ipv4.ecn = 1;
            hdr.ipv4.dscp = 46;
            
        }
        else if(marking_decision == 2){
            hdr.ipv4.ecn = 1;
            hdr.ipv4.dscp = 34;
        }
        else if(marking_decision == 3){
            hdr.ipv4.dscp = 50;
        }

        // Check ECN for ECT(1)
         if(hdr.ipv4.ecn == 1){
        ig_tm_md.qid = 1; // Set L4S Queue 
                 }

        //** Insert ingress timestamp into bridge header to be used in the egress**//
        hdr.bridge.setValid();
        hdr.bridge.bridge_ingress_port = (bit<16>)ig_intr_md.ingress_port;
        hdr.bridge.ingress_global_tstamp = ig_intr_prsr_md.global_tstamp;
        hdr.bridge.bridge_qid = (bit<16>)ig_tm_md.qid; 
        
        //ig_tm_md.ucast_egress_port = 100; //"""drop the original"
        //decisionMirror();

        
        //hdr.cute_new_header.setInvalid();
    }
    
    }    
    
}

// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------
control SwitchIngressDeparser(
        packet_out pkt,
        inout headers_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {

    //Mirror() mirror;
    Checksum() ipv4_checksum;

    apply {
        // if (ig_intr_dprsr_md.mirror_type == HEADER_TYPE_MIRROR_INGRESS){
            
        //     mirror.emit<mirror_h>(hdr.mirror.mirror_session, {hdr.mirror.header_type, 
        //                               hdr.mirror.header_info, 
        //                               hdr.mirror.egress_port,
        //                               hdr.mirror.mirror_session,
        //                               //hdr.mirror.egress_global_tstamp,
        //                               hdr.mirror.packet_size,
        //                               hdr.mirror.frame_size,
        //                               hdr.mirror.ipg,
        //                               hdr.mirror.ifg,
        //                               hdr.mirror.flowid});
        
        // }
        hdr.ipv4.hdr_checksum = ipv4_checksum.update({
            hdr.ipv4.version,
            hdr.ipv4.ihl,
            hdr.ipv4.dscp,
            //hdr.ipv4.l4s,        
            hdr.ipv4.ecn,
            hdr.ipv4.total_len,
            hdr.ipv4.identification,
            hdr.ipv4.flags,
            hdr.ipv4.frag_offset,
            hdr.ipv4.ttl,
            hdr.ipv4.protocol,
            hdr.ipv4.src_addr,
            hdr.ipv4.dst_addr});

        // hdr.udp.checksum = ipv4_checksum.update({
        //     hdr.udp.srcPort,
        //     hdr.udp.length_,
        //     hdr.udp.checksum});
        // pkt.emit(hdr.bridge);
        // pkt.emit(hdr.ethernet);
        // pkt.emit(hdr.ipv4);
        pkt.emit(hdr);
    }
}


// ---------------------------------------------------------------------------
// Traffic Manager - non-programmable block (queues)
// ---------------------------------------------------------------------------
// TM will receive the packet cloned at the Egress, and will recirculate this 
// packet to the Ingress.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Egress parser
// ---------------------------------------------------------------------------
parser EgressParser(
        packet_in pkt,
        out headers_t hdr,
        out metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {

    state start {
        pkt.extract(eg_intr_md);
        transition select(eg_intr_md.egress_port){
            (MIRROR_PORT): parse_mirror;
            (_): parse_bridge;
        }
    }

    /** E2E MIRRORED PKTS **/
    state parse_mirror{
        pkt.extract(hdr.mirror);
        transition accept;
    }


    state parse_bridge{
        pkt.extract(hdr.bridge);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select (hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_UDP: parse_udp;
            PROTO_INT: parse_count;
            //PROTO_ICMP: parse_icmp;
        default: accept;
      }
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition parse_rtp;
    }

    state parse_rtp {
        pkt.extract(hdr.rtp);
        transition accept;
    }

    state parse_count{
        pkt.extract(hdr.nodeCount);
        eg_md.parser_metadata.remaining = hdr.nodeCount.count;
        transition select(eg_md.parser_metadata.remaining) {
            2 : parse_two_int;
            1: parse_one_int;
            0: accept;
        }
    }

    state parse_two_int{
        pkt.extract(hdr.INT.next);
        pkt.extract(hdr.INT.next);
        transition accept;
    }

    state parse_one_int{
        pkt.extract(hdr.INT.next);
        transition accept;
    }
}

control Egress(
        inout headers_t hdr,
        inout metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
        
        value_t queue_delay; //IMPORTANTE! DA PRA USAR UMA VARIAVEL DEFINIDA AQUI NA REGISTER ACTION
        bit<32> enq_qdepth;
        value_t EWMA;
        bit<32> EWMA_enq_qdepth;
        bit<16> rand_classic;
        bit<16> rand_l4s;

        //IPG
        bit<32> IPG;

        //Hash<bit<8>>(HashAlgorithm_t.CRC8) decider_hash; //2 to the power of 8 possible flows
        Hash<bit<8>>(HashAlgorithm_t.CRC8) decider_hash_tcp; //2 to the power of 8 possible flows
        Hash<bit<8>>(HashAlgorithm_t.CRC8) decider_hash_udp; //2 to the power of 8 possible flows

        
        Counter<bit<32>, bit<32>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES, CounterType_t.PACKETS) mark_ecn_pkt;
        Counter<bit<32>, bit<32>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES, CounterType_t.PACKETS) thresholdPkts;
        //Counter<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES, CounterType_t.PACKETS) totalPkts;
        //Register<bit<16>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) qdelay_classic;

        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES+NUMBER_OF_QUEUES) qdelay_l4s;
        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES+NUMBER_OF_QUEUES) switchLatency;
        Register<bit<16>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) dropProbability;
        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) target_violation;
        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) totalPkts;

        //IPG
        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) previous_timestamp_reg;
        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) IPG_reg;
        //PACKET SIZE
        Register<bit<16>, bit<16>>(NUMBER_OF_QUEUES+NUMBER_OF_QUEUES) packet_size_reg;
        //TOTALPACKETS TO IPG AND PACKETSIZE
        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) totalPkts_IPG; //used also for Packet Size

        Random<bit<16>>() rand;
        // MathUnit<bit<16>>(MathOp_t.DIV, 1) right_shift;


        // Declaração do LPF com tamanho para 1024 índices
        //Lpf<bit<32>, bit<16>>(size=1024) queue_delay_lpf;

        
        
        
        Register<enq_qdepth_v, bit<16>>(NUMBER_OF_QUEUES+NUMBER_OF_QUEUES) enq_qdepth_reg;
        Register<bit<16>, bit<16>>(1) index_reg;
        Register<bit<32>, _>(N_PORTS) timestamp_ifg_reg;
        Register<bit<16>, _>(N_PORTS) frame_size_reg;
        Register<bit<32>, _>(N_PORTS) timestamp_ipg_reg;


        RegisterAction<bit<32>, bit<8>, bit<32>>(timestamp_ipg_reg) update_timestamp_ipg_reg = {
            void apply(inout bit<32> value, out bit<32> result) {
                result = value;
                value = eg_md.metadata_ipg_temp;
            }
        }; 



        RegisterAction<bit<16>, bit<8>, bit<16>>(frame_size_reg) reset_frame_size_reg = {
            void apply(inout bit<16> value, out bit<16> result) {
                result = value + hdr.udp.length_;
                value = 0;
            }
        }; 


        RegisterAction<bit<16>, bit<8>, bit<16>>(frame_size_reg) increase_frame_size_reg = {
            void apply(inout bit<16> value) {
                value = value + hdr.udp.length_;
            }
        };


        
        RegisterAction<bit<32>, bit<8>, bit<32>>(timestamp_ifg_reg) update_timestamp_ifg_reg = {
            void apply(inout bit<32> value, out bit<32> result) {
                result = value;
                value = eg_md.metadata_ifg_temp;
            }
        }; 


        RegisterAction<bit<16>, bit<16>, bit<16>>(packet_size_reg) increase_packet_size = {
            void apply(inout bit<16> value, out bit<16> result) {
                value = value + hdr.ipv4.total_len;
                result = value + hdr.ipv4.total_len;;
            }
        };        

        RegisterAction<bit<32>, bit<16>, bit<32>>(previous_timestamp_reg) store_previous_timestamp = {
            void apply(inout bit<32> value, out bit<32> result) {
                result = value;
                value = eg_intr_md_from_prsr.global_tstamp[31:0];
            }
        };

        RegisterAction<bit<32>, bit<16>, bit<32>>(IPG_reg) increase_IPG = {
            void apply(inout bit<32> value, out bit<32> result) {
                bit<32> avg_temp;
              
                avg_temp =  (IPG + value);
               
                // update register        
                value = avg_temp;
                result = avg_temp;
            }
        };


        
        RegisterAction<bit<16>, bit<16>, bit<16>>(index_reg) store_index_reg = {
            void apply(inout bit<16> value, out bit<16> result) {
                result = eg_md.metadata_index;
                value = eg_md.metadata_index;
            }
        };
        
        //INT

        RegisterAction<bit<32>, bit<16>, bit<32>>(totalPkts) reset_totalPkts = {
            void apply(inout enq_qdepth_v value, out enq_qdepth_v result) {
                result = value;
                value = 0;
            }
        };

        RegisterAction<bit<32>, bit<16>, bit<32>>(totalPkts) increase_totalPkts = {
            void apply(inout enq_qdepth_v value, out enq_qdepth_v result) {
                value = value + 1;
                result = value +1;
            }
        };

        RegisterAction<bit<32>, bit<16>, bit<32>>(totalPkts_IPG) increase_totalPkts_IPG = {
            void apply(inout bit<32> value, out bit<32> result) {
                value = value + 1;
                result = value +1;
            }
        };
        

        // RegisterAction<enq_qdepth_v, bit<16>, enq_qdepth_v>(enq_qdepth_reg) read_enq_qdepth_reg_RegisterAction = {
        //     void apply(inout enq_qdepth_v value, out enq_qdepth_v result) {
        //         result = value;
        //         value = 0;
        //     }
        // };
        
        
        // RegisterAction<enq_qdepth_v, bit<16>, enq_qdepth_v>(enq_qdepth_reg) write_enq_qdepth_reg_RegisterAction = {
        //     void apply(inout enq_qdepth_v value, out enq_qdepth_v result) {

        //         bit<32> avg_temp;
              
        //         avg_temp =  enq_qdepth + value;
               
        //         // update register        
        //         value = avg_temp;
        //         result = avg_temp;
        //     }
        // };

        

        // RegisterAction<bit<16>, bit<16>, bit<16>>(qdelay_classic) qdelay_classic_action = {
        //     void apply(inout bit<16> value, out bit<16> result) {
        //         //* Compute Exponentially-Weighted Mean Average (EWMA) of queue delay *//
        //         // EWMA = alpha*qdelay + (1 - alpha)*previousEWMA
        //         // We use alpha = 0.5 such that multiplications can be replaced by bit shifts
        //         bit<16> avg_temp;
                
        //         avg_temp =  queue_delay + value;
               
        //         // update register        
        //         value = avg_temp;
        //         result = avg_temp;
        //     }

        // };

        
        RegisterAction<bit<32>, bit<16>, bit<32>>(qdelay_l4s) qdelay_l4s_action = {
            void apply(inout bit<32> value, out bit<32> result) {
                
                    
                //Compute Exponentially-Weighted Mean Average (EWMA) of queue delay 
                // EWMA = alpha*qdelay + (1 - alpha)*previousEWMA
                // We use alpha = 0.5 such that multiplications can be replaced by bit shifts
                bit<32> avg_temp;
              
                avg_temp =  (queue_delay + value);
               
                // update register        
                value = avg_temp;
                result = avg_temp;
              

            }

        };
        
        
        RegisterAction<bit<32>, bit<16>, bit<32>>(qdelay_l4s) read_qdelay_l4s_action = {
            void apply(inout bit<32> value, out bit<32> result) {
                result = value;
                value = 0;

            }

        };
        

        
        RegisterAction<bit<16>, bit<16>,bool>(dropProbability) getProb_l4s = {
            void apply(inout bit<16> value, out bool result){

                if (rand_l4s < value){
                    
                    value = value - 1;
                    result = true;

                }else{

                    value = value + 1;
                    result = false;
                
                }
            }
        };
        
        // RegisterAction<bit<16>, bit<16>, bool>(dropProbability) getProb_classic = {
        //     void apply(inout bit<16> value, out bool result){
        //         if (rand_classic < value){

        //             value = value - 1;
        //             result = true;
               // temp_index
        //         }else{

        //             value = value + 1;
        //             result = false;
                
        //         }
        //     }
        // };

        Register<bit<8>, _> (1) flow_id_reg;

        RegisterAction<bit<8>, bit<8>, bit<8>>(flow_id_reg) store_flow_id_reg = {
            void apply(inout bit<8> value) {
                value = eg_md.metadata_flowID ;

            }
        };
        
        RegisterAction<bit<32>, bit<16>, bit<32>>(target_violation) compute_target_violations = {
            void apply(inout bit<32> value, out bit<32> violation){

                value = EWMA;
                
                // No drop 
                if (value <= TARGET_DELAY_L4S){
                    
                    violation = 0;
                }

                //Maybe drop 
                if ((value > TARGET_DELAY_L4S) && (value < TARGET_DELAY_L4S_DOUBLE)){
                    
                    violation = 1;

                }

                //Drop 
                if (value > TARGET_DELAY_L4S_DOUBLE){

                    violation = 2;

                }

            }
        };    

    action find_flowID_ipv4_udp(){
        eg_md.metadata_flowID = decider_hash_udp.get({hdr.ipv4.dst_addr, hdr.udp.dstPort, hdr.ipv4.protocol});
    }

    action find_flowID_ipv4_tcp(){
        // bit<1> base = 0;
        // bit<16> max = 0xffff;
        // bit<16> hash_result;
        // //bit<48> IP_Port = hdr.ipv4.dstAddr ++ hdr.udp.dstPort;
        // bit<48> IP_dst_add = hdr.ipv4.dstAddr;
        // bit<48> UDP_dst_port = hdr.udp.dstPort;
        // bit<8> IP_Proto = hdr.ipv4.proto;
        // hash(
        //      hash_result,
        //      HashAlgorithm.crc8,
        //      base,
        //      {
        //         IP_Port, UDP_dst_port, IP_Proto
        //      },
        //      max
        //      );

        //bit<48> concatenated_hash_input = (bit<48>) (hdr.ipv4.dst_addr ++ hdr.udp.dstPort);

        eg_md.metadata_flowID = decider_hash_tcp.get({hdr.ipv4.dst_addr, hdr.tcp.dstPort, hdr.ipv4.protocol});
        //ig_md.metadata_flowID = decider_hash.get({concatenated_hash_input});
        // bit<8> temp_index = 0;
        // store_flow_id_reg.execute(temp_index);
    }
        
        

        action decisionMirror(){
            hdr.mirror.egress_port = MIRROR_PORT;
            //hdr.mirror.egress_port = eg_intr_md.egress_port;
            hdr.mirror.header_info = 1;
            hdr.mirror.mirror_session = 1;
            eg_intr_dprs_md.mirror_type = HEADER_TYPE_MIRROR_EGRESS;
            //hdr.mirror.IPG = IPG; THIS IS THE ACCUMMULATIVE ONE
            //hdr.mirror.egress_global_tstamp = (bit<48>)eg_intr_md_from_prsr.global_tstamp;
            eg_intr_dprs_md.mirror_io_select = 1;
            hdr.mirror.testing = 99;
            hdr.mirror.flowid = eg_md.metadata_flowID;
            hdr.mirror.packet_size = hdr.ipv4.total_len;
            hdr.mirror.frame_size = eg_md.metadata_frame_size;
            hdr.mirror.ifg = eg_md.metadata_ifg;
            hdr.mirror.IPG = eg_md.metadata_ipg;
            hdr.mirror.rtp_marker = hdr.rtp.marker;
            hdr.mirror.has_rtp = eg_md.metadata_has_rtp;
            hdr.mirror.queue_delay = eg_md.metadata_queue_delay_new;
        }


        //INT
        
        RegisterAction<bit<32>, bit<16>, bit<32>>(switchLatency) switchLatencyAction = {
            void apply(inout bit<32> value, out bit<32> result) {
       
                value = queue_delay;

            }

        };    

        // action define_index_for_registers_action(bit<16> index){
        //     eg_md.metadata_index = index;
        //     //bit<16> nada = store_index_reg.execute(0); //n da fala que gera muitos stages
        // }

        action _drop(){
            eg_intr_dprs_md.drop_ctl = 1;
        }



        // table define_index_for_registers_table{
        //     key = {
        //         hdr.bridge.bridge_qid: exact;
        //         eg_intr_md.egress_port: exact;
        //     }
        //     actions = {
        //         define_index_for_registers_action;
        //         _drop;
        //         NoAction;
        //     }
        //     size = 1024;
        // }



        //TESTE
        // action add_swtrace(){
        //     hdr.nodeCount.count = hdr.nodeCount.count + 1;
        //     hdr.INT.push_front(1);
        //     hdr.INT[0].setValid();
        //     hdr.INT[0].ingress_mac_tstamp = eg_md.metadata_ingress_mac_tstamp;
        //     hdr.INT[0].ingress_global_tstamp = eg_md.metadata_ingress_global_tstamp;
        //     hdr.INT[0].egress_global_tstamp = eg_md.metadata_egress_global_tstamp;
        //     hdr.INT[0].queue_delay = eg_md.metadata_egress_queue_delay;
        //     hdr.ipv4.total_len = hdr.ipv4.total_len + 16; //32 + 32 + 32 + 32 = 16
        // }


        action add_swtrace(){
            // enq_qdepth_v enq_qdepth_avg;
            // deq_timedelta_v deq_timedelta_avg;
            // deq_qdepth_v deq_qdepth_avg;
            // deq_timedelta_v processing_time_avg;

            

            // enq_qdepth_reg.read(enq_qdepth_avg, (bit<32>)reg_index);
            //enq_qdepth_avg = read_enq_qdepth_reg_RegisterAction.execute(eg_md.metadata_index); //o read zera

            // deq_timedelta_reg.read(deq_timedelta_avg, (bit<32>)reg_index);
            // deq_qdepth_reg.read(deq_qdepth_avg, (bit<32>)reg_index);
            // processing_time_reg.read(processing_time_avg, (bit<32>)reg_index);
            
            hdr.nodeCount.count = hdr.nodeCount.count + 1;
            hdr.INT.push_front(1);
            hdr.INT[0].setValid();
            //1 para ida, 2 para volta
            
            hdr.INT[0].ingress_port = (bit<9>)hdr.bridge.bridge_ingress_port;
            //hdr.INT[0].egress_port = eg_intr_md.egress_port;
            // hdr.INT[0].egress_spec = (egressSpec_v)standard_metadata.egress_spec;
            // hdr.INT[0].priority = (priority_v)standard_metadata.priority;
            //hdr.INT[0].qid = eg_intr_md.egress_qid;
            hdr.INT[0].qid = (bit<7>)hdr.bridge.bridge_qid;
            // hdr.INT[0].ingress_global_timestamp = (ingress_global_timestamp_v)standard_metadata.ingress_global_timestamp;
            hdr.INT[0].egress_global_timestamp = (bit<32>)eg_intr_md_from_prsr.global_tstamp;
            // hdr.INT[0].enq_timestamp = (enq_timestamp_v)standard_metadata.enq_timestamp;
            hdr.INT[0].enq_qdepth = eg_md.metadata_enq_qdepth;
            // hdr.INT[0].deq_timedelta = (deq_timedelta_v)deq_timedelta_avg;
            // hdr.INT[0].deq_qdepth = (deq_qdepth_v)deq_qdepth_avg;
            hdr.INT[0].processing_time = eg_md.metadata_queue_delay;
            hdr.INT[0].number_of_packets_for_average = eg_md.metadata_totalPkts;

            hdr.ipv4.total_len = hdr.ipv4.total_len + 37;

            // enq_qdepth_avg = 0;
            // deq_timedelta_avg = 0;
            // deq_qdepth_avg = 0;
            // processing_time_avg = 0;

            // enq_qdepth_reg.write((bit<32>)reg_index, enq_qdepth_avg);
            // deq_timedelta_reg.write((bit<32>)reg_index, deq_timedelta_avg);
            // deq_qdepth_reg.write((bit<32>)reg_index, deq_qdepth_avg);
            // processing_time_reg.write((bit<32>)reg_index, processing_time_avg);
        }
       
    apply {
        //* Only regular pkts *//a
        // if (eg_intr_md.egress_port != recirc_port){   
        
        //egress_intrinsic_metadata_t   
        //loopbackport
        //hdr.bridge.bridge_ingress_port
       //hdr.bridge.bridge_qid
        // if (eg_intr_md.egress_port == 136){ //136 = supermario/superppeach
        //     //QueueId_t = 7bits IN TOFINO2
        //     //hdr.bridge.bridge_qid
        //     eg_md.metadata_index = (bit<16>)eg_intr_md.egress_qid ; 
        // } else if(eg_intr_md.egress_port == 137) { //superluigi = 137
        //     eg_md.metadata_index = (bit<16>)eg_intr_md.egress_qid + NUMBER_OF_QUEUES;
        // }


        //OPERACAO MT COMPLEXA DE INDEX
        // bit<16> temp_porta;
        // if(eg_intr_md.egress_port == 136){
        //     temp_porta = 0;
        // }
        // else{
        //     temp_porta = 1;
        // }
        // bit<16> temp_qid = hdr.bridge.bridge_qid << 1;
        // eg_md.metadata_index = temp_qid + temp_porta;



        // if(hdr.bridge.bridge_qid == 0){
        //     if(hdr.bridge.bridge_ingress_port == 137){ //VEIO do superluigi (a saida eh 136)
        //         eg_md.metadata_index = 0;
        //     }
        //     else{ //VEIO do supermario ou loopback (a saida eh 137 ou 136 na recirculacao)
        //         if(hdr.nodeCount.isValid()){ 
        //             //depende se vem do 136 ou da looback
        //             if(hdr.bridge.bridge_ingress_port == 136){ //se vem da 136, vai em relacao a outra porta
        //                 eg_md.metadata_index = 1;
        //             }
        //             else{
        //                 //porta looback, vai ler agoa na volta a 136
        //                 eg_md.metadata_index = 0;
        //             }
                    
        //         }
        //         else{
        //             //vindo do 136 indo pro 137
        //             eg_md.metadata_index = 1;
        //         }
        //     }
        // }
        // else if(hdr.bridge.bridge_qid == 1){
        //     if(hdr.bridge.bridge_ingress_port == 137){ //VEIO do superluigi (a saida eh 136)
        //         eg_md.metadata_index = 2;
        //     }
        //     else{ //VEIO do supermario ou loopback (a saida eh 137 ou 136 na recirculacao)
        //         if(hdr.nodeCount.isValid()){ 
        //             //depende se vem do 136 ou da looback
        //             if(hdr.bridge.bridge_ingress_port == 136){ //se vem da 136, vai em relacao a outra porta
        //                 eg_md.metadata_index = 3;
        //             }
        //             else{
        //                 //porta looback, vai ler agoa na volta a 136
        //                 eg_md.metadata_index = 2;
        //             }
                    
        //         }
        //         else{
        //             //vindo do 136 indo pro 137
        //             eg_md.metadata_index = 3;
        //         }
        //     }
        // }




        // if(hdr.bridge.bridge_qid == 0){
        //     if(eg_intr_md.egress_port == 136){ 
        //         //tem 2 casos: com int e sem int
        //         // if(nodeCount.isValid()){
        //         //     eg_md.metadata_index = 0x00;
        //         // }
        //         eg_md.metadata_index = 0x00;
        //     }
        //     else if(eg_intr_md.egress_port == 137){
        //         eg_md.metadata_index = 0x01;
        //     }
        //     else{ //loopbackport (com int)
        //         eg_md.metadata_index = 0x01;
        //     }
        // }
        // else if(hdr.bridge.bridge_qid == 1){
        //     if(eg_intr_md.egress_port == 136){ 
        //         //tem 2 casos: com int e sem int
        //         // if(nodeCount.isValid()){
        //         //     eg_md.metadata_index = 0x02;
        //         // }
        //         eg_md.metadata_index = 0x02;
        //     }
        //     else if(eg_intr_md.egress_port == 137){
        //         eg_md.metadata_index = 0x03;
        //     }
        //     else{//loopbackport com int
        //         eg_md.metadata_index = 0x03;
        //     }
        // }

        //using a table to define the index
        //N DA PRA USAR ISSO AQUI POR CAUSA DO LOOPBACK PORT QUE EU NAO SEI
        //define_index_for_registers_table.apply();

        


        
        if(hdr.bridge.bridge_qid == 0){
            if(eg_intr_md.egress_port == 136){ 
                //tem 2 casos: com int e sem int
                // if(nodeCount.isValid()){
                //     eg_md.metadata_index = 0x00;
                // }
                eg_md.metadata_index = 0x00;
            }
            else if(eg_intr_md.egress_port == 137){
                eg_md.metadata_index = 0x01;
            }
            else{ //loopbackport (com int)
                eg_md.metadata_index = 0x01;
            }
        }
        else if(hdr.bridge.bridge_qid == 1){
            if(eg_intr_md.egress_port == 136){ 
                //tem 2 casos: com int e sem int
                // if(nodeCount.isValid()){
                //     eg_md.metadata_index = 0x02;
                // }
                eg_md.metadata_index = 0x02;
            }
            else if(eg_intr_md.egress_port == 137){
                eg_md.metadata_index = 0x03;
            }
            else{//loopbackport com int
                eg_md.metadata_index = 0x03;
            }
        }
        else if(hdr.bridge.bridge_qid == 2){
            if(eg_intr_md.egress_port == 136){ 
                //tem 2 casos: com int e sem int
                // if(nodeCount.isValid()){
                //     eg_md.metadata_index = 0x02;
                // }
                eg_md.metadata_index = 0x04;
            }
            else if(eg_intr_md.egress_port == 137){
                eg_md.metadata_index = 0x05;
            }
            else{//loopbackport com int
                eg_md.metadata_index = 0x05;
            }
        }


        bit<16> nada = store_index_reg.execute(0); 




        if(hdr.nodeCount.isValid()){
            //eg_md.metadata_enq_qdepth = read_enq_qdepth_reg_RegisterAction.execute(eg_md.metadata_index);
            eg_md.metadata_queue_delay = read_qdelay_l4s_action.execute(eg_md.metadata_index); 
            eg_md.metadata_totalPkts = reset_totalPkts.execute(eg_md.metadata_index);
            add_swtrace();
            if (hdr.nodeCount.count == 1){
                hdr.INT[0].swid = 1;
            } else {
                hdr.INT[0].swid = 2;
            }
            hdr.bridge.setInvalid();
            //totalPkts.count(temp_index);
        }
        else{
            //save telemetry average
            //Counter of packets
            //bit<16> temp_index = eg_md.metadata_index;
            //totalPkts.count(temp_index);
            
            
            //if(hdr.ipv4.ecn == 1 || hdr.ipv4.ecn == 2){//n preciso checar. so vai pro index correto e boa
            increase_totalPkts.execute(eg_md.metadata_index);
            //}
            
            
            /*enq qdepth avg*/
            //bit<32> EWMA_enq_qdepth_temp;
            // if(hdr.ipv4.ecn == 1 || hdr.ipv4.ecn == 2){
            //     EWMA_enq_qdepth_temp = write_enq_qdepth_reg_RegisterAction.execute(eg_md.metadata_index);
            // }

            // EWMA_enq_qdepth = EWMA_enq_qdepth_temp>>1; //alpha = 0.5
            // eg_md.metadata_enq_qdepth = EWMA_enq_qdepth;

            //* Compute queue delay *//
            queue_delay = (value_t)eg_intr_md_from_prsr.global_tstamp - (value_t)hdr.bridge.ingress_global_tstamp;
            hdr.bridge.setInvalid();

            //switchLatencyAction.execute((value_t)eg_intr_md.egress_port); //sei la pra q isso

            bit<32> EWMA_temp;
            
            //if (hdr.ipv4.l4s == 1){
            //if(hdr.ipv4.ecn == 1 || hdr.ipv4.ecn == 2){ //n preciso checar, so ir pro correto (inclui classic)
            EWMA_temp = qdelay_l4s_action.execute(eg_md.metadata_index);
            //}
            //else{

            //     EWMA_temp = qdelay_classic_action.execute((value_t)eg_intr_md.egress_port);
            // }
            
            EWMA = EWMA_temp>>1; //o original ta comentado aqui msm (?)
            //eg_md.metadata_queue_delay = (bit<32>)EWMA; //OBS: o valor aqui ta com 16 bits em, ob2 n sei se tem pra q isso se n ta no int

            //EWMA = queue_delay;


            //INTER PACKET GAP
            bit<32> previous_timestamp = store_previous_timestamp.execute(eg_md.metadata_index);
            IPG = eg_intr_md_from_prsr.global_tstamp[31:0] - previous_timestamp; //previous = 32bits
            increase_IPG.execute(eg_md.metadata_index);
            increase_totalPkts_IPG.execute(eg_md.metadata_index);

            //PACKET SIZE
            increase_packet_size.execute(eg_md.metadata_index);












            
            //* Check if the queue delay reach the target limit *//
            //* 0 = no drop    *//
            //* 1 = Maybe drop *//
            //* 2 = drop       *//

            
            bit<32> target_violations;
            target_violations = compute_target_violations.execute(eg_md.metadata_index);

            if (target_violations == 1){
                
                //Counter of packets
                thresholdPkts.count((bit<32>)eg_md.metadata_index);

                //* rand_classic is a is a random number, used to compute the new drop probability of Classic flows *//
                //** For each new pkt with average queue occupancy between MinTh-MaxTh, the drop probability will be a double *// 
                rand_classic = rand.get();
                rand_l4s = rand_classic >> 1; //Coupling
                
                bool drop_decision_l4s;
                bool drop_decision_classic;

                //if (hdr.ipv4.l4s ==1){
                if(hdr.ipv4.ecn == 1 || hdr.ipv4.ecn == 2){ //MARKING L4S + CG QUEUE

                    drop_decision_l4s = getProb_l4s.execute(eg_md.metadata_index);
                    
                    if (drop_decision_l4s == true){
                        
                        mark_ecn_pkt.count((bit<32>)eg_md.metadata_index);
                        //hdr.ipv4.ecn = 3;    
                    
                    } 

                }
                // else{

                //     drop_decision_classic = getProb_classic.execute((bit<16>)eg_intr_md.egress_port);

                //     if (drop_decision_classic == true){
                    
                //         decisionMirror();

                //     }

                // }
                
            }
            else if (target_violations == 2){
        
                //if (hdr.ipv4.l4s == 1){
                if(hdr.ipv4.ecn == 1 || hdr.ipv4.ecn == 2){ //MARKING L4S AND CG
                        mark_ecn_pkt.count((bit<32>)eg_md.metadata_index);
                        //hdr.ipv4.ecn = 3;

                }
                // else{

                //         //decisionMirror();

                // }
                    
            }

        }
        

        if (hdr.ipv4.isValid() && hdr.ipv4.protocol == PROTO_UDP) {
            find_flowID_ipv4_udp();
        }
        else if(hdr.ipv4.isValid() && hdr.ipv4.protocol == PROTO_TCP){
            find_flowID_ipv4_tcp();
        }



        //IPG 

        eg_md.metadata_ipg_temp = (bit<32>) eg_intr_md_from_prsr.global_tstamp;
        bit<32> previous_tstamp_ipg = update_timestamp_ipg_reg.execute(eg_md.metadata_flowID);

        eg_md.metadata_ipg = (bit<32>)eg_intr_md_from_prsr.global_tstamp - previous_tstamp_ipg;
        

        //IFG AND FRAME SIZE
        //SAVE TIMESTAMP OF THE EGRESS WHEN MARKER IS 1 AND JUST READSUBSTITUTE AGAIN WHEN A NEW MARKER
        if(hdr.rtp.isValid() ){
            if(hdr.rtp.marker == 1){
                if(hdr.rtp.version == RTP_VERSION){
                    eg_md.metadata_has_rtp = 1;
                    // STORE THE TIMESTAM
                    eg_md.metadata_ifg_temp = (bit<32>)eg_intr_md_from_prsr.global_tstamp;
                    bit<32> previous_tstamp_ifg;
                    previous_tstamp_ifg = update_timestamp_ifg_reg.execute(eg_md.metadata_flowID);
                    eg_md.metadata_ifg = (bit<32>)eg_intr_md_from_prsr.global_tstamp - previous_tstamp_ifg;

                    //AND SEND PA

                    eg_md.metadata_frame_size = reset_frame_size_reg.execute(eg_md.metadata_flowID);
                }
                else{
                    eg_md.metadata_frame_size = 0;
                    eg_md.metadata_ifg = 0;
                }

            }
            else{
                if(hdr.rtp.version == RTP_VERSION){
                    eg_md.metadata_ifg = 0;
                    eg_md.metadata_frame_size = 0;
                    increase_frame_size_reg.execute(eg_md.metadata_flowID);
                    eg_md.metadata_has_rtp = 1;
                }
                else{
                    eg_md.metadata_ifg = 0;
                    eg_md.metadata_frame_size = 0;
                }
            }
        }
        else{
            eg_md.metadata_frame_size = 0;
            eg_md.metadata_ifg = 0;
        }

        eg_md.metadata_queue_delay_new = (bit<32>) eg_intr_md_from_prsr.global_tstamp - (bit<32>)hdr.bridge.ingress_global_tstamp;

        hdr.bridge.setInvalid(); //vai que
        if(eg_intr_md.egress_port != MIRROR_PORT && eg_intr_md.egress_port != recirc_port){
            decisionMirror();
        }

        
     }
}

control EgressDeparser(
        packet_out pkt,
        inout headers_t hdr,
        in metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t eg_intr_dprs_md) {
    
    Mirror() mirror;
    Checksum() ipv4_checksum;
    
    apply {
        if (eg_intr_dprs_md.mirror_type == HEADER_TYPE_MIRROR_EGRESS){
            
            mirror.emit<mirror_h>(hdr.mirror.mirror_session, {hdr.mirror.header_type, 
                                      hdr.mirror.header_info,
                                      hdr.mirror.egress_port,
                                      hdr.mirror.mirror_session,
                                      hdr.mirror.IPG,
                                      hdr.mirror.testing,
                                      hdr.mirror.flowid,
                                      hdr.mirror.ifg,
                                      hdr.mirror.packet_size,
                                      hdr.mirror.frame_size,
                                      hdr.mirror.rtp_marker,
                                      hdr.mirror.queue_delay,
                                      hdr.mirror.has_rtp
                                      //hdr.mirror.egress_global_tstamp
                                      });
        
        }

        hdr.ipv4.hdr_checksum = ipv4_checksum.update({
            hdr.ipv4.version,
            hdr.ipv4.ihl,
            hdr.ipv4.dscp,
            //hdr.ipv4.l4s,
            hdr.ipv4.ecn,
            hdr.ipv4.total_len,
            hdr.ipv4.identification,
            hdr.ipv4.flags,
            hdr.ipv4.frag_offset,
            hdr.ipv4.ttl,
            hdr.ipv4.protocol,
            hdr.ipv4.src_addr,
            hdr.ipv4.dst_addr});
        
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
