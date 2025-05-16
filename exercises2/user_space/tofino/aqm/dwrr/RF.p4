// Automatic AR/CG Flow Classification & Packet Marking + L4S Queue (CoDel AQM)
// Authors: Mateus, Alireza Shirmarz
// Date: 2025-04-15
// Netsoft 2025 
/* It is Including 
    (1) Definition & Headers, 
    (2) Ingress (Parser/Ingress (MA) Processing/Deparser), 
    (3) Egress (Parser/Ingress (MA) Processing/Deparser)
*/
/*
==============================================================================================================
******************************************* (1) Definition & Headers *****************************************
==============================================================================================================
*/
#include <core.p4>
#include <t2na.p4>
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

//Low Pass Filter EWMA Type 
typedef bit<16> lpf_type_16;
typedef bit<32> lpf_type_32;

/*** INT TYPE ***/
typedef bit<6> switchID_v; 
typedef bit<9> ingress_port_v;
typedef bit<9> egress_port_v;
typedef bit<9> egressSpec_v;
typedef bit<7> qid_v;

// Timestamp is 48 bit but store 32 bits
typedef bit<32>  ingress_global_timestamp_v; 
typedef bit<32>  egress_global_timestamp_v; 
typedef bit<32>  enq_timestamp_v;
typedef bit<32> enq_qdepth_v;           //the value stored in the metadata is 19
typedef bit<32> deq_timedelta_v;
//typedef bit<3>  priority_v;
typedef bit<32> deq_qdepth_v;

/*** Constants ***/
const number_of_ports_t N_PORTS                = 512; 
const header_type_t HEADER_TYPE_NORMAL_PKT     = 0;
const header_type_t HEADER_TYPE_MIRROR_EGRESS  = 1;
const header_type_t HEADER_TYPE_MIRROR_INGRESS  = 2;
const ether_type_t ETHERTYPE_IPV4              = 16w0x0800;

// CoDel AQM for EC Marking (Min/Max in Nano Seconds)
const value_t CoDel_Min                     = 2000000; 
const value_t CoDel_Max              = 10000000; 

// Recirculation for INT and Mirroring Packets
const ports_t recirc_port                        = 6;   // Recirculation INT 
const ports_t MIRROR_PORT                        = 128; // Mirroring Packets

const bit<8> PROTO_INT = 253;                           // ipv4.protocol for INT in Ingress Packet Parsing
const bit<8> PROTO_UDP = 17;                            // ipv4.protocol for UDP in Ingress Packet Parsing
const bit<8> PROTO_TCP = 6;                             //ipv4.protocol  for TCP in Ingress Packet Parsing
const bit<2> RTP_VERSION = 2;

// Internal Header for Mirroring
#define INTERNAL_HEADER         \
    header_type_t header_type;  \
    header_info_t header_info

// Reserved Variable 
#define MAX_HOPS 10
// Number of Queues 
    // L4S: 0 
    // Classic: 1
#define NUMBER_OF_QUEUES 2 //CLASSIC L4S AND CG

/*** Packet Headers ***/
/*
The headers were parsed/deparsed:

    (1) Ethernet [14 Byte]
    (2) IPv4 [20 Byte]
    (3) UDP [8 Byte]  Or TCP [2 Byte]
    (4) RTP [12 Byte]
*/
// Ethernet
header ethernet_h {
    mac_addr_t dst_mac_addr;
    mac_addr_t src_mac_addr;
    bit<16> ether_type;
}

// IPv4
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

// UDP
header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

// TCP
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

// RTP
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

//*** Classification Result ***//
header classification_h {
    bit<8> metadata_classT1;
    bit<8> metadata_classT2;
    bit<8> metadata_classT3;
    bit<8> metadata_classT4;
    bit<8> metadata_classT5;
    bit<8> metadata_final_classification;
}


/*** Mirror Header to carry port metadata ***/
header mirror_h {
    INTERNAL_HEADER;
    @flexible PortId_t egress_port;
    @flexible MirrorId_t  mirror_session;
    @flexible bit<32> IPG;
    @flexible bit<8> flowid;
    @flexible bit<32> ifg;
    @flexible bit<16> packet_size;
    @flexible bit<16> frame_size;
    @flexible bit<1> rtp_marker; // RTP Marker is still 1 bit!
    @flexible bit<32> queue_delay;
    @flexible bit<8> has_rtp;
    @flexible bit<32> host_ifg;
    //@flexible bit<48> egress_global_tstamp;
}

//*** Bridge header to carry ingress timestamp from Ingress to Egress ***//
header bridge_h {
    bit<16> bridge_ingress_port; 
    bit<48> ingress_global_tstamp;
    bit<16> bridge_qid;
}

//*** Count the Hops ***//
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

// Struct of the headers
struct headers_t {
    bridge_h            bridge;
    mirror_h            mirror;
    ethernet_h          ethernet;
    ipv4_h              ipv4;
    tcp_t        tcp;
    udp_t        udp;
    rtp_t        rtp;

    nodeCount_h        nodeCount;
    InBandNetworkTelemetry_h[2] INT;
}

struct ingress_metadata_t {
    bit<16>  count;
}

struct parser_metadata_t {
    bit<16>  remaining;
}

struct Thresholds {
    bit<8> frame_thresh;
    bit<8> packet_thresh;
}

struct Counters {
    bit<8> frame_count;
    bit<8> packet_count;
}

struct metadata_t{
    classification_h classification; // Alireza 
    bridge_h    bridge;
    mirror_h    mirror;
    MirrorId_t  mirror_session;
    PortId_t    egress_port;
    header_type_t header_type;
    header_info_t header_info;
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
    bit<32> metadata_queue_delay_new;
    
    bit<16> metadata_frame_size;
    bit<16> metadata_packet_size;
    bit<8> metadata_flowID;  
    bit<8> metadata_rtp_marker;                                         // Although RTP Marker is 1 bit, it assigned 8 bit to handle easier in Pipeline!
    bit<8> metadata_has_rtp;
    bit<32> metadata_rtp_timestamp; //Alireza added
    bit<32> metadata_host_ifg; // Alireza added
    bit<20> metadata_ipg_20lsb;
    bit<20> metadata_ifg_20lsb;
}

// ****************************** (2) Ingress (Parser/Ingress (MA) Processing/Deparser) ******************************************//
/*
Normal Packet ===>	start → parse_port_metadata → parse_ethernet → ...
Mirrored Packet ===>	start → parse_mirror → parse_bridge → parse_ethernet → ...
INT-Tagged Packet ===>	parse_ipv4 → parse_count → parse_one/two_int
*/
// *** (2-1) Ingress Parsing ***//
parser IngressParser(
        packet_in pkt,                                              // Received Packets
        out headers_t hdr,                                          
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {
    
    // Start State in State Machine        
    state start {                                                    // Divide the parsing into 2: (Cloned:ingress port is 128 or 6) or (original packets)
       pkt.extract(ig_intr_md);
        transition select(ig_intr_md.ingress_port){
            (MIRROR_PORT): parse_mirror;                            // If ingress port is mirrorring (cloned), so go to parse mirror headers
            (_): parse_port_metadata;                               // If ingress port is other, so go to pars normal packets
            
        }
    }

    /*** Original PKTS ***/
    state parse_port_metadata{                                      // Original Packets go to Parse Ehterent State!
        pkt.advance(PORT_METADATA_SIZE);                            // Ignore Tofino intinsic Metadata
        transition parse_ethernet;
    }

    /*** Cloned or MIRRORED PKTS ***/
    state parse_mirror{                                             // Extract the Mirrorring header and transmit to bridge state
        pkt.advance(PORT_METADATA_SIZE);                            // Ignore Tofino intinsic Metadata
        pkt.extract(hdr.mirror);        
        transition parse_bridge;
    }

    //*** JUST FOR MIRRPRED PPACKETS ***/
    state parse_bridge{                                             // Extract the bridge header (holds original ingress info like timestamps) and go to Ethernet Header!
        pkt.extract(hdr.bridge);
        transition parse_ethernet;
    }

    
    // *** Ethernet Header Parsing ***//
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select (hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : accept;
        }
    }

    // *** IPv4 Header Parsing ***//
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {                          // Select next step: (TCP,6), (UDP,17), (INT,253)
            PROTO_UDP: parse_udp;
            PROTO_INT: parse_INT;
            PROTO_TCP: parse_tcp;
            //PROTO_ICMP: parse_icmp;
        default: accept;
      }
    }
    
    // *** TCP Header Parsing ***//
    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }

    // *** UDP Header Parsing ***//
    state parse_udp {
        pkt.extract(hdr.udp);
        transition parse_rtp;
    }

    // *** RTP Header Parsing ***//
    state parse_rtp {
        pkt.extract(hdr.rtp);
        transition accept;
    }
    
    // *** INT Header Parsing ***//
    state parse_INT{ // 
        pkt.extract(hdr.nodeCount);
        //transition accept;
        ig_md.parser_metadata.remaining = hdr.nodeCount.count;
        transition select(ig_md.parser_metadata.remaining) {
            2 : parse_two_INT;
            1: parse_one_INT;
            0: accept;
        }
    }

    // *** Two INT for loop ***// 
    state parse_two_INT{
        pkt.extract(hdr.INT.next);
        pkt.extract(hdr.INT.next);
        transition accept;
    }

    // *** One INT for loop ***//
    state parse_one_INT{
        pkt.extract(hdr.INT.next);
        transition accept;
    }
}
// *** (2-2) Ingress (Match/Action) Processing ***//
control Ingress( 
        inout headers_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // Stateful Variables For Clasification and Marking!
    Register<bit<8>, _> (1) flow_id_reg; //after LPF calculates the EWMA
    Register<bit<16>, _> (N_PORTS) packet_size_reg;

    Register<bit<32>, _> (0x100) ingress_timestamp_reg_every_pkt;
    Register<bit<16>, _> (0x100) packet_size_ewma_reg;
    Register<bit<16>, _> (0x100) frame_size_reg; //uses mathunit
    Register<bit<16>, _> (0x100) frame_size_ewma_reg; //after LPF calculates the EWMA
    Register<bit<32>, _> (0x100) ingress_host_ifg_reg; //Alireza added

    // lpf_type_16 is 16 bit and lpf_type_32 is 32 bit (defined in the global declaration) 

    // low pass filter variables for EWMA
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
    

    // Function to return the CRC8 of TCP/UDP Flow ID in 16 bits
    Hash<bit<8>>(HashAlgorithm_t.CRC8) hash_function_tcp; 
    Hash<bit<8>>(HashAlgorithm_t.CRC8) hash_function_udp; 

    
    // Write packet size from metadat into 'packet_size_reg'
    RegisterAction<bit<16>, bit<8>, bit<16>>(packet_size_reg) write_packet_size_reg= {
            void apply(inout bit<16> value) {
                value = ig_md.metadata_packet_size;                                         // Note: Packet Size is EWMA value!          
            }
    };
    // This is only for monitoring the Flow ID Values Stored in the Register!
    RegisterAction<bit<8>, bit<8>, bit<8>>(flow_id_reg) write_flow_id_reg = {
            void apply(inout bit<8> value) {
                value = ig_md.metadata_flowID;

            }
    };


     /*
    ======================================================================
    Shared Register is used for Synching the Flow Classification with Packet ECT(1) Marking
    ----------------------------------------------------------------------
    Shared Register <Flow_ID, Class), Class: 0,1,2,3
    Size = 8 bit
    Number of Cuncurrent Flows Capacity = 256
    ======================================================================
    */
    // Shared Register <Flow_ID. Class> 
    Register<bit<8>, _> (N_PORTS) shared_register;

    // Read Shared Register 
    RegisterAction<bit<8>, bit<8>, bit<8>>(shared_register) read_shared_register = {
            void apply(inout bit<8> value, out bit<8> output_value) {
                value = value;
                output_value = value;
            }

    };

    //Write the Final Class in Shared Register 
    RegisterAction<bit<8>, bit<8>, bit<8>>(shared_register)  write_shared_register = {
            void apply(inout bit<8> value) {
                value = (bit<8>) ig_md.classification.metadata_final_classification;  //ig_md.metadata_final_classification;
            }

    };

    //********************************************************************//
    // Update (Read/Write) the ingress timestamp
    RegisterAction<bit<32>, bit<8>, bit<32>>(ingress_timestamp_reg_every_pkt) update_ingress_timestamp_reg_every_pkt = {
            void apply(inout bit<32> value, out bit<32> result) {
                result = value; // We read the previous stored value in the register and return the previous stored value!
                value = ig_intr_prsr_md.global_tstamp[31:0]; // Set new value in the register!
            }
    };

    // Write host ifg from RTP Header!
        RegisterAction<bit<32>, bit<8>, bit<32>>(ingress_host_ifg_reg) write_ingress_host_ifg_reg = {
            void apply(inout bit<32> value) {
                value = hdr.mirror.host_ifg;
            }
    };

    // Write frame size in 'frame_size_reg'
    RegisterAction<bit<16>, bit<8>, bit<16>>(frame_size_reg) write_frame_size_reg = {
            void apply(inout bit<16> value, out bit<16> result) {
                value = ig_md.metadata_frame_size;
            }
    };

    // Read frame size from 'frame_size_reg'
    RegisterAction<bit<16>, bit<8>, bit<16>>(frame_size_reg) read_frame_size_reg = {
            void apply(inout bit<16> value, out bit<16> result) {
                //value = value;
                result = value;
            }
    };


    Register<bit<32>, _>(N_PORTS) drop_cloned_pkt;
    Register<bit<32>, _>(N_PORTS) counter_packets_frame;

    // Write the dropped packets for each interface (Increasing step by step when it is called!)
    RegisterAction<bit<32>, bit<8>, bit<32>>(drop_cloned_pkt) write_drop_cloned_pkt = {
            void apply(inout bit<32> value) {
                value = value+1;
            }
    };

    // Write the Marker bit counter for each interface (Increasing step by step when it is called!)
    RegisterAction<bit<32>, bit<8>, bit<32>>(counter_packets_frame) write_counter_packets_frame = {
            void apply(inout bit<32> value) {
                value = value+1;
            }
    };

    /*
    ====================================================================================
    Flow ID Extraction From the Ingress Packets
    ====================================================================================
    */
    // Write RTP Marker Header Value in 'rtp_marker_reg' 
    // Note: rtp_marker_reg is 8 bit while RTP Marker header is one bit! 
    Register<bit<8>, _> (N_PORTS) rtp_marker_reg;
    RegisterAction<bit<8>, bit<8>, bit<8>>(rtp_marker_reg) write_rtp_marker_reg = {
            void apply(inout bit<8> value) {
                value = ig_md.metadata_rtp_marker;
            }

    };

    // Register to read the IPG from cloned packets 
    Register<bit<32>, _> (N_PORTS) ipg_cloned_packets_reg;
    
    // Write IPG from Metadata to Register
    RegisterAction<bit<32>, bit<32>, bit<32>>(ipg_cloned_packets_reg) write_ipg_cloned_packets_reg = {
            void apply(inout bit<32> value) {
                value = ig_md.metadata_ipg;
            }

    };

    // Write IFG in the register 'ifg_reg'
    Register<bit<32>, _> (N_PORTS) ifg_reg;
    RegisterAction<bit<32>, bit<8>, bit<32>>(ifg_reg) write_ifg_reg = {
            void apply(inout bit<32> value) {
                value = ig_md.metadata_ifg;
            }

    };
    
    // Read the IFG register
    RegisterAction<bit<32>, bit<8>, bit<32>>(ifg_reg) read_ifg_reg = {
            void apply(inout bit<32> value, out bit<32> result) {
                value = value;
                result = value;
            }

    };

    // Queue Delay Accumulative Value for INT
    Register<bit<32>, _> (1) queue_delay;
    RegisterAction<bit<32>, bit<8>, bit<32>>(queue_delay) sum_queue_delay = {
            void apply(inout bit<32> value) {
                value = value + ig_md.metadata_queue_delay_new;
            }

    };


    // Write the Class of Tree1 in Register 'metadata_classT1'
    Register<bit<8>, _> (N_PORTS) metadata_classT1;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT1)  write_metadata_classT1 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT1;
            }

    };

    // Write the Class of Tree 2 in Register 'metadata_classT2'
    Register<bit<8>, _> (N_PORTS) metadata_classT2;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT2)  write_metadata_classT2 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT2;
            }

    };

    // Write the Class of Tree 3 in Register 'metadata_classT3'
    Register<bit<8>, _> (N_PORTS) metadata_classT3;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT3)  write_metadata_classT3 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT3;
            }

    };

    // Write the Class of Tree 4 in Register 'metadata_classT4'
    Register<bit<8>, _> (N_PORTS) metadata_classT4;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT4)  write_metadata_classT4 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT4;
            }

    };

    // Write the Class of Tree 5 in Register 'metadata_classT5'
    Register<bit<8>, _> (N_PORTS) metadata_classT5;
    RegisterAction<bit<8>, bit<8>, bit<8>>(metadata_classT5)  write_metadata_classT5 = {
            void apply(inout bit<8> value) {
                value = ig_md.classification.metadata_classT5;
            }

    };

    /*
    ========================================================================
    Actions Definition: drop pkts, extract flow ID, and Classification
    ========================================================================
    */
        // Define Drop Original Pkts 
    // action drop_regular_pkts(){
    //     ig_intr_dprsr_md.drop_ctl = 0x1;
    // }

    // Define Drop Cloned Pkts
    action drop_cloned_pkts(){
        ig_intr_dprsr_md.drop_ctl = 0x1;
    }

    // Define to Extract FlowID (IP.dst, Port.dst, Protocol) {UDP Packets}
    action extract_flowID_ipv4_udp(){
        ig_md.metadata_flowID = hash_function_udp.get({hdr.ipv4.dst_addr, hdr.udp.dstPort, hdr.ipv4.protocol});
    }

    // Define to Extract FlowID (IP.dst, Port.dst, Protocol) {TCP Packets}
    action extract_flowID_ipv4_tcp(){
        ig_md.metadata_flowID = hash_function_tcp.get({hdr.ipv4.dst_addr, hdr.tcp.dstPort, hdr.ipv4.protocol});
    }

    /*
    ###################################################################################
    Classification for each Tree (1-5) 
    ###################################################################################
    */

    //T1
    action classify_T1(bit<8> classify_result){
        ig_md.classification.metadata_classT1 = classify_result;
    }

    // action classify_T1_IPG(bit<8> classify_result){
    //     ig_md.classification.metadata_classT1 = classify_result;
    // }

    table table_T1_FS{
        key = {
            ig_md.metadata_frame_size: range;
        }
        actions = {
            classify_T1();
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
            classify_T1();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    //T2
    action classify_T2(bit<8> classify_result){
        ig_md.classification.metadata_classT2 = classify_result;
    }

    // action classify_T2_IPG(bit<8> classify_result){
    //     ig_md.classification.metadata_classT2 = classify_result;
    // }
    
    table table_T2_IFG{
        key = {
            ig_md.metadata_ifg_20lsb: range;
        }
        actions = {
            classify_T2();
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
            classify_T2();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    action classify_T3(bit<8> classify_result){
        ig_md.classification.metadata_classT3 = classify_result;
    }

    // action classify_T3_IFG(bit<8> classify_result){
    //     ig_md.classification.metadata_classT3 = classify_result;
    // }

    // action classify_T3_IPG(bit<8> classify_result){
    //     ig_md.classification.metadata_classT3 = classify_result;
    // }


    //T3
    table table_T3_FS{
        key = {
            ig_md.metadata_frame_size: range;
        }
        actions = {
            classify_T3();
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
            classify_T3();
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
            classify_T3();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    //T4
    action classify_T4(bit<8> classify_result){
        ig_md.classification.metadata_classT4 = classify_result;
    }
    // action classify_T4_IFG(bit<8> classify_result){
    //     ig_md.classification.metadata_classT4 = classify_result;
    // }
    // action classify_T4_FS(bit<8> classify_result){
    //     ig_md.classification.metadata_classT4 = classify_result;
    // }

    
    table table_T4_IFG{
        key = {
            ig_md.metadata_ifg_20lsb: range;
        }
        actions = {
            classify_T4();
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
            classify_T4();
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
            classify_T4();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    //T5
    action classify_T5(bit<8> classify_result){
        ig_md.classification.metadata_classT5 = classify_result;
    }

    // action classify_T5_IPG(bit<8> classify_result){
    //     ig_md.classification.metadata_classT5 = classify_result;
    // }

    table table_T5_FS{
        key = {
            ig_md.metadata_frame_size: range;
        }
        actions = {
            classify_T5();
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
            classify_T5();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    //Aggregation of Trees (majority)
    action trees_aggregation(bit<8> majority){
        //ig_md.metadata_final_classification = (bit<8>) majority;
        ig_md.classification.metadata_final_classification = (bit<8>) majority;
    }
    table table_majority{
        key = {
            ig_md.classification.metadata_classT1: exact;
            ig_md.classification.metadata_classT2: exact;
            ig_md.classification.metadata_classT3: exact;
            ig_md.classification.metadata_classT4: exact;
            ig_md.classification.metadata_classT5: exact;
        }
        actions = {
            trees_aggregation();
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }  
    




    apply {
    /*
    =============================================================================================
    Ingress Processing
    Input: (1) Original Packets or (2) Cloned Packets
    Output: (1) Marked Original Packets with ECT(1), (2) Classify the flow based on the Cloned Packets 
    =============================================================================================
    */   
    
    //******************************************* Check for cloned pkts (Port: 128) *******************************************//
    if (ig_intr_md.ingress_port == MIRROR_PORT){
        ig_md.metadata_flowID = hdr.mirror.flowid; 
        bit<8> temp_index = 0;
        write_flow_id_reg.execute(temp_index);

        ig_md.metadata_rtp_marker = (bit<8>) hdr.mirror.rtp_marker;
        write_rtp_marker_reg.execute(ig_md.metadata_flowID);

        //drop_cloned_pkt.count((bit<32>)ig_md.metadata_flowID); //maybe flowid?  (bit<32>)hdr.mirror.flowid
        write_drop_cloned_pkt.execute(ig_md.metadata_flowID);

        if(hdr.mirror.rtp_marker == 1){
            //counter_packets_frame.count(ig_md.metadata_flowID);
            write_counter_packets_frame.execute(ig_md.metadata_flowID);
        }
        //packet size
        lpf_input_packet_size = hdr.mirror.packet_size;
        lpf_output_packet_size = lpf_packet_size.execute(lpf_input_packet_size,0);
        //ig_md.metadata_packet_size = hdr.mirror.packet_size;
        ig_md.metadata_packet_size = lpf_output_packet_size;
        write_packet_size_reg.execute(ig_md.metadata_flowID);


        //bit<32> shit = 0;
        //ipg
        lpf_input_ipg = hdr.mirror.IPG;
        lpf_output_ipg = lpf_ipg.execute(lpf_input_ipg, 0);
        //ig_md.metadata_input_ipg = hdr.mirror.IPG;
        ig_md.metadata_ipg = lpf_output_ipg;
        write_ipg_cloned_packets_reg.execute((bit<32>)ig_md.metadata_flowID);


        //frame size
        if(hdr.mirror.frame_size != 0){
            lpf_input_frame_size = hdr.mirror.frame_size;
            lpf_output_frame_size = lpf_frame_size.execute(lpf_input_frame_size, 0);
            //ig_md.metadata_frame_size = hdr.mirror.frame_size;
            ig_md.metadata_frame_size = lpf_output_frame_size;
            write_frame_size_reg.execute(ig_md.metadata_flowID);
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
            write_ifg_reg.execute(ig_md.metadata_flowID);
        }
        else if(hdr.mirror.ifg == 0){
            ig_md.metadata_ifg = read_ifg_reg.execute(ig_md.metadata_flowID);
        }
        

        // IFG Extracted from RTP Header (Deterministic From Host)
        if (hdr.mirror.host_ifg!=0){
            //store_ingress_host_ifg_reg(address) without return
            write_ingress_host_ifg_reg.execute(ig_md.metadata_flowID);
        }
      
        // Accumulative Queue Delay (For monitoring the Queue Delay using INT, so each reading by INT remoe the Register!)
        // queue delay (previous + current. the collect_info python resets when read)
        ig_md.metadata_queue_delay_new = hdr.mirror.queue_delay;
        sum_queue_delay.execute(temp_index); // Register for Accumulative Latency


        // Flow Classification 
        ig_md.metadata_ifg_20lsb = ig_md.metadata_ifg[19:0];
        ig_md.metadata_ipg_20lsb = ig_md.metadata_ipg[19:0];

        // Note: Max variable size of the 'Range' for Tables is 20 bits, so we convert the features values received from 
        // cloned packets from 32 bits into 20 bits.

        // T1
        table_T1_FS.apply();
        if(ig_md.classification.metadata_classT1 == 0){
            table_T1_IPG.apply();
        }

        // T2
        table_T2_IFG.apply();
        if(ig_md.classification.metadata_classT2 == 0){
            table_T2_IPG.apply();
        }

        // T3
        table_T3_FS.apply();
        if(ig_md.classification.metadata_classT3 == 0){
            table_T3_IFG.apply();
        }
        if(ig_md.classification.metadata_classT3 == 0){
            table_T3_IPG.apply();
        }

        // T4
        table_T4_IFG.apply();
        if(ig_md.classification.metadata_classT4 == 5){
            table_T4_IPG.apply();
        }
        else if(ig_md.classification.metadata_classT4 == 6){
            table_T4_FS.apply();
        }

        // T5
        table_T5_FS.apply();
        if(ig_md.classification.metadata_classT5 == 0){
            table_T5_IPG.apply();
        }

        table_majority.apply();
        write_metadata_classT1.execute(ig_md.metadata_flowID);
        write_metadata_classT2.execute(ig_md.metadata_flowID);
        write_metadata_classT3.execute(ig_md.metadata_flowID);
        write_metadata_classT4.execute(ig_md.metadata_flowID);
        write_metadata_classT5.execute(ig_md.metadata_flowID);

        write_shared_register.execute(ig_md.metadata_flowID);

        drop_cloned_pkts();

    } // end of If block for Cloned Packets


    //******************************************* Check for Original pkts (Port: 128) *******************************************//
    else{
        /***
        Forwarding Rules 
        If Ingress is Interface(137) Then Forward Interface(136)
        If Ingress is Interface(136) Then Forward Interface(137)

        ***/
        // 1/1 -> 1/0
        if (ig_intr_md.ingress_port == 137) {
            ig_tm_md.ucast_egress_port = 136;
        }
        // 1/0 -> 1/1
        else if (ig_intr_md.ingress_port == 136) {
            if(hdr.nodeCount.isValid()){
                if(hdr.nodeCount.count == 0){

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

        // Tofino1 (P4TG)
        // 23/0 (448) TF1 -> 1/1(137) Luigi
        else if (ig_intr_md.ingress_port == 448) {
            ig_tm_md.ucast_egress_port = 137;
        }
        // Tofino1 (P4TG)
        // 24/0 (40) TF1 -> 1/1(137) Luigi
        else if (ig_intr_md.ingress_port == 440) {
            ig_tm_md.ucast_egress_port = 137;
        }

        // Original Packets whose ingress is not 128 (for cloned packets) and is out of list [136,137,440,448] 
        else{
            //Recirculating INT Packets 
            ig_tm_md.ucast_egress_port = 136;
        }

        /***
        ##############################################################################################################
        Packet ECT(1) Marking 
        ##############################################################################################################
        ***/  

        if(hdr.tcp.isValid()){
            extract_flowID_ipv4_tcp();
        }
        else if(hdr.udp.isValid()){
            extract_flowID_ipv4_udp();
        }
        
        //*** marking_decision --> 0 or 1 or 2 or 3 ***// 
        // 0--> Non-Classified, 1--> AR | 2--> CG | 3--> Other (Non-[Ar or CG])
        bit<8> marking_decision = read_shared_register.execute(ig_md.metadata_flowID);

        //*** Not Mark INT Packet ***/
        if(hdr.nodeCount.isValid()){
            marking_decision = 0;
        }

        //*** marking_decision = 1 (AR) ***//
        //*** Action --> Mark ECT(1) & Set DSCP(46)
        if(marking_decision == 1){ // AR
            hdr.ipv4.ecn = 1;
            hdr.ipv4.dscp = 46;    
        }
        //*** marking_decision = 2 (CG) ***//
        //*** Action --> Mark ECT(1) & Set DSCP(34)
        else if(marking_decision == 2){ // CG
            hdr.ipv4.ecn = 1;
            hdr.ipv4.dscp = 34;
        }

        //*** marking_decision = 1 (Other) ***//
        //*** Action --> Mark ECT(0) & Set DSCP(46)
        else if(marking_decision == 3){ // Other or Non-(AR or CG)
            //hdr.ipv4.ecn = 0;
            hdr.ipv4.dscp = 50;
        }

        /***
        ##############################################################################################################
        L4S vs Classic Queue Allocation
        Queue_ID ==1 ==> L4S
        Queue_ID ==0 ==> Classic 
        ##############################################################################################################
        ***/  
        if(hdr.ipv4.ecn == 1){
        ig_tm_md.qid = 1; // Set L4S Queue 
                 }
        // Note: Default Queue ID is 0 for other packets that we let it go on!

        //** Insert ingress timestamp into bridge header to be used in the egress**//
        hdr.bridge.setValid();
        hdr.bridge.bridge_ingress_port = (bit<16>)ig_intr_md.ingress_port;
        hdr.bridge.ingress_global_tstamp = ig_intr_prsr_md.global_tstamp;
        hdr.bridge.bridge_qid = (bit<16>)ig_tm_md.qid; 
    }
    
    }    
    
}
// ---------------------------------------------------------------------------
// *** (2-3) Ingress Deparser *** ///
// ---------------------------------------------------------------------------
control IngressDeparser( //SwitchIngressDeparser
        packet_out pkt,
        inout headers_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {

    //Mirror() mirror;
    Checksum() ipv4_checksum;

    apply {

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


/***
-------------------------------------------------------------------------------------------------------------------------------------------
Traffic Manager (TM) - non-programmable block located between Ingress and Egress (e.g., Queue Management)

Note: TM will receive the packet cloned at the Egress, and will recirculate this packet to the Ingress.
// ----------------------------------------------------------------------------------------------------------------------------------------
***/


// ****************************** (3) Egress (Parser/Ingress (MA) Processing/Deparser)  ******************************************//

// ---------------------------------------------------------------------------
// *** (3-1) Egress Parser *** ///
// ---------------------------------------------------------------------------
parser EgressParser(
        packet_in pkt,
        out headers_t hdr,
        out metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    
    //*** Starts State Machine ***///
    state start {
        pkt.extract(eg_intr_md);
        transition select(eg_intr_md.egress_port){
            (MIRROR_PORT): parse_mirror;                    // Mirror Packet Port = 128 
            (_): parse_bridge;
        }
    }

    /** E2E MIRRORED PKTS (Parsing Cloned Packets)**/
    state parse_mirror{
        pkt.extract(hdr.mirror);
        transition accept;
    }

    //** Parsing Bridge for original Packets **//
    state parse_bridge{
        pkt.extract(hdr.bridge);
        transition parse_ethernet;
    }

    // ** Ethernet Header Parsing **//
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select (hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : accept;
        }
    }

    // ** IPv4 Header Parsing **//
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_UDP: parse_udp;
            PROTO_INT: parse_INT;
            //PROTO_ICMP: parse_icmp;
        default: accept;
      }
    }

    // ** UDP Header Parsing **//
    state parse_udp {
        pkt.extract(hdr.udp);
        transition parse_rtp;
    }

    // ** RTP Header Parsing **//
    state parse_rtp {
        pkt.extract(hdr.rtp);
        transition accept;
    }

    // ** INT Header Parsing **//
    state parse_INT{
        pkt.extract(hdr.nodeCount);
        eg_md.parser_metadata.remaining = hdr.nodeCount.count;
        transition select(eg_md.parser_metadata.remaining) {
            2 : parse_two_INT;
            1: parse_one_INT;
            0: accept;
        }
    }
    // ** INT Header Parsing **//
    state parse_two_INT{
        pkt.extract(hdr.INT.next);
        pkt.extract(hdr.INT.next);
        transition accept;
    }
    // ** INT Header Parsing **//
    state parse_one_INT{
        pkt.extract(hdr.INT.next);
        transition accept;
    }
}




// ---------------------------------------------------------------------------
// *** (3-2) Egress Match/Action (MA) Processing *** ///
// ---------------------------------------------------------------------------
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

        // *** Variables for Classification Mirroring Threshold *** //
        // ****** Number of RTP Frames ******* //
        // ****** Number of UDP Packets ******* //
        bit<8> frame_counter_value;
        bit<8> packet_counter_value;


        //IPG
        bit<32> IPG;

        // *** Hash Function Using Extern Function *** //
        Hash<bit<8>>(HashAlgorithm_t.CRC8) hash_function_tcp; //2 to the power of 8 possible flows
        Hash<bit<8>>(HashAlgorithm_t.CRC8) hash_function_udp; //2 to the power of 8 possible flows

        
        Counter<bit<32>, bit<32>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES, CounterType_t.PACKETS) mark_ecn_pkt;
        Counter<bit<32>, bit<32>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES, CounterType_t.PACKETS) thresholdPkts;
        //Counter<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES, CounterType_t.PACKETS) totalPkts;
        //Register<bit<16>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) qdelay_classic;

        // *** Dropping/ECN Marking Policy ***
        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) qdelay_l4s;
        Register<bit<32>, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) switchLatency;
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
    



        // Declaration of LPF with 1024 index
        // Lpf<bit<32>, bit<16>>(size=1024) queue_delay_lpf;

        
        
        
        Register<enq_qdepth_v, bit<16>>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) enq_qdepth_reg;
        Register<bit<16>, bit<16>>(1) index_reg;
        Register<bit<32>, _>(N_PORTS) timestamp_ifg_reg;
        Register<bit<16>, _>(N_PORTS) frame_size_reg;
        Register<bit<32>, _>(N_PORTS) timestamp_ipg_reg;
               
        
        Register<bit<8>, _>(NUMBER_OF_QUEUES + NUMBER_OF_QUEUES) frame_counter_threshhold; // for mirroring control
        Register<bit<8>, _>(N_PORTS) frame_counter; // for mirroring control

        Register<bit<8>, _>(NUMBER_OF_QUEUES+NUMBER_OF_QUEUES) packet_counter_threshhold; // for mirroring control
        Register<bit<8>, _>(N_PORTS) packet_counter_flowbased; // for mirroring control

        //Packet counter
          RegisterAction<bit<8>, bit<8>, bit<8>>(frame_counter) increase_packet_counter_flowbased = { // for Frame counting threshold
            void apply(inout bit<8> value, out bit<8> result) {
                result = value + 1;
                value =  value + 1; 
            }
        };     

        RegisterAction<bit<8>, bit<8>, bit<8>>(frame_counter_threshhold) read_packet_counter_threshhold = { // for Frame counting threshold
            void apply(inout bit<8> value, out bit<8> result) {
                result = value;
                value =  value; //eg_md.metadata_ipg_temp;
            }
        };           
        //frame counter
         RegisterAction<bit<8>, bit<8>, bit<8>>(frame_counter) increase_frame_counter = { // for Frame counting threshold
            void apply(inout bit<8> value, out bit<8> result) {
                result = value + 1;
                value =  value + 1; 
            }
        };     

        RegisterAction<bit<8>, bit<8>, bit<8>>(frame_counter_threshhold) read_frame_counter_threshhold = { // for Frame counting threshold
            void apply(inout bit<8> value, out bit<8> result) {
                result = value;
                value =  value; //eg_md.metadata_ipg_temp;
            }
        };        
       //frame counter
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


    // Alireza

        Register<bit<32>, _> (0x100) host_ifg_reg; //Host IFG register

        // Storing previous timestamp
        Register<bit<32>, _> (0x100) rtp_timestamp_reg; //Host Timestamp register
        RegisterAction<bit<32>, bit<8>, bit<32>>(rtp_timestamp_reg) update_rtp_timestamp_reg = {
            void apply(inout bit<32> value, out bit<32> result) {
                result = value;
                value = hdr.rtp.timestamp;

            }
    };
        // Storing the ifg host
        RegisterAction<bit<32>, bit<8>, bit<32>>(host_ifg_reg) update_host_ifg_reg = {
        void apply(inout bit<32> value) {
                value =  eg_md.metadata_host_ifg;
            }
    };
    // Alireza





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
        
        RegisterAction<bit<32>, bit<16>, bit<32>>(qdelay_l4s) qdelay_l4s_action = {
            void apply(inout bit<32> value, out bit<32> result) {
                
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
        
        Register<bit<8>, _> (1) flow_id_reg;

        RegisterAction<bit<8>, bit<8>, bit<8>>(flow_id_reg) write_flow_id_reg = {
            void apply(inout bit<8> value) {
                value = eg_md.metadata_flowID ;

            }
        };
        
        RegisterAction<bit<32>, bit<16>, bit<32>>(target_violation) compute_target_violations = {
            void apply(inout bit<32> value, out bit<32> violation){

                value = EWMA;
                


    /*
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Experienced Congestion (EC) Marking/Droping the Packets (L4S and Classic) Queues
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    */
                // No drop 
                if (value <= CoDel_Min){
                    violation = 0;
                }

                //Dropping/Marking with a Probability 
                if ((value > CoDel_Min) && (value < CoDel_Max)){
                    violation = 1;
                }

                ////Dropping/Marking (Queue sounds full!)
                if (value > CoDel_Max){
                    violation = 2;
                }

            }
        };    

    action extract_flowID_ipv4_udp(){
        eg_md.metadata_flowID = hash_function_udp.get({hdr.ipv4.dst_addr, hdr.udp.dstPort, hdr.ipv4.protocol});
    }

    action extract_flowID_ipv4_tcp(){
        eg_md.metadata_flowID = hash_function_tcp.get({hdr.ipv4.dst_addr, hdr.tcp.dstPort, hdr.ipv4.protocol});
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
            //hdr.mirror.testing = 99;
            hdr.mirror.flowid = eg_md.metadata_flowID;
            hdr.mirror.packet_size = hdr.ipv4.total_len;
            hdr.mirror.frame_size = eg_md.metadata_frame_size;
            hdr.mirror.ifg = eg_md.metadata_ifg;
            hdr.mirror.IPG = eg_md.metadata_ipg;
            hdr.mirror.rtp_marker = hdr.rtp.marker;
            hdr.mirror.has_rtp = eg_md.metadata_has_rtp;
            hdr.mirror.queue_delay = eg_md.metadata_queue_delay_new;
            hdr.mirror.host_ifg = eg_md.metadata_host_ifg; // Alireza added
        }


        // Table for Thresholds / counters.frame_count < 3 ||  counters.packet_count < 20
        table table_threshold{
        key = {
            packet_counter_value: range;
            frame_counter_value: range;
        }
        actions = {
            decisionMirror;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }



        //INT
        RegisterAction<bit<32>, bit<16>, bit<32>>(switchLatency) switchLatencyAction = {
            void apply(inout bit<32> value, out bit<32> result) {
                value = queue_delay;
            }

        };    

        action _drop(){
            eg_intr_dprs_md.drop_ctl = 1;
        }


        action add_swtrace(){
           
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
        }
       
    apply {
           
        if(hdr.bridge.bridge_qid == 0){
            if(eg_intr_md.egress_port == 136){ 
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
                eg_md.metadata_index = 0x02;
            }
            else if(eg_intr_md.egress_port == 137){
                eg_md.metadata_index = 0x03;
            }
            else{//loopbackport com int
                eg_md.metadata_index = 0x03;
            }
        }
        // else if(hdr.bridge.bridge_qid == 2){
        //     if(eg_intr_md.egress_port == 136){ 
        //         eg_md.metadata_index = 0x04;
        //     }
        //     else if(eg_intr_md.egress_port == 137){
        //         eg_md.metadata_index = 0x05;
        //     }
        //     else{//loopbackport com int
        //         eg_md.metadata_index = 0x05;
        //     }
        // }


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
                        hdr.ipv4.ecn = 3;    // Mark EC !!!
                    
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
        
        // Flow ID starts
        if (hdr.ipv4.isValid() && hdr.ipv4.protocol == PROTO_UDP) {
            extract_flowID_ipv4_udp();
        }
        else if(hdr.ipv4.isValid() && hdr.ipv4.protocol == PROTO_TCP){
            extract_flowID_ipv4_tcp();
        }


        // Alireza
        Counters counters;
        Thresholds thresholds;

        // increase the packets for each flow
        //bit<8> my_packet_counter = increase_packet_counter_flowbased.execute(eg_md.metadata_flowID);
        //Increase  

        eg_md.metadata_ipg_temp = (bit<32>) eg_intr_md_from_prsr.global_tstamp;
        bit<32> previous_tstamp_ipg = update_timestamp_ipg_reg.execute(eg_md.metadata_flowID);

        eg_md.metadata_ipg = (bit<32>)eg_intr_md_from_prsr.global_tstamp - previous_tstamp_ipg;
        
        // frm_thr = 3
        //IFG AND FRAME SIZE
        //SAVE TIMESTAMP OF THE EGRESS WHEN MARKER IS 1 AND JUST READSUBSTITUTE AGAIN WHEN A NEW MARKER
        

        bit<8> counter_frames = 0;
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
                    //Alireza start
                    //RTP TIMESTAMP STORE 
                    eg_md.metadata_rtp_timestamp = update_rtp_timestamp_reg.execute(eg_md.metadata_flowID);
                    eg_md.metadata_host_ifg = hdr.rtp.timestamp - eg_md.metadata_rtp_timestamp;

                    update_host_ifg_reg.execute(eg_md.metadata_flowID);

                    // Counting the Frame Number
                    increase_frame_counter.execute((bit<8>)eg_md.metadata_index, frame_counter_value);// counters.frame_count);
                    //frame_counter_value = counters.frame_count;

                    //counter_frames = increase_frame_counter.execute(eg_md.metadata_flowID);
                    //Alireza end
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

        increase_packet_counter_flowbased.execute((bit<8>)eg_md.metadata_index, packet_counter_value); //counters.packet_count);
        //packet_counter_value = counters.packet_count;
        // Read Thresholds from register
        read_frame_counter_threshhold.execute((bit<8>)eg_md.metadata_index, thresholds.frame_thresh);
        read_packet_counter_threshhold.execute((bit<8>)eg_md.metadata_index, thresholds.packet_thresh);
        //bit<8> frame_threshhold_value = read_frame_counter_threshhold.execute((bit<8>)eg_md.metadata_index); // set thrshhold value
        //bit<8> packet_threshhold_value = read_packet_counter_threshhold.execute((bit<8>)eg_md.metadata_index); // set thrshhold value
        
        if(eg_intr_md.egress_port != MIRROR_PORT && eg_intr_md.egress_port != recirc_port){
           
            //if (counters.frame_count < 3 ||  counters.packet_count < 20){ //  thresholds.packet_thresh){ thresholds.frame_thresh
                decisionMirror(); //mirrorring is trigered!
            //}
            
            // *** table for thresholds
            //table_threshold.apply();
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
                                      hdr.mirror.flowid,
                                      hdr.mirror.ifg,
                                      hdr.mirror.packet_size,
                                      hdr.mirror.frame_size,
                                      hdr.mirror.rtp_marker,
                                      hdr.mirror.queue_delay,
                                      hdr.mirror.has_rtp,
                                      hdr.mirror.host_ifg // Alireza added
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


Pipeline(IngressParser(), 
        Ingress(),
        IngressDeparser(),
        EgressParser(),
        Egress(),
        EgressDeparser()) pipe;

Switch(pipe) main;
