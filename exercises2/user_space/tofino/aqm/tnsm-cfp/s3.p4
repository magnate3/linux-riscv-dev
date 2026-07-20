/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4 = 0x800;

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

const bit<6> DSCP_INT = 0x17;
const bit<6> DSCP_MASK = 0x3F;
const bit<8>  IP_PROTO_UDP = 0x11;
const bit<8>  IP_PROTO_TCP = 0x6;
const bit<8> INT_HEADER_LEN_WORD = 3;

register<bit<32>>(1) procTime_reg;

typedef bit<32> switch_id_t;
typedef bit<8>  pkt_type_t;

header ethernet_t {
    macAddr_t dst_addr;
    macAddr_t src_addr;
    bit<16>   etherType;
}
const bit<8> ETH_HEADER_LEN = 14;

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
const bit<8> UDP_HEADER_LEN = 8;

header tcp_t {
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4>  data_offset;
    bit<3>  res;
    bit<3>  ecn;
    bit<6>  ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}
const bit<8> TCP_HEADER_LEN = 20;

// INT shim header for TCP/UDP
header intl4_shim_t {
    bit<4> int_type;                // Type of INT Header
    bit<2> npt;                     // Next protocol type
    bit<2> rsvd;                    // Reserved
    bit<8> len;                     // Length of INT Metadata header and INT stack in 4-byte words, not including the shim header (1 word)
    bit<6> udp_ip_dscp;            // depends on npt field. either original dscp, ip protocol or udp dest port
    bit<10> udp_ip;                // depends on npt field. either original dscp, ip protocol or udp dest port
}

const bit<16> INT_SHIM_HEADER_SIZE = 4;

// INT header
header int_header_t {
    bit<4>   ver;                    // Version
    bit<1>   d;                      // Discard
    bit<27>  rsvd;                   // 12 bits reserved, set to 0
    bit<8>   class;
    bit<32>   latency;
    bit<16>   c_p_counter;

    // Optional domain specific 'source only' metadata
}
const bit<16> INT_HEADER_SIZE = 9;
const bit<16> INT_TOTAL_HEADER_SIZE = 13; // 8 + 4


struct headers {

    // Original Packet Headers
    ethernet_t                  ethernet;
    ipv4_t			            ipv4;
    udp_t			            udp;
    tcp_t			            tcp;

    // INT Headers
    int_header_t                int_header;
    intl4_shim_t                intl4_shim;

}

struct queueing_metadata_t {
    bit<32>   enq_timestamp;
    bit<19>   enq_qdepth;
    bit<32>   deq_timedelta;
    bit<19>   deq_qdepth;

}

struct local_metadata_t {
    bit<8>        class_id;
    bit<32>        latency;
    bit<32>        q_delay;

    bit<32> current_queue_bound;
    bit<32> current_queue_delay;
    bit<32> rank;
    bit<32> procTime;

    bit<14> action_select1;
    bit<14> action_select2;
    bit<14> action_select3;
    bit<14> action_select4;
    bit<16> flowID;
    bit<16> packetSize;
    bit<16> packet_counter;
    bit<8>  c_p_counter;

    bit<16> classID;

    queueing_metadata_t       queueing_metadata;
}



/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/


parser MyParser(packet_in packet,
                out headers hdr,
                inout local_metadata_t metadata,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            IP_PROTO_UDP: parse_udp;
            IP_PROTO_TCP: parse_tcp;
            default: accept;
        }
    }
    state parse_udp {
    packet.extract(hdr.udp);
    transition select(hdr.ipv4.dscp) {
        DSCP_INT &&& DSCP_MASK: parse_shim;
        default:  accept;
    }
}

state parse_tcp {
    packet.extract(hdr.tcp);
    transition select(hdr.ipv4.dscp) {
        DSCP_INT &&& DSCP_MASK: parse_shim;
        default:  accept;
    }
}

state parse_shim {
    packet.extract(hdr.intl4_shim);
    transition parse_int_hdr;
}

state parse_int_hdr {
    packet.extract(hdr.int_header);
    transition accept;
}

}


/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout local_metadata_t metadata) {
    apply {  }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

// Registers
/*Queue with index 0 is the bottom one, with lowest priority*/
register<bit<32>>(8) queue_bound;
register<bit<32>>(8) queue_delay;
register<bit<16>>(100000) packet_counter_reg;
register<bit<16>>(100000) c_p_counter_reg;

/************   Adding INT to the Packet  *************/


control process_int (
    inout headers hdr,
    inout local_metadata_t metadata) {

    action int_a(){

      hdr.intl4_shim.setValid();                              // insert INT shim header

    }

    table int_t {

    actions = {
        int_a;
        NoAction;
    }
    const default_action = NoAction();
}

apply {
    int_t.apply();

}

}


control MyIngress(inout headers hdr,
                  inout local_metadata_t metadata,
                  inout standard_metadata_t standard_metadata) {


    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(macAddr_t dst_addr, egressSpec_t port) {
        standard_metadata.egress_spec = port;
        hdr.ethernet.src_addr = hdr.ethernet.dst_addr;
        hdr.ethernet.dst_addr = dst_addr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }


    table ipv4_lpm {
        key = {
            hdr.ipv4.dst_addr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = drop();
    }


    apply {
        if (hdr.ipv4.isValid()) {

            if(hdr.ipv4.protocol ==6) {

            // INT + Queueing + Admission Control

            // 1. ADD INT header
            process_int.apply(hdr, metadata);

            // 2. Calculate packet counter for class
            packet_counter_reg.read(metadata.packet_counter, (bit<32>)hdr.int_header.class);
            metadata.packet_counter = metadata.packet_counter + 1;
            packet_counter_reg.write((bit<32>)hdr.int_header.class, metadata.packet_counter);

            // 2. Apply Multi-Priority Queueing based on remaining Latency

            //standard_metadata.priority = (bit<3>)hdr.int_header.class;
            //standard_metadata.priority = (bit<3>)0;


            // Multi-Priority Queueing Starts here

            // High priority
            if (hdr.int_header.class == 3){
                metadata.rank = (bit<32>)hdr.int_header.latency;
                queue_bound.read(metadata.current_queue_bound, 6);
                if ((metadata.current_queue_bound <= metadata.rank)) {
                    standard_metadata.priority = (bit<3>)6;
                    queue_bound.write(6, metadata.rank);
                    } else {
                            standard_metadata.priority = (bit<3>)7;
                            queue_bound.read(metadata.current_queue_bound, 7);

                            /*Blocking reaction*/
                            if(metadata.current_queue_bound > metadata.rank) {
                                bit<32> cost = metadata.current_queue_bound - metadata.rank;
                                queue_bound.read(metadata.current_queue_bound, 6);
                                queue_bound.write(6, (bit<32>)(metadata.current_queue_bound-cost));
                                queue_bound.write(7, metadata.rank);
                            } else {
                                queue_bound.write(7, metadata.rank);
                            }
                    }
            } else {

                    // Medium priority
                    if (hdr.int_header.class == 2){
                        metadata.rank = (bit<32>)hdr.int_header.latency;
                        queue_bound.read(metadata.current_queue_bound, 4);
                        if ((metadata.current_queue_bound <= metadata.rank)) {
                            standard_metadata.priority = (bit<3>)4;
                            queue_bound.write(4, metadata.rank);
                            } else {
                                    standard_metadata.priority = (bit<3>)5;
                                    queue_bound.read(metadata.current_queue_bound, 5);

                                    /*Blocking reaction*/
                                    if(metadata.current_queue_bound > metadata.rank) {
                                        bit<32> cost = metadata.current_queue_bound - metadata.rank;
                                        queue_bound.read(metadata.current_queue_bound, 4);
                                        queue_bound.write(4, (bit<32>)(metadata.current_queue_bound-cost));
                                        queue_bound.write(5, metadata.rank);
                                    } else {
                                        queue_bound.write(5, metadata.rank);
                                    }
                            }
                    } else {

                            // Low priority
                            if (hdr.int_header.class == 1){
                                metadata.rank = (bit<32>)hdr.int_header.latency;
                                queue_bound.read(metadata.current_queue_bound, 2);
                                if ((metadata.current_queue_bound <= metadata.rank)) {
                                    standard_metadata.priority = (bit<3>)2;
                                    queue_bound.write(2, metadata.rank);
                                    } else {
                                            standard_metadata.priority = (bit<3>)3;
                                            queue_bound.read(metadata.current_queue_bound, 3);

                                            /*Blocking reaction*/
                                            if(metadata.current_queue_bound > metadata.rank) {
                                                bit<32> cost = metadata.current_queue_bound - metadata.rank;
                                                queue_bound.read(metadata.current_queue_bound, 2);
                                                queue_bound.write(2, (bit<32>)(metadata.current_queue_bound-cost));
                                                queue_bound.write(3, metadata.rank);
                                            } else {
                                                queue_bound.write(3, metadata.rank);
                                            }
                                    }
                            }

                            // Best effort
                            if (hdr.int_header.class == 0){
                                metadata.rank = (bit<32>)hdr.int_header.latency;
                                queue_bound.read(metadata.current_queue_bound, 0);
                                if ((metadata.current_queue_bound <= metadata.rank)) {
                                    standard_metadata.priority = (bit<3>)0;
                                    queue_bound.write(0, metadata.rank);
                                    } else {
                                            standard_metadata.priority = (bit<3>)1;
                                            queue_bound.read(metadata.current_queue_bound, 1);

                                            /*Blocking reaction*/
                                            if(metadata.current_queue_bound > metadata.rank) {
                                                bit<32> cost = metadata.current_queue_bound - metadata.rank;
                                                queue_bound.read(metadata.current_queue_bound, 0);
                                                queue_bound.write(0, (bit<32>)(metadata.current_queue_bound-cost));
                                                queue_bound.write(1, metadata.rank);
                                            } else {
                                                queue_bound.write(1, metadata.rank);
                                            }
                                    }
                            }
                    }
            }

            // Multi-Priority Queueing Ends here


            queue_delay.read(metadata.current_queue_delay, (bit<32>)standard_metadata.priority); // read the delay of particular queue
            procTime_reg.read(metadata.procTime, 0);

            //log_msg(" ClassID : {} Priority : {} PacketCounter : {} Latency : {} procTime : {} ", {hdr.int_header.class, standard_metadata.priority, metadata.packet_counter, hdr.int_header.latency, metadata.current_queue_delay});
            log_msg(" INFO ClassID : {} PacketCounter : {} Latency : {} qDelay : {} ", {hdr.int_header.class, hdr.int_header.c_p_counter, hdr.int_header.latency, metadata.current_queue_delay});

            // 3. Admission Control Policyy
            if (hdr.int_header.latency > metadata.current_queue_delay){
                ipv4_lpm.apply();
            }
            }
        }
    }
}


/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout local_metadata_t metadata,
                 inout standard_metadata_t standard_metadata) {
    apply {

          // Store queueing data into metadata
          //metadata.queueing_metadata.enq_timestamp = standard_metadata.enq_timestamp;
          //metadata.queueing_metadata.enq_qdepth = standard_metadata.enq_qdepth;
          //metadata.queueing_metadata.deq_timedelta = standard_metadata.deq_timedelta;
          //metadata.queueing_metadata.deq_qdepth = standard_metadata.deq_qdepth;
          //metadata.q_delay = (bit<32>)standard_metadata.deq_timedelta;


          bit<3> qid = standard_metadata.priority;
          metadata.procTime = (bit<32>)standard_metadata.egress_global_timestamp - (bit<32>)standard_metadata.ingress_global_timestamp;
          procTime_reg.write((bit<32>)qid, metadata.procTime);
          queue_delay.write((bit<32>)qid, standard_metadata.deq_timedelta);

          // Update Latency by substracting queue delay
          //if (hdr.int_header.latency > metadata.current_queue_delay){
          hdr.int_header.latency = hdr.int_header.latency - (bit<32>)metadata.current_queue_delay;   // Process time included queueing delay
          //}
     }
}


/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers  hdr, inout local_metadata_t metadata) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            {                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.dscp,
                hdr.ipv4.ecn,
                hdr.ipv4.len,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.frag_offset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.src_addr,
                hdr.ipv4.dst_addr },
            hdr.ipv4.hdr_checksum,
            HashAlgorithm.csum16);
    }
}


/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
    packet.emit(hdr.ethernet);
    packet.emit(hdr.ipv4);
    packet.emit(hdr.udp);
    packet.emit(hdr.tcp);

    packet.emit(hdr.intl4_shim);
    packet.emit(hdr.int_header);
    }
}



/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;
