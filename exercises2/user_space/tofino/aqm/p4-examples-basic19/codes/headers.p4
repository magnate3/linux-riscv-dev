
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
    bit<4>   class;
    bit<32>   latency;
    bit<4>   priority;

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
    bit<4>        class_id;
    bit<32>        latency;
    bit<32>        q_delay;
    bit<4>        priority;

    bit<32> current_queue_bound;
    bit<32> current_queue_delay;
    bit<32> rank;
    bit<48> procTime;

    bit<14> action_select1;
    bit<14> action_select2;
    bit<14> action_select3;
    bit<14> action_select4;
    bit<16> flowID;
    bit<16> packetSize;
    bit<16> packet_counter;
    bit<16> firstPacketSize;
    bit<16> secondPacketSize;
    bit<16> thirdPacketSize;
    bit<16> fourthPacketSize;

    bit<16> classID;

    queueing_metadata_t       queueing_metadata;
}
