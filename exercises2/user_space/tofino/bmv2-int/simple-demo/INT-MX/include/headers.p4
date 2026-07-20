/********************************************************************
*************** C O N S T A N T S   A N D   T Y P E S ***************
********************************************************************/
#define MAX_PORTS 511
#define CLONE_PKT 1

const bit<6> HW_ID = 1;

const bit<16> TYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_UDP = 0x11;
const bit<8>  IP_PROTO_TCP = 0x6;

const bit<6> DSCP_INT = 0x17;
const bit<6> DSCP_MASK = 0x3F;

typedef bit<48> mac_t;
typedef bit<32> ip_address_t;
typedef bit<16> l4_port_t;
typedef bit<9>  port_t;
typedef bit<32> node_id_t;

const bit<32> REPORT_MIRROR_SESSION_ID = 500;

/************************************************************
*********************** H E A D E R S ***********************
************************************************************/
header ethernet_t {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
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


// INT shim header for INT over TCP/UDP
header intl4_shim_t {
    bit<4> int_type;                // Type of INT Header
    bit<2> npt;                     // Next protocol type
    bit<2> rsvd1;                   // Reserved
    bit<8> len;                     // Length of INT Metadata header and INT stack in 4-byte words, not including the shim header (1 word)
    bit<8> rsvd2;                   // Reserved
    bit<6> original_dscp;           // Original DSCP value
    bit<2> rsvd3;                   // Reserved
}

const bit<16> INT_SHIM_HEADER_SIZE = 4;


// INT-MX header
header int_header_t {
    bit<4>   ver;                    // Version
    bit<1>   d;                      // Discard
    bit<27>  rsvd;                   // Reserved              
    bit<4>   instruction_mask_0003;  // split the bits for lookup
    bit<4>   instruction_mask_0407;
    bit<4>   instruction_mask_0811;
    bit<4>   instruction_mask_1215;
    bit<16>  domain_specific_id;     // Unique INT Domain ID
    bit<16>  ds_instruction;         // Instruction bitmap specific to the INT Domain identified by the Domain specific ID
    bit<16>  ds_flags;               // Domain specific flags
    // Optional Domain Specific 'Source-Inserted' Metadata
}

const bit<16> INT_HEADER_SIZE = 12;
const bit<16> INT_TOTAL_HEADER_SIZE = INT_SHIM_HEADER_SIZE + INT_HEADER_SIZE;


// INT meta-value headers - different header for each value type to monitor
header int_node_id_t {
    bit<32> node_id;
}
header int_level1_port_ids_t {
    bit<16> ingress_port_id;
    bit<16> egress_port_id;
}
header int_hop_latency_t {
    bit<32> hop_latency;
}
header int_q_occupancy_t {
    bit<8> q_id;
    bit<24> q_occupancy;
}
header int_ingress_tstamp_t {
    bit<64> ingress_tstamp;
}
header int_egress_tstamp_t {
    bit<64> egress_tstamp;
}
header int_level2_port_ids_t {
    bit<32> ingress_port_id;
    bit<32> egress_port_id;
}

// These two are not implemented yet
header int_egress_port_tx_util_t {
    bit<32> egress_port_tx_util;            // This is used but the queue latency is gathered instead
}

header int_buffer_t {
    bit<8> buffer_id;
    bit<24> buffer_occupancy;
}

// Report Telemetry Headers - Group Header
header report_group_header_t {
    bit<4>  ver;
    bit<6>  hw_id;
    bit<22> seq_no;
    bit<32> node_id;
}
const bit<8> REPORT_GROUP_HEADER_LEN = 8;

// Report Telemetry Headers - Individual Header
header report_individual_header_t {
    bit<4>  rep_type;
    bit<4>  in_type;
    bit<8>  len;
    bit<8>  rep_md_len;
    bit<1>  d;
    bit<1>  q;
    bit<1>  f;
    bit<1>  i;
    bit<4>  rsvd;
    // Individual report contents for Reptype 1 = INT
    bit<16> rep_md_bits;
    bit<16> domain_specific_id;
    bit<16> domain_specific_md_bits;
    bit<16> domain_specific_md_status;
}
const bit<8> REPORT_INDIVIDUAL_HEADER_LEN = 12;

// Preserving metadata header for the cloned packet
struct preserving_metadata_t {
    @field_list(1)
    bit<9> ingress_port;
    @field_list(1)
    bit<9> egress_port;
    @field_list(1)
    bit<32> deq_timedelta;
    @field_list(1)
    bit<19> deq_qdepth;
    @field_list(1)
    bit<48> ingress_global_timestamp;
    @field_list(1)
    bit<48> egress_global_timestamp;
}

struct headers {

    // Original Packet Headers
    ethernet_t                  ethernet;
    ipv4_t			            ipv4;
    udp_t			            udp;
    tcp_t			            tcp;

    // INT Report Encapsulation
    ethernet_t                  report_ethernet;
    ipv4_t                      report_ipv4;
    udp_t                       report_udp;

    // INT Headers
    int_header_t                int_header;
    intl4_shim_t                intl4_shim;

    //INT Metadata
    int_node_id_t               int_node_id;
    int_level1_port_ids_t       int_level1_port_ids;
    int_hop_latency_t           int_hop_latency;
    int_q_occupancy_t           int_q_occupancy;
    int_ingress_tstamp_t        int_ingress_tstamp;
    int_egress_tstamp_t         int_egress_tstamp;
    int_level2_port_ids_t       int_level2_port_ids;
    int_egress_port_tx_util_t   int_egress_tx_util;

    // INT Report Headers
    report_group_header_t       report_group_header;
    report_individual_header_t  report_individual_header;
}

struct int_metadata_t {
    node_id_t node_id;
    bit<16> new_bytes;
    bit<8>  new_words;
    bool  source;
    bool  sink;
}

struct local_metadata_t {
    bit<16>       l4_src_port;
    bit<16>       l4_dst_port;
    int_metadata_t int_meta;
    preserving_metadata_t perserv_meta;
}