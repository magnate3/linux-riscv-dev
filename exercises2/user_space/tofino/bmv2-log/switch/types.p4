//v1model
typedef bit<9> portId_t;
typedef bit<32> clone_session_t;
typedef bit<48> timestamp_t;

//v1model meters
typedef bit<2> meter_color_t;
const meter_color_t METER_GREEN = 0;
const meter_color_t METER_YELLOW = 1;
const meter_color_t METER_RED = 2;
const meter_color_t METER_INVALID = 3;

//We reserve resources for each VQ so we need to limit the number of VQs. A VQ ID consists of a port and a flow id,
//  therefore by storing the port IDs in a smaller type we don't need to reserve as much space for VQs.
#define SMALL_PORT_T_WIDTH 4
typedef bit<SMALL_PORT_T_WIDTH> small_port_t;
//Limits how many flows we can differentiate, also affects how large the VQ ID is
#define FLOW_ID_T_WIDTH 12
typedef bit<FLOW_ID_T_WIDTH> flow_id_t;
#define VQ_ID_T_WIDTH (SMALL_PORT_T_WIDTH + FLOW_ID_T_WIDTH)
typedef bit<VQ_ID_T_WIDTH> vq_id_t;

struct metadata {
    flow_id_t flow_id;
    vq_id_t vq_id;
}

//Ethernet
typedef bit<16> etherType_t;
typedef bit<48> macAddr_t;

//IPv4, UDP, TCP
typedef bit<32> ip4Addr_t;
typedef bit<8> protocol_t;
typedef bit<16> protocol_port_t;

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    etherType_t etherType;
}

header ipv4_t {
    bit<4> version;
    bit<4> ihl;
    bit<6> dscp;
    bit<2> ecn;
    bit<16> totalLen;
    bit<16> identification;
    bit<3> flags;
    bit<13> fragOffset;
    bit<8> ttl;
    protocol_t protocol;
    bit<16> hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_t {
    protocol_port_t srcPort;
    protocol_port_t dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4> dataOffset;
    bit<4> res;
    bit<1> cwr;
    bit<1> ece;
    bit<1> urg;
    bit<1> ack;
    bit<1> psh;
    bit<1> rst;
    bit<1> syn;
    bit<1> fin;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    protocol_port_t srcPort;
    protocol_port_t dstPort;
    bit<16> udplen;
    bit<16> udpchk;
}

const etherType_t ETHER_TYPE_IPV4 = 0x0800; //2048
const etherType_t ETHER_TYPE_IPV6 = 0x86DD; //34525
const protocol_t IPV4_PROTOCOL_TCP = 0x06; //6
const protocol_t IPV4_PROTOCOL_UDP = 0x11; //17

struct headers {
    ethernet_t ethernet;
    ipv4_t ipv4;
    tcp_t tcp;
    udp_t udp;
}

//Extracts the 4 8-bit IPv4 components, separated by commas. Useful for logging.
#define SLICE_IPV4_ADDRESS(ADDRESS) (ADDRESS)[31:24], (ADDRESS)[23:16], (ADDRESS)[15:8], (ADDRESS)[7:0]
