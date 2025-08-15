
const bit<16> TYPE_IPV4 = 0x800;
const bit<8> TYPE_UDP = 0x10001;
const bit<8> TYPE_TCP = 0x00110;

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;
typedef bit<8> ip_protocol_t;

header ethernet_h {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_h {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    tos;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4> data_offset;
    bit<4> res;
    bit<8> flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}

header udp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> hdr_length;
    bit<16> checksum;
    bit<64> result;
}

header worker_h{
    bit<32> ucast_egress_port;
    bit<32> qid;
    bit<32> queue_length;
    bit<32> round_number;
}

header WFQheader_h{
    bit<32> weight;      //可以不用于计算，直接作为一个指示标志使用
  //  bit<64> VFT;        //virtual finish time
}

struct header_t {
    ethernet_h ethernet;
    ipv4_h ipv4;
    tcp_h tcp;
    udp_h udp;
    worker_h worker_t;
    WFQheader_h wfq_t;
    // Add more headers here.
}

