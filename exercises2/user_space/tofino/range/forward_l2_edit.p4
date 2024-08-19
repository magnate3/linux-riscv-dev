/* -*- P4_16 -*- */

#include <core.p4>
#include <tna.p4>

/*************************************************************************
 ************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/
const bit<16> ETHERTYPE_TPID = 0x8100;
const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8>  TYPE_TCP = 6;

/* Table Sizes */
const int IPV4_HOST_SIZE = 65536;
const int IPV4_LPM_SIZE  = 12288;

#define BLOOM_FILTER_ENTRIES 4096

//typedef bit<9> egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

struct user_metadata_t {
    bit<1> bf_tmp;
}

/*************************************************************************
 ***********************  H E A D E R S  *********************************
 *************************************************************************/

header ethernet_h {
    bit<48>   dstAddr;
    bit<48>   srcAddr;
    bit<16>   ether_type;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<4>  res;
    bit<1>  cwr;
    bit<1>  ece;
    bit<1>  urg;
    bit<1>  ack;
    bit<1>  psh;
    bit<1>  rst;
    bit<1>  syn;
    bit<1>  fin;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/
 
    /***********************  H E A D E R S  ************************/

struct my_ingress_headers_t {
    ethernet_h   ethernet;
    ipv4_t       ipv4;
    tcp_t        tcp;

}

    /******  G L O B A L   I N G R E S S   M E T A D A T A  *********/

struct my_ingress_metadata_t {
    user_metadata_t md;
    bit<1> bloom_read_1;
    bit<1> bloom_read_2;
}

    /***********************  P A R S E R  **************************/
parser IngressParser(packet_in        pkt,
    /* User */    
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    /* Intrinsic */
    out ingress_intrinsic_metadata_t  ig_intr_md)
{
    Checksum() ipv4_checksum;
    Checksum() tcp_checksum;

    /* This is a mandatory state, required by Tofino Architecture */
     state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP: parse_tcp;
            default: accept;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }
}

    /***************** M A T C H - A C T I O N  *********************/

control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{

    // define two registers to hold the bloom filters 
    Register<bit<1>,_>(BLOOM_FILTER_ENTRIES) bloom_filter_1; // [format of individual entries], [index size (DONTCARE)], [number of entries], [name]
    Register<bit<1>,_>(BLOOM_FILTER_ENTRIES) bloom_filter_2; 
    bit<1> direction;

    // hash functions used by the bloom filter
    Hash<bit<32>>(HashAlgorithm_t.CRC16) hash_10;
    Hash<bit<32>>(HashAlgorithm_t.CRC16) hash_11;
    Hash<bit<32>>(HashAlgorithm_t.CRC32) hash_20;
    Hash<bit<32>>(HashAlgorithm_t.CRC32) hash_21;

    @name(".bloom_filter1_get") RegisterAction<bit<1>,bit<32>,bit<1>>(bloom_filter_1) bloom_filter1_get = {
        void apply(inout bit<1> value, out bit<1> ret) {
            ret = value;
        }
    };

    @name(".bloom_filter2_get") RegisterAction<bit<1>,bit<32>,bit<1>>(bloom_filter_2) bloom_filter2_get = {
        void apply(inout bit<1> value, out bit<1> ret) {
            ret = value;
        }
    };

    @name(".bloom_filter1_set") RegisterAction<bit<1>,bit<32>,bit<1>>(bloom_filter_1) bloom_filter1_set = {
        void apply(inout bit<1> value, out bit<1> ret) {
            value = 1;
            ret = 0;
        }
    };

    @name(".bloom_filter2_set") RegisterAction<bit<1>,bit<32>,bit<1>>(bloom_filter_2) bloom_filter2_set = {
       void apply(inout bit<1> value, out bit<1> ret) {
            value = 1;
            ret = 0;
        }
    };

    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }

    action ipv4_forward(macAddr_t dstMac, PortId_t port) { 
        hdr.ethernet.dstAddr = dstMac;
        ig_tm_md.ucast_egress_port = port;
    }

    action set_direction(bit<1> dir) { 
        direction = dir;
    }

    table ipv4_lpm {
        key     =  { hdr.ipv4.dstAddr : lpm; }
        actions =  { ipv4_forward; drop; NoAction; }
        const default_action = drop;
        size = 1024;
    }

    /* Table to determine whether the packet is incoming or outgoing, based on ports used */
    table check_ports {
        key = {
            ig_intr_md.ingress_port : range;
            ig_tm_md.ucast_egress_port : range;
        }
        actions = {
            set_direction;
            NoAction;
        }
        const default_action = NoAction();
        size = 1024;
    }

    apply {
        if (hdr.ipv4.isValid()) {
            ipv4_lpm.apply();

            if (hdr.ipv4.ttl > 1) {
                hdr.ipv4.ttl = hdr.ipv4.ttl - 1; // decrease TTL
            } else {
                drop();
            }

            if (hdr.tcp.isValid()) {

                if (check_ports.apply().hit) {

                    
                    if (direction == 0) { // Packet comes from internal network

                        if (hdr.tcp.syn == 1) { // if syn update bloom filter and add entry
                            bit<32> index1 = (bit<32>)(hash_10.get({hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.tcp.srcPort, hdr.tcp.dstPort, hdr.ipv4.protocol })[11:0]);
                            bit<32> index2 = (bit<32>)(hash_20.get({hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.tcp.srcPort, hdr.tcp.dstPort, hdr.ipv4.protocol })[11:0]);
                            bloom_filter1_set.execute(index1);
                            bloom_filter2_set.execute(index2);
                        }
                    }

                    else if (direction == 1) { // Packet comes from outside

                        // Read bloom filters to check for 1's
                        bit<32> index1 = (bit<32>)(hash_11.get({hdr.ipv4.dstAddr, hdr.ipv4.srcAddr, hdr.tcp.dstPort, hdr.tcp.srcPort, hdr.ipv4.protocol })[11:0]);
                        bit<32> index2 = (bit<32>)(hash_21.get({hdr.ipv4.dstAddr, hdr.ipv4.srcAddr, hdr.tcp.dstPort, hdr.tcp.srcPort, hdr.ipv4.protocol })[11:0]);
                        meta.bloom_read_1 = bloom_filter1_get.execute(index1);
                        meta.bloom_read_2 = bloom_filter2_get.execute(index2);

                        if (meta.bloom_read_1 != 1 || meta.bloom_read_2 != 1) {
                            drop(); // drop packet if it's not in the bloom filter
                        }
                    }
                } else {
                  drop(); // Drop if no match in check_ports
                }
             }
        }
    }
}

    /*********************  D E P A R S E R  ************************/

control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    Checksum() ipv4_checksum; // Initialize checksum func from tofino.p4
    apply { // Update checksum with new ttl
        hdr.ipv4.hdrChecksum = ipv4_checksum.update(
                 {hdr.ipv4.version,
                 hdr.ipv4.ihl,
                 hdr.ipv4.diffserv,
                 hdr.ipv4.totalLen,
                 hdr.ipv4.identification,
                 hdr.ipv4.flags,
                 hdr.ipv4.fragOffset,
                 hdr.ipv4.ttl,
                 hdr.ipv4.protocol,
                 hdr.ipv4.srcAddr,
                 hdr.ipv4.dstAddr});

        pkt.emit(hdr);
    }
}


/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  H E A D E R S  ************************/
/*
struct my_egress_headers_t {
}
*/

struct my_egress_headers_t {
    ethernet_h   ethernet;
    ipv4_t       ipv4;
    tcp_t        tcp;
}

    /********  G L O B A L   E G R E S S   M E T A D A T A  *********/

struct my_egress_metadata_t {}

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
    apply {
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

        /*pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        */
        pkt.emit(hdr);
    }
}


/************ F I N A L   P A C K A G E ******************************/
Pipeline( // define a pipeline consisting of the individual control blocks and in which order
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main; // make sure the switch uses the defined pipeline
