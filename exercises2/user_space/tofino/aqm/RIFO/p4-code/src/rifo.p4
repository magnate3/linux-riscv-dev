/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>
/*************************************************************************
************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/
const bit<16> BufferSize=25000;
#define RIFO_PORT 9000
#define RIFOWORKER_PORT 9001
#define rank_range_threshold 150
#define rifo_index 0

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

header rifoWorker_h {
    bit<18>     qlength;    // Queue occupancy in cells
    bit<1>      ping_pong;  // Reserved for internal purposes
    bit<11>     qid;        // port_group[3:0] ++ queue_id[6:0]
    bit<2>      pipe_id;
}

header rifo_h {
   bit<16> rank;
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
   rifo_h              rifo;
   rifoWorker_h       rifoWorker;
}

   /******  G L O B A L   I N G R E S S   M E T A D A T A  *********/

struct my_ingress_metadata_t {
   bit<16>      queue_length;
   bit<16>      available_queue;
   bit<16>      min_pkt_rank;
   bit<16>      max_pkt_rank;
   bit<16>      dividend;
   bit<16>      divisor;
   bit<24>      left_side;
   bit<24>      right_side;
   bit<24>      rifo_admission;
   bit<16>      rank_range;
   bit<16>      max_min;
   bit<5>       max_min_exponent;
   bit<5>       buffer_exponent;
   bit<5>       dividend_exponent;

}
    /***********************  P A R S E R  **************************/

parser TofinoIngressParser(
       packet_in pkt,
       inout my_ingress_metadata_t ig_md,
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
       transition parse_rifo;
   }

   state parse_udp {
       pkt.extract(hdr.udp);
       transition select(hdr.udp.dst_port) {
           RIFO_PORT: parse_rifo;
           RIFOWORKER_PORT: parse_rifoWorker;
           default: accept;
       }
   }

   state parse_rifo {
       pkt.extract(hdr.rifo);
       transition accept;
   }

   state parse_rifoWorker {
       pkt.extract(hdr.rifoWorker);
       transition accept;
   }
}

parser SwitchIngressParser(
       packet_in pkt,
       out headers_t hdr,
       out my_ingress_metadata_t ig_md,
       out ingress_intrinsic_metadata_t ig_intr_md) {

   TofinoIngressParser() tofino_parser;
   EtherIPTCPUDPParser() layer4_parser;

   state start {
       tofino_parser.apply(pkt, ig_md, ig_intr_md);
       layer4_parser.apply(pkt, hdr);
       transition accept;
   }
}
   /***************** M A T C H - A C T I O N  *********************/

control Ingress(
   /* User */
   inout headers_t                       hdr,
   inout my_ingress_metadata_t                      ig_md,
   /* Intrinsic */
   in    ingress_intrinsic_metadata_t               ig_intr_md,
   in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
   inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
   inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{
   action send(PortId_t port) {
       ig_tm_md.ucast_egress_port = port;
   }

   action drop() {
       ig_dprsr_md.drop_ctl = 1;
   }
   table ipv4_host {
       key     = { hdr.ipv4.dst_addr : exact; }
       actions = { send; drop; }

       //default_action = send(64);
       size           = 512;
   }

   // register to store the queue length (l)
    Register<bit<16>, _>(32w1) ig_queue_length_reg;
    RegisterAction<bit<16>, _, bit<16>>(ig_queue_length_reg) ig_queue_length_reg_write = {
       void apply(inout bit<16> value, out bit<16> read_value){
            value=hdr.rifoWorker.qlength[15:0];
            read_value = value;
       }
   };
   RegisterAction<bit<16>, _, bit<16>>(ig_queue_length_reg) ig_queue_length_reg_read = {
       void apply(inout bit<16> value, out bit<16> read_value){
               read_value = value;
       }
   };
   /* registers to track min and max values of ranks*/
   Register<bit<16>, _>(32w1) min_rank_reg;
   RegisterAction<bit<16>, _, bit<16>>(min_rank_reg) min_rank_reg_write_action = {
       void apply(inout bit<16> value, out bit<16> read_value){
           if (value == 0x0)
               {
               value = hdr.rifo.rank;
           }
           else if(hdr.rifo.rank < value){
               value= hdr.rifo.rank;
           }
           read_value=value;
       }
   };

   Register<bit<16>, _>(32w1) max_rank_reg;
   RegisterAction<bit<16>, _, bit<16>>(max_rank_reg) max_rank_reg_write_action = {
       void apply(inout bit<16> value, out bit<16> read_value){
           if(hdr.rifo.rank > value)
               {
                   value = hdr.rifo.rank;
           }
           read_value=value;
       }
   };

   action action_subtract_queueLength() {
           ig_md.available_queue=(bit<16>) (BufferSize - ig_md.queue_length);
       }

    table subtract_queueLength{
       actions = { action_subtract_queueLength;}
       default_action = action_subtract_queueLength();
       size=1;
   }
   action action_compute_dividend(){
       ig_md.dividend = hdr.rifo.rank - ig_md.min_pkt_rank;
   }
   table compute_dividend{
       actions = { action_compute_dividend;}
       default_action = action_compute_dividend();
       size=1;
   }
   action action_compute_divisor(){
       ig_md.divisor = ig_md.max_pkt_rank - ig_md.min_pkt_rank;
       ig_md.max_min=ig_md.divisor;
    }
   table compute_divisor{
       actions = { action_compute_divisor;}
       default_action = action_compute_divisor();
       size=1;
   }

   action action_calculate_left_side(){
    ig_md.left_side =(bit<24>) ig_md.dividend_exponent << 14;
   }

   table calculate_left_side{
       actions = { action_calculate_left_side;}
       default_action = action_calculate_left_side();
       size=1;
   }

   action action_do_RIFO_admission(){
       ig_md.rifo_admission = max( ig_md.left_side, ig_md.right_side);
    }
    table RIFO_admission{
       actions = { action_do_RIFO_admission;}
       default_action = action_do_RIFO_admission();
       size=1;
   }

   action recirculation(bit<9> port){
       ig_tm_md.ucast_egress_port = port;
   }

   action rifoWorker_recirculate(){
       //packet routing: for now we simply bounce back the packet.
       //any routing match-action logic should be added here.
       ig_tm_md.ucast_egress_port=196;
   }
   action set_rank(){
       hdr.rifo.rank =(bit<16>) hdr.ipv4.src_addr[7:0];
    }
   action set_exponent_buffer(bit<5> exponent_value){
       ig_md.buffer_exponent = exponent_value ;
    }
   table queue_length_lookup {
       key = {
           ig_md.available_queue: ternary;
       }
       actions = {
           set_exponent_buffer;
       }
       size = 512;
   }

    action set_exponent_dividend(bit<5> exponent_value){
       ig_md.dividend_exponent= exponent_value ;
    }
   table dividend_lookup {
       key = {
           ig_md.dividend: ternary;
       }
       actions = {
           set_exponent_dividend;
       }
       size = 512;
   }

   action set_exponent_max_min(bit<5> exponent_value){
       ig_md.max_min_exponent= exponent_value ;
    }
   table max_min_lookup {
       key = {
           ig_md.max_min: ternary;
       }
       actions = {
           set_exponent_max_min;
       }
       size = 512;
   }

   action action_get_ig_queue_length(){
        ig_md.queue_length=ig_queue_length_reg_read.execute(rifo_index);

   }

   table get_ig_queue_length {
       actions = {
           action_get_ig_queue_length();
       }
       default_action =action_get_ig_queue_length();
       size = 1;
   }

   action action_set_ig_queue_length(){
        ig_queue_length_reg_write.execute(rifo_index);
   }

   table set_ig_queue_length {
       actions = {
           action_set_ig_queue_length();
       }
       default_action =action_set_ig_queue_length();
       size = 1;
   }


   action calculate_max_min_buffer_mul(bit<24> mul){
       ig_md.right_side= mul ;
    }
   table max_min_buffer_lookup {
       key = {
           ig_md.max_min_exponent: exact;
           ig_md.buffer_exponent: exact;
       }
       actions = {
           calculate_max_min_buffer_mul;
       }
       size = 512;
   }
   apply {
        if (hdr.ipv4.isValid()) {

            // do routing to get the egress port and qid
            ipv4_host.apply();
            set_rank();
            if(hdr.rifoWorker.isValid()){
                    set_ig_queue_length.apply();
                    rifoWorker_recirculate();
            }
            else if(hdr.udp.isValid() || hdr.tcp.isValid()){

                //Get Max and Min ranks
                ig_md.min_pkt_rank = min_rank_reg_write_action.execute(rifo_index);
                ig_md.max_pkt_rank = max_rank_reg_write_action.execute(rifo_index);

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

                if ( ig_md.rifo_admission == ig_md.left_side) {
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
   in    my_ingress_metadata_t                      ig_md,
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
   out my_egress_metadata_t         eg_md,
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
   inout my_egress_metadata_t                         eg_md,
   /* Intrinsic */
   in    egress_intrinsic_metadata_t                  eg_intr_md,
   in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
   inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
   inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{

    Register<bit<16>, _>(32w1) eg_queue_length_reg;
    RegisterAction<bit<16>, _, bit<16>>(eg_queue_length_reg) eg_queue_length_reg_write = {
       void apply(inout bit<16> value, out bit<16> read_value){
            value = eg_intr_md.enq_qdepth[15:0];
            read_value = value;
       }
   };
   RegisterAction<bit<16>, _, bit<16>>(eg_queue_length_reg) eg_queue_length_reg_read = {
       void apply(inout bit<16> value, out bit<16> read_value){
            read_value = value;
       }
   };

   apply {
	   if(hdr.rifoWorker.isValid()){
            hdr.rifoWorker.qlength=(bit<18>)eg_queue_length_reg_read.execute(0);
  	    }
        else if (hdr.rifo.isValid()){
            eg_queue_length_reg_write.execute(0);
        }
   }
}
   /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
   /* User */
   inout headers_t                       hdr,
   in    my_egress_metadata_t                      eg_md,
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
