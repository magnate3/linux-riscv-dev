#include <core.p4>
#include <tna.p4>

#include "common/headers.p4"
#include "common/util.p4"

// const bit<9> LoopBackPort = 0x0;
const PortId_t LoopBackPort = 164;
const PortId_t OutputPort = 132;


// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------




struct metadata_t{
    bit<32> flow_index;          
    bit<32> weight;               
    bit<32> limit_normalized;    
    bit<32> flow_round_index; 
    bit<32> round;                 
    bit<32> round_mult_wf;          
    bit<32> compare_unit_temp;      
    bit<32> compare_unit;          
    bit<32> round_add;         
    bit<32> accept_flag;
    bit<3> drop_or_not;
    bit<32> round_number_eg;
}



parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t meta,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    TofinoIngressParser() tofino_parser;
    state start {
        tofino_parser.apply(pkt, ig_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet{
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            17 : parse_udp;
            6 :  parse_tcp;
            default : reject;
        }
    }

    state parse_udp{
        pkt.extract(hdr.udp);
        transition select(hdr.udp.dst_port){
            7001:  parse_worker;   //worker
            default: parse_WFQ;
        }
    }

    state parse_tcp{
        pkt.extract(hdr.tcp);
        transition accept;
    }

    state parse_worker{
        pkt.extract(hdr.worker_t);
        transition accept;
    }

    state parse_WFQ{
        pkt.extract(hdr.wfq_t);
        transition accept;
    }

}

control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t meta,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    Checksum() ipv4_checksum;
    apply {        
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
            hdr.ipv4.dst_addr});

         pkt.emit(hdr);
    }
}

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t meta,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md
){

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
            // hdr.worker_t.isValid(): exact;
        }

        actions = {
            forward;
            drop;
        }
        const default_action = drop();   
        size = 512;
    }

      //table for round index:
    action get_workerround_index_action(bit<32> round_index){
        hdr.worker_t.round_index = round_index;
    }
    table get_workerround_index_table{
        key = {
            hdr.worker_t.egress_port: exact;
            hdr.worker_t.qid: exact;
        }
        actions = {
            get_workerround_index_action;
        }
        size = 128;
    }
    //table for get round index (not worker):
    action get_round_index_action(bit<32> flow_round_index){
        meta.flow_round_index = flow_round_index; 
    }
    table get_flow_round_index_table{
        key = {
            ig_tm_md.ucast_egress_port:exact;
            ig_tm_md.qid:exact;
        }
        actions = {
            get_round_index_action;
        }
        size = 128;
    }



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


    //table getweightUDP
    action get_weightindex_UDP(bit<32> flow_idx){
        meta.flow_index = flow_idx;      //flow_index
    }
    table get_weightindex_UDP_table{
        key = {
            hdr.ipv4.src_addr: exact;
            hdr.udp.dst_port : exact;
        }
        actions = {
            get_weightindex_UDP;
        }   
        size = 512;
    }
    

    //ingress round register
    Register<bit<32>,bit<5>> (32,0) Ingress_Round_Reg;
    RegisterAction<bit<32>,bit<5>,bit<32>> (Ingress_Round_Reg) Set_Ingress_Round_REG_Action = {
        void apply(inout bit<32> value){
            value = hdr.worker_t.round_number;
        }
    };

    RegisterAction<bit<32>,bit<5>,bit<32>> (Ingress_Round_Reg) Get_Ingress_Round_REG_Action = {
        void apply(inout bit<32> value,out bit<32> result){
            result = value;
        }
    };


    //Bf register
    Register<bit<32>,bit<32>> (32w500,0) Packet_Sent_Reg;
    RegisterAction<bit<32>,bit<32>,bit<3>> (Packet_Sent_Reg) UpdateBf_Action = {
        void apply(inout bit<32> value,out bit<3> result){   
            if(value + add_unit_Bf <= meta.compare_unit){
                result =0;
                if(value<meta.round_mult_wf)
                {
                    value = meta.round_mult_wf + add_unit_Bf;
                }
                else
                    value = value+add_unit_Bf;
            }
            else{
                result = 1;
            }
        }
    };




    //Get limit(Q*wf/R)
    action get_limit_action(bit<32> limit){
        meta.limit_normalized = limit;     
    }
    table get_limit_table{
        key = {
            meta.weight:exact;
        }
        actions = {
            get_limit_action;
        }
        const default_action = get_limit_action(0);   
        size = 512;
    }
    

    //Get weight ,1/wf
    action get_weight_action(bit<32> weight){
        meta.weight = weight;      
    }
    table get_weight_table{
        key = {
            meta.flow_index:exact;
        }
        actions = {
            get_weight_action;
        }  
        size = 512;
    }


    //get r*wf
    action shift_r_0(){
        meta.round_mult_wf = meta.round;
    }
    action shift_r_1(){
        meta.round_mult_wf = meta.round>>1;
    }
    action shift_r_2(){
        meta.round_mult_wf = meta.round>>2;
    }
    action shift_r_3(){
        meta.round_mult_wf = meta.round>>3;
    }
    action shift_r_4(){
        meta.round_mult_wf = meta.round>>4;
    }
    action shift_r_5(){
        meta.round_mult_wf = meta.round>>5;
    }
    action shift_r_6(){
        meta.round_mult_wf = meta.round>>6;
    }
    action shift_r_7(){
        meta.round_mult_wf = meta.round>>7;
    }
    action shift_r_8(){
        meta.round_mult_wf = meta.round>>8;
    }

    table tbl_get_rwf{
        key = {
            meta.weight:exact;
        }
        actions = {
            shift_r_0;
            shift_r_1;
            shift_r_2;
            shift_r_3;
            shift_r_4;
            shift_r_5;
            shift_r_6;
            shift_r_7;
            shift_r_8;
        }
        const default_action = shift_r_0();   
        size = 512;
    }

    apply{
         
        table_forward.apply(); 
        // ig_tm_md.bypass_egress = 1w1;    //just for debug
        //set round


        if(hdr.worker_t.isValid()){
            //set r
            get_workerround_index_table.apply();
            Set_Ingress_Round_REG_Action.execute(hdr.worker_t.round_index);
            ig_tm_md.ucast_egress_port = LoopBackPort;
            ig_dprsr_md.drop_ctl = 0;
        }
        else if(ig_dprsr_md.drop_ctl == 0 && ig_tm_md.ucast_egress_port == OutputPort)
        {
                // get flow_index
                if(hdr.udp.isValid()){
                    get_weightindex_UDP_table.apply();
                }
                else{             
                    get_weightindex_TCP_table.apply();
                }
                //get wf
                get_weight_table.apply();
                //get Q*wf
                get_limit_table.apply();
                //get round index
                get_flow_round_index_table.apply();
                //get round
                meta.round = Get_Ingress_Round_REG_Action.execute(meta.flow_round_index);
                
                //get r*wf
                tbl_get_rwf.apply();
                //get comparison
                meta.compare_unit = meta.round_mult_wf +meta.limit_normalized;
                //make decision
                ig_dprsr_md.drop_ctl = UpdateBf_Action.execute(meta.flow_index);
                
        }
    }
}

parser SwitchEgressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t meta_eg,
        out egress_intrinsic_metadata_t eg_intr_md) {


    TofinoEgressParser() tofino_eparser;
    state start {
        tofino_eparser.apply(pkt, eg_intr_md);
        transition parse_ethernet;
    }
    state parse_ethernet{
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            17 : parse_udp;
            6 :  parse_tcp;
            default : reject;
        }
    }

    state parse_udp{
        pkt.extract(hdr.udp);
        transition select(hdr.udp.dst_port){
            7001:  parse_worker;   //worker
            default : parse_WFQ;
        }
    }

    state parse_tcp{
        pkt.extract(hdr.tcp);
        transition accept;
    }

    state parse_worker{
        pkt.extract(hdr.worker_t);
        transition accept;
    }

    state parse_WFQ{
        pkt.extract(hdr.wfq_t);
        transition accept;
    }

}

control SwitchEgress(
    inout header_t hdr,
    inout metadata_t meta_eg,
    in egress_intrinsic_metadata_t eg_intr_md,
    in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
    inout egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprs,
    inout egress_intrinsic_metadata_for_output_port_t eg_intr_md_for_oport
) {
    
    //get round add 
    action get_round_add_action(bit<32> ra){
        meta_eg.round_add = ra;
    }
    table get_round_add_tbl{
        key = {
            eg_intr_md.deq_qdepth:range;
        }
        actions = {
            get_round_add_action;
        }
        const default_action = get_round_add_action(0);   
        size = 512;
    }
    

     //Engress round register
    Register<bit<32>,bit<5>> (32,0) Egress_Round_Reg;
    RegisterAction<bit<32>,bit<5>,bit<32>> (Egress_Round_Reg) Set_Egress_Round_REG_Action = {
        void apply(inout bit<32> value,out bit<32> result){
            value = value + meta_eg.round_add;
            result = value;
        }
    };

    RegisterAction<bit<32>,bit<5>,bit<32>> (Egress_Round_Reg) Get_Egress_Round_REG_Action = {
        void apply(inout bit<32> value,out bit<32> result){
            result = value;
        }
    };


    apply{
        
        if( !hdr.worker_t.isValid() && eg_intr_md.egress_port == OutputPort)
        {
            //update round
            get_round_add_tbl.apply();
            Set_Egress_Round_REG_Action.execute(0);

        }    
        else if(hdr.worker_t.isValid()){
            //get round
            hdr.worker_t.round_number  = Get_Egress_Round_REG_Action.execute(hdr.worker_t.round_index);
        }


    }
}

control SwitchEgressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t meta_eg,
        in egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprs) {
    Checksum() ipv4_checksum;
    apply {        
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
            hdr.ipv4.dst_addr});

         pkt.emit(hdr);
    }
}

Pipeline(
    SwitchIngressParser(),
    SwitchIngress(),
    SwitchIngressDeparser(),
    SwitchEgressParser(),
    SwitchEgress(),
    SwitchEgressDeparser()) pipe;

Switch(pipe) main;

