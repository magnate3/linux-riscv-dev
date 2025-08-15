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
    bit<32> round;                   
    bit<32> round_mult_wf;                  
    bit<32> compare_unit;          
    bit<32> round_add;          
    bit<32> pkt_round_idx;
    // bit<16> timediff_high;
    bit<16> timediff_low;
    bit<16> newavetime;
    bit<16> timeindex;
    bit<16> time0;
    bit<16> time1;
    bit<16> time2;
    bit<16> time3;
    bit<16> time1_1;
    bit<16> time2_2;
    bit<16> time_history;
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
        //inout metadata_t ig_md,
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
        }

        actions = {
            forward;
            drop;
        }
        const default_action = drop();   
        size = 512;
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
    Register<bit<32>,bit<32>> (1,0) Ingress_Round_Reg;
    RegisterAction<bit<32>,bit<32>,bit<32>> (Ingress_Round_Reg) Set_Ingress_Round_REG_Action = {
        void apply(inout bit<32> value){
            value = hdr.worker_t.round_number;
        }
    };

    RegisterAction<bit<32>,bit<32>,bit<32>> (Ingress_Round_Reg) Get_Ingress_Round_REG_Action = {
        void apply(inout bit<32> value,out bit<32> result){
            result = value;
        }
    };


    //Bf register
    Register<bit<32>,bit<32>> (500,0) Packet_Sent_Reg;
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



    action GetRoundIndex(bit<32> roundidx)
    {
        meta.pkt_round_idx = roundidx;
    }
    table tbl_Get_Ingress_Round{
        key = {
            ig_tm_md.ucast_egress_port:exact;
            ig_tm_md.qid:exact;
        }
        actions = {
            GetRoundIndex;
        }
        const default_action = GetRoundIndex(0);
        size = 512;
    }
    

    //Get weight ,1/wf
    action get_weight_action(bit<32> weight,bit<32> limit){
        meta.weight = weight;      
        meta.limit_normalized = limit;  
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

    //newf table
    action get_limit_new(bit<32> new_limit){
        meta.limit_normalized = new_limit;
    }
    action no_change(){}

    table tbl_get_new_wf{
        key = {
            meta.newavetime:range;
            // meta.newavesize:range;
            meta.weight:exact;
        }
        actions = {
            get_limit_new;
            no_change;
        }
        const default_action = no_change();   
    }

    //timestamp:
    Register<bit<16>,bit<32>> (512) TimestampReg;
    RegisterAction<bit<16>,bit<32>,bit<16>> (TimestampReg) UpdateTime = {
        void apply(inout bit<16> value,out bit<16> result){
            result = ig_prsr_md.global_tstamp[15:0] - value;
            value = ig_prsr_md.global_tstamp[15:0];
        }
    };

    //time ring buffer
    
    Register<bit<16>,bit<16>> (512) Time0;
    RegisterAction<bit<16>,bit<16>,bit<16>> (Time0) GetTime0 = {
        void apply(inout bit<16> value,out bit<16> result){
            result = value;
        }
    }; 
    RegisterAction<bit<16>,bit<16>,bit<16>> (Time0) UpdateTime0 = {
        void apply(inout bit<16> value,out bit<16> result){
            result = value;
            value =  meta.timediff_low;
        }
    };


    Register<bit<16>,bit<16>> (512) Time1;
    RegisterAction<bit<16>,bit<16>,bit<16>> (Time1) GetTime1 = {
        void apply(inout bit<16> value,out bit<16> result){
            result = value;
        }
    }; 
    RegisterAction<bit<16>,bit<16>,bit<16>> (Time1) UpdateTime1 = {
        void apply(inout bit<16> value,out bit<16> result){
            result = value;
            value =  meta.timediff_low;
        }
    };

    Register<bit<16>,bit<16>> (512) Time2;
    RegisterAction<bit<16>,bit<16>,bit<16>> (Time2) GetTime2 = {
        void apply(inout bit<16> value,out bit<16> result){
            result = value;
        }
    }; 
    RegisterAction<bit<16>,bit<16>,bit<16>> (Time2) UpdateTime2 = {
        void apply(inout bit<16> value,out bit<16> result){
            result = value;
            value =  meta.timediff_low;
        }
    };

    Register<bit<16>,bit<16>> (512) Time3;
    RegisterAction<bit<16>,bit<16>,bit<16>> (Time3) GetTime3 = {
        void apply(inout bit<16> value,out bit<16> result){
            result = value;
        }
    }; 
    RegisterAction<bit<16>,bit<16>,bit<16>> (Time3) UpdateTime3 = {
        void apply(inout bit<16> value,out bit<16> result){
            result = value;
            value =  meta.timediff_low;
        }
    };


    Register<bit<16>,bit<32>> (512) TimeIdx;
    RegisterAction<bit<16>,bit<32>,bit<16>> (TimeIdx) UpdateTimeIdx = {
        void apply(inout bit<16> value,out bit<16> result){
            if(value < 3)
            {
                value = value+1;
            }
            else{
                value = 0;
            }
            result = value;
        }
    }; 

    

    action addtime()
    {
        meta.time_history = meta.time1_1 + meta.time2_2;
    }

    table tbl_addtime_history{
        key = {}
        actions = {addtime;}
        const default_action = addtime();
    }


    apply{
        table_forward.apply();
        if(hdr.worker_t.isValid()){
            //set r
            Set_Ingress_Round_REG_Action.execute(0);
            ig_tm_md.ucast_egress_port = LoopBackPort;    
            ig_dprsr_md.drop_ctl = 0;
        }
        else if(ig_dprsr_md.drop_ctl == 0 && ig_tm_md.ucast_egress_port == OutputPort)
        {


            if(hdr.udp.isValid()){
                get_weightindex_UDP_table.apply();
            }
            else{             
                get_weightindex_TCP_table.apply();
            }
            get_weight_table.apply();  
           
            // get time
            meta.timediff_low = UpdateTime.execute(meta.flow_index);
            meta.timeindex = UpdateTimeIdx.execute(meta.flow_index);
            // tbl_gettime.apply();
            // tbl_shifttime.apply();
            if(meta.timeindex == 0)
            {
                meta.time0 = UpdateTime0.execute(meta.timeindex) ;
                meta.time1 = GetTime1.execute(meta.timeindex) ;
                meta.time2 = GetTime2.execute(meta.timeindex) ;
                meta.time3 = GetTime3.execute(meta.timeindex);
                meta.time0 = meta.time0>>3;
                meta.time1 = meta.time1>>3;
                meta.time2 = meta.time2>>2;
                meta.time3 = meta.time3>>1;
            }
            else if (meta.timeindex == 1)
            {
                meta.time0 = GetTime0.execute(meta.timeindex) ;
                meta.time1 = UpdateTime1.execute(meta.timeindex) ;
                meta.time2 = GetTime2.execute(meta.timeindex) ;
                meta.time3 = GetTime3.execute(meta.timeindex);
                meta.time0 = meta.time0>>1;
                meta.time1 = meta.time1>>3;
                meta.time2 = meta.time2>>3;
                meta.time3 = meta.time3>>2;
            }
            else if (meta.timeindex ==2)
            {
                meta.time0 = GetTime0.execute(meta.timeindex) ;
                meta.time1 = GetTime1.execute(meta.timeindex) ;
                meta.time2 = UpdateTime2.execute(meta.timeindex) ;
                meta.time3 = GetTime3.execute(meta.timeindex);
                meta.time0 = meta.time0>>2;
                meta.time1 = meta.time1>>1;
                meta.time2 = meta.time2>>3;
                meta.time3 = meta.time3>>3;
            }
            else
            {
                meta.time0 = GetTime0.execute(meta.timeindex) ;
                meta.time1 = GetTime1.execute(meta.timeindex) ;
                meta.time2 = GetTime2.execute(meta.timeindex) ;
                meta.time3 = UpdateTime3.execute(meta.timeindex);
                meta.time0 = meta.time0>>3;
                meta.time1 = meta.time1>>2;
                meta.time2 = meta.time2>>1;
                meta.time3 = meta.time3>>3;
            }
            meta.time1_1 = meta.time0 + meta.time1;
            meta.time2_2 = meta.time2 + meta.time3;
            meta.timediff_low = meta.timediff_low>>1;
            tbl_addtime_history.apply();
            meta.newavetime = meta.timediff_low + meta.time_history;
            tbl_get_new_wf.apply();
            //get round
            tbl_Get_Ingress_Round.apply();
            meta.round = Get_Ingress_Round_REG_Action.execute(0);
                
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
        out metadata_t meta,
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
    Register<bit<32>,bit<32>> (16,0) Egress_Round_Reg;
    RegisterAction<bit<32>,bit<32>,bit<32>> (Egress_Round_Reg) Set_Egress_Round_REG_Action = {
        void apply(inout bit<32> value){
             value = value + meta_eg.round_add;
        }
    };

    RegisterAction<bit<32>,bit<32>,bit<32>> (Egress_Round_Reg) Get_Egress_Round_REG_Action = {
        void apply(inout bit<32> value,out bit<32> result){
            result = value;
        }
    };

 //action Get_Round_Idx
    action GetRoundIndex(bit<32> roundidx)
    {
        meta_eg.pkt_round_idx = roundidx;
    }
    table tbl_Get_Ingress_Round{
        key = {
            eg_intr_md.egress_port:exact;
            eg_intr_md.egress_qid:exact;
        }
        actions = {
            GetRoundIndex;
        }
        const default_action = GetRoundIndex(0);
        size = 512;
    }


    apply{
        if( !hdr.worker_t.isValid() && eg_intr_md.egress_port == OutputPort)
        {
            //update round
            get_round_add_tbl.apply();
            tbl_Get_Ingress_Round.apply();
            Set_Egress_Round_REG_Action.execute(meta_eg.pkt_round_idx);
            // hdr.wfq_t.enqueue_depth[18:0] = eg_intr_md.enq_qdepth;
            // hdr.wfq_t.dequeue_depth[18:0] = eg_intr_md.deq_qdepth;
            // hdr.wfq_t.round_add = meta_eg.round_add;
            // hdr.wfq_t.egress_round =  Set_Egress_Round_REG_Action.execute(0);
        }    
        else if(hdr.worker_t.isValid()){
            //get round
            //  meta_eg.round_number_eg = Get_Egress_Round_REG_Action.execute(0);
            //  hdr.worker_t.round_number =meta_eg.round_number_eg ;
            // hdr.worker_t.qid = Get_Egress_Round_REG_Action.execute(0);
            hdr.worker_t.round_number  = Get_Egress_Round_REG_Action.execute(0);
        }    
    }
}

control SwitchEgressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t meta,
        in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
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

