#include <core.p4>
#include <tna.p4>

#include "../common/headers.p4"
#include "../common/util.p4"
#include "../headers.p4"

// Hardcoded ID of each switch, needed for letting the switches to communicate with each other
#define SWITCH_ID 16w100 

control SpineIngress(
        inout saqr_header_t hdr,
        inout saqr_metadata_t saqr_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

    Random<bit<16>>() random_ds_id;

    Register<queue_len_t, _>(ARRAY_SIZE) queue_len_list_1; // List of queue lens for all vclusters
                RegisterAction<queue_len_t, _, queue_len_t>(queue_len_list_1) read_queue_len_list_1 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        rv = value;
                    }
                };
                 RegisterAction<queue_len_t, _, queue_len_t>(queue_len_list_1) write_queue_len_list_1 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        value = hdr.saqr.qlen;
                        rv = value;
                    }
                };

            Register<queue_len_t, _>(ARRAY_SIZE) queue_len_list_2; // List of queue lens for all vclusters
                RegisterAction<queue_len_t, _, queue_len_t>(queue_len_list_2) read_queue_len_list_2 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        rv = value;
                    }
                };
                RegisterAction<queue_len_t, _, queue_len_t>(queue_len_list_2) write_queue_len_list_2 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        value = hdr.saqr.qlen;
                        rv = value;
                    }
                };
    
    /********  Action/table decelarations *********/
    action _drop() {
        ig_intr_dprsr_md.drop_ctl = 0x1; // Drop packet.
    }

    action get_leaf_start_idx () {
        saqr_md.cluster_ds_start_idx = (bit <16>) (hdr.saqr.cluster_id * MAX_LEAFS_PER_CLUSTER);
    }
    
    action gen_random_leaf_index_16() {
        saqr_md.random_ds_index_1 = (bit<16>) random_ds_id.get();
        saqr_md.random_ds_index_2 = (bit<16>) random_ds_id.get();

    }
    action adjust_random_leaf_index_8() {
        saqr_md.random_ds_index_1 = saqr_md.random_ds_index_1 >> 8;
        saqr_md.random_ds_index_2 = saqr_md.random_ds_index_2 >> 8;
    }

    action adjust_random_leaf_index_4() {
        saqr_md.random_ds_index_1 = saqr_md.random_ds_index_1 >> 12;
        saqr_md.random_ds_index_2 = saqr_md.random_ds_index_2 >> 12;
    }

    action adjust_random_leaf_index_2() {
        saqr_md.random_ds_index_1 = saqr_md.random_ds_index_1 >> 14;
        saqr_md.random_ds_index_2 = saqr_md.random_ds_index_2 >> 14;
    }

    action adjust_random_leaf_index_1() {
        saqr_md.random_ds_index_1 = saqr_md.random_ds_index_1 >> 15;
        saqr_md.random_ds_index_2 = saqr_md.random_ds_index_2 >> 15;
    }

    table adjust_random_range_sq_leafs { // Adjust the random generated number (16 bit) based on number of available queue len signals
        key = {
            saqr_md.cluster_num_valid_queue_signals: exact; 
        }
        actions = {
            adjust_random_leaf_index_8(); // # == 256
            adjust_random_leaf_index_4(); // == 16
            adjust_random_leaf_index_2(); // == 4
            adjust_random_leaf_index_1(); // == 2
            NoAction; // == 2^16
        }
        size = 16;
        default_action = NoAction;
    }
    
    action act_forward_saqr(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port;
        // TESTBEDONLY: comment the line below when no need for emulating multiple leaf schedulers using one switch. 
        // See Saqr spine comments for details.
        hdr.saqr.cluster_id = hdr.saqr.dst_id; // We use different cluster ids for each virtual leaf switch 
    }
    table forward_saqr_switch_dst {
        key = {
            hdr.saqr.dst_id: exact;
        }
        actions = {
            act_forward_saqr;
            NoAction;
        }
        size = HDR_SRC_ID_SIZE;
        default_action = NoAction;
    }
    
    action act_get_cluster_num_valid_leafs(bit<16> num_leafs) {
        saqr_md.cluster_num_valid_queue_signals = num_leafs;
    }
    table get_cluster_num_valid_leafs { // TODO: fix typo: Leaves !
        key = {
            hdr.saqr.cluster_id : exact;
        }
        actions = {
            act_get_cluster_num_valid_leafs;
            NoAction;
        }
        size = HDR_CLUSTER_ID_SIZE;
        default_action = NoAction;
    }
    
    /////////
    action act_get_rand_leaf_id_1(bit <16> leaf_id){
        saqr_md.random_id_1 = leaf_id;
    }
    table get_rand_leaf_id_1 {
        key = {
            saqr_md.random_ds_index_1: exact;
            hdr.saqr.cluster_id: exact;
        }
        actions = {
            act_get_rand_leaf_id_1();
            NoAction;
        }
        size = 16;
        default_action = NoAction;
    }
    action act_get_rand_leaf_id_2(bit <16> leaf_id){
        saqr_md.random_id_2 = leaf_id;
    }
    table get_rand_leaf_id_2 {
        key = {
            saqr_md.random_ds_index_2: exact;
            hdr.saqr.cluster_id: exact;
        }
        actions = {
            act_get_rand_leaf_id_2();
            NoAction;
        }
        size = 16;
        default_action = NoAction;
    }
    ///////////
    action compare_queue_len() {
        saqr_md.selected_ds_qlen = min(saqr_md.random_ds_qlen_1, saqr_md.random_ds_qlen_2);
    }
    /********  Control block logic *********/
    apply {
        if (hdr.saqr.isValid()) {  // Saqr packet
           
        if (hdr.saqr.dst_id == SWITCH_ID) { // If this packet is destined for this spine do saqr processing ot. its just an intransit packet we need to forward on correct port
            // TESTBEDONLY: See saqr spine comments. comment the line below when no need for emulating multiple leaf schedulers.
            hdr.saqr.cluster_id = 0;
            @stage(1) {
                get_leaf_start_idx ();
                get_cluster_num_valid_leafs.apply();
                gen_random_leaf_index_16();
            }

            @stage(2) {
                adjust_random_range_sq_leafs.apply(); //  select a random leaf index
            }

            @stage(3) {
                get_rand_leaf_id_1.apply(); // Read the leaf ID associated with generated random index
                get_rand_leaf_id_2.apply(); // Read the leaf ID associated with generated random index
            }

            @stage(4) {
                if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK){
                    saqr_md.random_ds_qlen_1 = read_queue_len_list_1.execute(saqr_md.random_id_1);
                    saqr_md.random_ds_qlen_2 = read_queue_len_list_2.execute(saqr_md.random_id_2);
                } else if (hdr.saqr.pkt_type == PKT_TYPE_QUEUE_SIGNAL) {
                    write_queue_len_list_1.execute(hdr.saqr.src_id); // Write the qlen at corresponding index for the leaf in this cluster
                    write_queue_len_list_2.execute(hdr.saqr.src_id); // Write the qlen at corresponding index for the leaf in this cluster
                    hdr.saqr.dst_id = INVALID_VALUE_16bit; // Drop the packet
                }
            }

            @stage(5) {
                compare_queue_len();
            }

            @stage(6){
                if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                    if (saqr_md.selected_ds_qlen == saqr_md.random_ds_qlen_1) {
                        hdr.saqr.dst_id = saqr_md.random_id_1;
                    } else {
                        hdr.saqr.dst_id = saqr_md.random_id_2;
                    }
                }
            }
            
        }
        hdr.saqr.src_id = SWITCH_ID;
        forward_saqr_switch_dst.apply();
            
        } else if (hdr.ipv4.isValid()) { // Regular switching procedure
            // TODO: Not ported the ip matching tables for now
            _drop();
        } else {
            _drop();
        }
    }
}

control SpineIngressDeparser(
        packet_out pkt,
        inout saqr_header_t hdr,
        in saqr_metadata_t saqr_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {

    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.saqr);
    }
}

// Empty egress parser/control blocks
parser SpineEgressParser(
        packet_in pkt,
        out saqr_header_t hdr,
        out eg_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

control SpineEgressDeparser(
        packet_out pkt,
        inout saqr_header_t hdr,
        in eg_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md) {
    apply {}
}

control SpineEgress(
        inout saqr_header_t hdr,
        inout eg_metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
    apply {}
}
