#include <core.p4>
#include <tna.p4>

#include "../common/headers.p4"
#include "../common/util.p4"
#include "../headers.p4"


/*
 * Implementation of Saqr Spine scheduler
 *
 * Comments with the tag <TESTBEDONLY> mark the parts of the code that were modified for emulating multiple leaf schedulers using one switch.
 * These lines should be changed for normal operation (instructions for the changes are also provided in the commments)
 * 
 * The main procedures for scheduling is similar to Saqr Leaf (avoided duplicate explanation comments),
  please refer to the leaf for details
 * Explanation for parts that are different from leaf logic are commented in this code.
*/

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
    Random<bit<16>>() random_ds_id_retry;
    Random<bit<1>>() random_helper_1bit;

    /********  Register decelarations *********/
    Register<leaf_id_t, _>(MAX_LEAFS) idle_list; // Maintains the list of idle leafs for each vcluster (array divided based on cluster_id)
        RegisterAction<bit<16>, _, bit<16>>(idle_list) add_to_idle_list = {
            void apply(inout bit<16> value, out bit<16> rv) {
                value = hdr.saqr.src_id;
                rv = value;
            }
        };
        RegisterAction<leaf_id_t, _, leaf_id_t>(idle_list) write_idle_list = {
            void apply(inout bit<16> value, out bit<16> rv) {
                value = saqr_md.task_resub_hdr.ds_index_2;
                rv = value;
            }
        };
        RegisterAction<leaf_id_t, _, leaf_id_t>(idle_list) read_idle_list = {
            void apply(inout bit<16> value, out bit<16> rv) {
                rv = value;
            }
        };
        RegisterAction<leaf_id_t, _, leaf_id_t>(idle_list) read_invalidate_idle_list = {
            void apply(inout bit<16> value, out bit<16> rv) {
                rv = value;
                value = INVALID_VALUE_16bit;
            }
        };


    Register<bit<16>, _>(MAX_VCLUSTERS) idle_count; // Stores idle count for each vcluster
        RegisterAction<bit<16>, _, bit<16>>(idle_count) read_idle_count = { 
            void apply(inout bit<16> value, out bit<16> rv) {
                rv = value; // Retruns val    
            }
        };
        RegisterAction<bit<16>, _, bit<16>>(idle_count) read_and_inc_idle_count = { 
            void apply(inout bit<16> value, out bit<16> rv) {
                rv = value; // Retruns val before modificaiton
                value = value + 1; 
            }
        };
        RegisterAction<bit<16>, _, bit<16>>(idle_count) read_and_dec_idle_count = { 
            void apply(inout bit<16> value, out bit<16> rv) {
                if (value > 0) { 
                    rv = value;
                    value = value - 1;
                }
            }
        };

    Register<queue_len_t, _>(MAX_TOTAL_LEAFS) queue_len_list_1; // List of queue lens for all vclusters
        RegisterAction<queue_len_t, _, queue_len_t>(queue_len_list_1) update_queue_len_list_1 = {
            void apply(inout queue_len_t value, out queue_len_t rv) {
                value = saqr_md.selected_ds_qlen;
                rv = value;
            }
        };
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


    Register<queue_len_t, _>(MAX_TOTAL_LEAFS) queue_len_list_2; // List of queue lens for all vclusters
        RegisterAction<queue_len_t, _, queue_len_t>(queue_len_list_2) update_queue_len_list_2 = {
            void apply(inout queue_len_t value, out queue_len_t rv) {
                value = saqr_md.selected_ds_qlen;
                rv = value;
            }
        };
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

    Register<queue_len_t, _>(MAX_TOTAL_LEAFS) deferred_queue_len_list_1; // List of queue lens for all vclusters
        RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_1) check_deferred_queue_len_list_1 = {
            void apply(inout queue_len_t value, out queue_len_t rv) {
                if (value <= saqr_md.queue_len_diff) { // Queue len drift is not large enough to invalidate the decision
                    value = value + saqr_md.selected_ds_qlen_unit;
                    rv = 0;
                } else {
                    rv = value; 
                }
            }
        };
         RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_1) reset_deferred_queue_len_list_1 = {
            void apply(inout queue_len_t value, out queue_len_t rv) {
                value = 0;
                rv = value;
            }
        };

    Register<queue_len_t, _>(MAX_TOTAL_LEAFS) deferred_queue_len_list_2; // List of queue lens for all vclusters
        RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_2) inc_deferred_queue_len_list_2 = {
            void apply(inout queue_len_t value, out queue_len_t rv) {
                value = value + saqr_md.selected_ds_qlen_unit;
                rv = value;
            }
        };
        RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_2) read_deferred_queue_len_list_2 = {
            void apply(inout queue_len_t value, out queue_len_t rv) {
                rv = value + saqr_md.not_selected_ds_qlen; // to avoid using another stage for this calculation
            }
        };
        RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_2) reset_deferred_queue_len_list_2 = {
            void apply(inout queue_len_t value, out queue_len_t rv) {
                value = 0;
                rv = value;
            }
        };

    Register<bit<16>, _>(MAX_TOTAL_LEAFS) idle_list_idx_mapping; // Maintains the position of leaf in the idle list so we can later remove it in O(1)
        RegisterAction<bit<16>, _, bit<16>>(idle_list_idx_mapping) write_idle_list_idx_mapping = {
            void apply(inout bit<16> value, out bit<16> rv) {
                value = saqr_md.idle_ds_index;
                rv = value;
            }
        };
        RegisterAction<bit<16>, _, bit<16>>(idle_list_idx_mapping) update_idle_list_idx_mapping = {
            void apply(inout bit<16> value, out bit<16> rv) {
                value = saqr_md.task_resub_hdr.ds_index_1;
                rv = value;
            }
        };
        RegisterAction<bit<16>, _, bit<16>>(idle_list_idx_mapping) read_idle_list_idx_mapping = { // Read last value update current to saqr_md.cluster_idle_count -1 (we swap with top element when removing)
            void apply(inout bit<16> value, out bit<16> rv) {
                rv = value;
            }
        };

    Register<bit<16>, _>(MAX_VCLUSTERS) rr_counter; // 
        RegisterAction<bit<16>, _, bit<16>>(rr_counter) inc_rr_counter  = {
            void apply(inout bit<16> value, out bit<16> rv) {
                rv = value;
                if (value >= (bit<16>) saqr_md.cluster_num_valid_queue_signals - 1) {
                    value = 0;
                } else {
                    value = value + 1;
                }
            }
        };

    // TESTBEDONLY: This register is for collecting statistics for overheads in our experiments 
    // Should be removed when measuring resource usage.
    Register<bit<32>, _>(MAX_VCLUSTERS) stat_count_resub; 
        RegisterAction<bit<32>, _, bit<32>>(stat_count_resub) inc_stat_count_resub  = {
            void apply(inout bit<32> value, out bit<32> rv) {
                value = value + 1;
            }
    };
 

    Register<bit<32>, _>(1) stat_count_task; 
        RegisterAction<bit<32>, _, bit<32>>(stat_count_task) inc_stat_count_task  = {
            void apply(inout bit<32> value, out bit<32> rv) {
                rv = value;
                value = value + 1;
            }
    };
     
    Register<bit<32>, _>(65536) ingress_tstamp; 
        RegisterAction<bit<32>, _, bit<32>>(ingress_tstamp) write_ingress_tstamp  = {
            void apply(inout bit<32> value, out bit<32> rv) {
                value = saqr_md.ingress_tstamp_clipped;
            }
    };
    /********  Action/table decelarations *********/
    action _drop() {
        ig_intr_dprsr_md.drop_ctl = 0x1; // Drop packet.
    }

    action get_leaf_start_idx () {
        saqr_md.cluster_ds_start_idx = (bit <16>) (hdr.saqr.cluster_id * MAX_LEAFS_PER_CLUSTER);
    }
    action get_array_indices () {
        saqr_md.idle_ds_index = saqr_md.cluster_ds_start_idx + saqr_md.cluster_idle_count;
        saqr_md.cluster_absolute_leaf_index = saqr_md.cluster_ds_start_idx + hdr.saqr.src_id;
    }

    action decrement_indices() {
        saqr_md.idle_ds_index = saqr_md.idle_ds_index - 1;
    }

    action select_idle_index() {
        // saqr_md.idle_ds_index = saqr_md.cluster_ds_start_idx + saqr_md.idle_rr_index;
        saqr_md.idle_ds_index = saqr_md.idle_ds_index - 1;
    }
    //////// 
    action gen_random_leaf_index_16() {
        saqr_md.t1_random_ds_index_1 = (bit<16>) random_ds_id.get();
        saqr_md.t1_random_ds_index_2 = (bit<16>) random_ds_id.get();
    }

    action retry_gen_random_leaf_index_16(){
        saqr_md.t2_random_ds_index_1 = (bit<16>) random_ds_id_retry.get();
        saqr_md.t2_random_ds_index_2 = (bit<16>) random_ds_id_retry.get();
    }

    action adjust_random_leaf_index_5() {
        saqr_md.t1_random_ds_index_1 = saqr_md.t1_random_ds_index_1 >> 11;
        saqr_md.t1_random_ds_index_2 = saqr_md.t1_random_ds_index_2 >> 11;
        saqr_md.t2_random_ds_index_1 = saqr_md.t2_random_ds_index_1 >> 11;
        saqr_md.t2_random_ds_index_2 = saqr_md.t2_random_ds_index_2 >> 11;
    }
    action adjust_random_leaf_index_4() {
        saqr_md.t1_random_ds_index_1 = saqr_md.t1_random_ds_index_1 >> 12;
        saqr_md.t1_random_ds_index_2 = saqr_md.t1_random_ds_index_2 >> 12;
        saqr_md.t2_random_ds_index_1 = saqr_md.t2_random_ds_index_1 >> 12;
        saqr_md.t2_random_ds_index_2 = saqr_md.t2_random_ds_index_2 >> 12;
    }

    action adjust_random_leaf_index_3() {
        saqr_md.t1_random_ds_index_1 = saqr_md.t1_random_ds_index_1 >> 13;
        saqr_md.t1_random_ds_index_2 = saqr_md.t1_random_ds_index_2 >> 13;
        saqr_md.t2_random_ds_index_1 = saqr_md.t2_random_ds_index_1 >> 13;
        saqr_md.t2_random_ds_index_2 = saqr_md.t2_random_ds_index_2 >> 13;
    }

    action adjust_random_leaf_index_2() {
        saqr_md.t1_random_ds_index_1 = saqr_md.t1_random_ds_index_1 >> 14;
        saqr_md.t1_random_ds_index_2 = saqr_md.t1_random_ds_index_2 >> 14;
        saqr_md.t2_random_ds_index_1 = saqr_md.t2_random_ds_index_1 >> 14;
        saqr_md.t2_random_ds_index_2 = saqr_md.t2_random_ds_index_2 >> 14;
    }

    action adjust_random_leaf_index_1() {
        saqr_md.t1_random_ds_index_1 = saqr_md.t1_random_ds_index_1 >> 15;
        saqr_md.t1_random_ds_index_2 = saqr_md.t1_random_ds_index_2 >> 15;
        saqr_md.t2_random_ds_index_1 = saqr_md.t2_random_ds_index_1 >> 15;
        saqr_md.t2_random_ds_index_2 = saqr_md.t2_random_ds_index_2 >> 15;
    }

    table adjust_random_range_sq_leafs { // Adjust the random generated number (16 bit) based on number of available queue len signals
        key = {
            saqr_md.cluster_num_valid_queue_signals: exact; 
        }
        actions = {
            adjust_random_leaf_index_5(); // #bits == 5 --> 0-31
            adjust_random_leaf_index_4(); // #bits == 4 --> 0-15
            adjust_random_leaf_index_3(); // #bits == 3 --> 0-7
            adjust_random_leaf_index_2(); // #bits == 2 --> 0-3
            adjust_random_leaf_index_1(); // #bits == 1 --> 0-1
            NoAction; // default 16 bits
        }
        size = 64;
        default_action = NoAction;
    }
   
    action act_forward_saqr(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port;
        /* 
         * TESTBEDONLY: The line below is only useful for our testbed experiments: 
         * We use cluster_id to isolate the *leaf* switches. 
         * Therefore, we set different cluster_id for each leaf on the outgoing packets from spine to the leaf. 
         * The destination leaf, will have a dedicated register space based on the cluster_id and will work as a seperate leaf scheduler.
        */
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
    
    action inc_repeated_rand() {
        saqr_md.random_ds_index_2 = saqr_md.random_ds_index_2+1;
    }

    action dec_repeated_rand() {
        saqr_md.random_ds_index_2 = saqr_md.random_ds_index_2-1;
    }

    action compare_random_idx() {
        saqr_md.t1_random_ds_index_1 = min(saqr_md.cluster_num_valid_queue_signals, saqr_md.t1_random_ds_index_1);
        saqr_md.t1_random_ds_index_2 = min(saqr_md.cluster_num_valid_queue_signals, saqr_md.t1_random_ds_index_2);
        saqr_md.t2_random_ds_index_1 = min(saqr_md.cluster_num_valid_queue_signals, saqr_md.t2_random_ds_index_1);
        saqr_md.t2_random_ds_index_2 = min(saqr_md.cluster_num_valid_queue_signals, saqr_md.t2_random_ds_index_2);
    }

    action compare_queue_len() {
        saqr_md.selected_ds_qlen = min(saqr_md.random_ds_qlen_1, saqr_md.random_ds_qlen_2);
    }
    action compare_correct_queue_len() {
        saqr_md.min_correct_qlen = min(saqr_md.task_resub_hdr.qlen_1, saqr_md.task_resub_hdr.qlen_2);
    }

    action calculate_queue_len_diff() {
        saqr_md.queue_len_diff = saqr_md.not_selected_ds_qlen - saqr_md.selected_ds_qlen;
    }
    action get_larger_queue_len() {
        saqr_md.not_selected_ds_qlen = max(saqr_md.random_ds_qlen_1, saqr_md.random_ds_qlen_2);
    }
    
    action compare_idle_index() {
        saqr_md.idle_remove_min_id = min(saqr_md.task_resub_hdr.ds_index_1, saqr_md.idle_ds_index);
    }

    // This gives us the 1/#workers for each leaf switch in each vcluster 
    action act_set_queue_len_unit_1(len_fixed_point_t cluster_unit) {
        saqr_md.qlen_unit_1 = cluster_unit;
    }
    table set_queue_len_unit_1 {
        key = {
            hdr.saqr.cluster_id: exact;
            saqr_md.random_id_1: exact;
        }
        actions = {
            act_set_queue_len_unit_1;
            NoAction;
        }
        size = HDR_CLUSTER_ID_SIZE;
        default_action = NoAction;
    }

    action act_set_queue_len_unit_2(len_fixed_point_t cluster_unit) {
        saqr_md.qlen_unit_2 = cluster_unit;
    }
    table set_queue_len_unit_2 {
        key = {
            hdr.saqr.cluster_id: exact;
            saqr_md.random_id_2: exact;
        }
        actions = {
            act_set_queue_len_unit_2;
            NoAction;
        }
        size = HDR_CLUSTER_ID_SIZE;
        default_action = NoAction;
    }
    
    action act_get_switch_index(switch_id_t switch_index) {
        saqr_md.child_switch_index = switch_index;
    }
    table get_switch_index {
        key = {
            hdr.saqr.cluster_id: exact;
            hdr.saqr.src_id: exact;
        }
        actions = {
            act_get_switch_index;
            NoAction;
        }
            size = HDR_CLUSTER_ID_SIZE;
            default_action = NoAction;
    }

    action act_get_rand_leaf_id_2 (bit <16> leaf_id){
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
    
    /********  Control block logic *********/
    apply {
        if (hdr.saqr.isValid()) {  // saqr packet
        if (hdr.saqr.dst_id == SWITCH_ID) { // If this packet is destined for this spine do saqr processing ot. its just an intransit packet we need to forward on correct port
            /* 
             * TESTBEDONLY: The line below is only useful for our testbed experiments: 
             * We use cluster_id to isolate the *leaf* switches. However from spine prespective, all leaves belong to the same vcluster. 
             * Therefore, we set cluster_id to 0 for every packet received by spine but we set different cluster_id for each leaf on the outgoing packets from spine to the leaf. 
            */
            hdr.saqr.cluster_id = 0;
            gen_random_leaf_index_16();

            @stage(1) {
                get_leaf_start_idx();
                retry_gen_random_leaf_index_16();
                get_cluster_num_valid_leafs.apply();
                if (ig_intr_md.resubmit_flag != 0) {
                    compare_correct_queue_len();
                    inc_stat_count_resub.execute(hdr.saqr.cluster_id);
                } 
                if (hdr.saqr.pkt_type == PKT_TYPE_IDLE_SIGNAL || ((hdr.saqr.pkt_type == PKT_TYPE_IDLE_REMOVE) && (saqr_md.task_resub_hdr.ds_index_1 == INVALID_VALUE_16bit))) {
                    saqr_md.cluster_idle_count = read_and_inc_idle_count.execute(hdr.saqr.cluster_id); // If it was IDLE_REMOVE and we are in second path and ds_index_1 is INVALID it means that another remove is in progress, roll back removal (increment pointer)
                } else if (hdr.saqr.pkt_type == PKT_TYPE_IDLE_REMOVE && ig_intr_md.resubmit_flag == 0) { // Only decrement idle count (pointer to idle_list) in first pass of removal
                    saqr_md.cluster_idle_count = read_and_dec_idle_count.execute(hdr.saqr.cluster_id);
                } else {
                    
                    if (hdr.saqr.pkt_type == PKT_TYPE_QUEUE_SIGNAL){
                        get_switch_index.apply(); // Get index of the switch in the qlen list
                    }
                    saqr_md.cluster_idle_count = read_idle_count.execute(hdr.saqr.cluster_id); // Get num_idle leafs (pointer to top of stack)
                }   
            }

            @stage(2) {
                if (ig_intr_md.resubmit_flag!=0) {
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK){

                        if (saqr_md.min_correct_qlen == saqr_md.task_resub_hdr.qlen_1) {
                            saqr_md.selected_ds_index = saqr_md.task_resub_hdr.ds_index_1;
                            saqr_md.selected_ds_qlen = saqr_md.task_resub_hdr.qlen_1;
                        } else {
                            saqr_md.selected_ds_index = saqr_md.task_resub_hdr.ds_index_2;
                            saqr_md.selected_ds_qlen = saqr_md.task_resub_hdr.qlen_2;
                        }
                    }
                } else {
                    
                    adjust_random_range_sq_leafs.apply(); //  We want to select a random node from available qlen signals 
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK){
                        saqr_md.task_counter = inc_stat_count_task.execute(0);
                    }
                }
                get_array_indices();
                saqr_md.ingress_tstamp_clipped = (bit<32>)ig_intr_md.ingress_mac_tstamp[31:0];
            }

            @stage(3) {
                if (ig_intr_md.resubmit_flag != 0){
                    if (hdr.saqr.pkt_type == PKT_TYPE_IDLE_REMOVE && saqr_md.task_resub_hdr.ds_index_1 != INVALID_VALUE_16bit) { // Second pass in remove process, update the position for the leaf that was top of idle list in previous pass (we moved it to the position for the leaf that is removed)
                        update_idle_list_idx_mapping.execute(saqr_md.task_resub_hdr.ds_index_2);
                    }
                } else {
                    compare_random_idx();
                    
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                        select_idle_index(); // 
                    }  else if (hdr.saqr.pkt_type == PKT_TYPE_IDLE_SIGNAL) {
                        write_idle_list_idx_mapping.execute(saqr_md.cluster_absolute_leaf_index);
                    } else if (hdr.saqr.pkt_type == PKT_TYPE_IDLE_REMOVE) { // First pass in remove process, find position of the to-be-removed leaf in idle list from mapping reg 
                        saqr_md.task_resub_hdr.ds_index_1 = read_idle_list_idx_mapping.execute(saqr_md.cluster_absolute_leaf_index);
                        decrement_indices(); // decrement the idle index so we read the correct idle leaf ID
                    } 
                }
            }

            @stage(4) {
                if (ig_intr_md.resubmit_flag != 0) {
                    if(hdr.saqr.pkt_type == PKT_TYPE_IDLE_REMOVE) { // Second pass for removing idle
                        if (saqr_md.task_resub_hdr.ds_index_1 != INVALID_VALUE_16bit){
                            write_idle_list.execute(saqr_md.task_resub_hdr.ds_index_1);
                            decrement_indices(); // decrement the idle index so we read the correct idle leaf ID
                        }
                        _drop();
                    } else if(hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                        saqr_md.random_ds_index_1 = saqr_md.selected_ds_index; // we already selected minimum set this so we can get the corresponding id of the selected index in random_id_1 meta field
                    }
                } else {
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                        saqr_md.idle_ds_id = read_idle_list.execute(saqr_md.idle_ds_index);
                        if (saqr_md.t1_random_ds_index_1 == saqr_md.cluster_num_valid_queue_signals) {  // generated index was out-of-range
                        if(saqr_md.t2_random_ds_index_1 == saqr_md.cluster_num_valid_queue_signals) {
                            saqr_md.random_ds_index_1 = inc_rr_counter.execute(hdr.saqr.cluster_id);   // Both tries out-of-range, use RR index sample instead of random
                        } else {
                            /* 
                             * Note: Had compilation error without this @in_hash pragma (compiler error not clear), most probably related to PHV allocation based on MAU group:
                             * Every metadata fields that are related (overall in the pipeline), will be in the same MAU group which has a limited K-bit (e.g., here 16 bit) PHVs
                             * This will tell the compiler to put the saqr_md.random_ds_index_1 in seperate MAU group 
                             * See: //https://community.intel.com/t5/Intel-Connectivity-Research/PHV-allocation-problem/m-p/1392964
                            */
                            @in_hash { 
                                saqr_md.random_ds_index_1 = saqr_md.t2_random_ds_index_1;    // Use second try (another random generated number)
                            }   
                        }
                    } else {
                        @in_hash {
                            saqr_md.random_ds_index_1 = saqr_md.t1_random_ds_index_1; // First try was fine
                        }
                    }
                    
                    }  else if (hdr.saqr.pkt_type == PKT_TYPE_IDLE_REMOVE){
                        compare_idle_index();
                        saqr_md.idle_ds_id = read_invalidate_idle_list.execute(saqr_md.idle_ds_index);
                    }
                    else if(hdr.saqr.pkt_type == PKT_TYPE_IDLE_SIGNAL) {
                        add_to_idle_list.execute(saqr_md.idle_ds_index);
                        _drop();
                    } 
                }
            }

            
            @stage(5) {
                if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                    if(saqr_md.cluster_idle_count == 0) {
                        if (saqr_md.t1_random_ds_index_2 == saqr_md.cluster_num_valid_queue_signals) {  // generated index was out-of-range
                            if(saqr_md.t2_random_ds_index_2 == saqr_md.cluster_num_valid_queue_signals) {
                                saqr_md.random_ds_index_2 = saqr_md.random_ds_index_1 >> 1;  // Both tries out-of-range, Divide other by two (arbitrary resolve function)
                            } else {
                                @in_hash {
                                    saqr_md.random_ds_index_2 = saqr_md.t2_random_ds_index_2; // Use second try (another random generated number)
                                }
                            }
                        } else {
                            @in_hash {
                                saqr_md.random_ds_index_2 = saqr_md.t1_random_ds_index_2;  // First try was fine  
                            }
                        }
                    } 
                } else if (hdr.saqr.pkt_type == PKT_TYPE_IDLE_REMOVE) { 
                    saqr_md.task_resub_hdr.ds_index_2 = saqr_md.idle_ds_id; // put last idle node in resub header so we write it on the pos of deleted idle node in resub path
                } 
            }

            @stage(6) {
                get_rand_leaf_id_1.apply(); // Read the leaf ID 1 associated with generated random index
                get_rand_leaf_id_2.apply(); // Read the leaf ID 2 associated with generated random index
                if (ig_intr_md.resubmit_flag != 0) {
                     if(hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                        update_queue_len_list_1.execute(saqr_md.selected_ds_index);
                        update_queue_len_list_2.execute(saqr_md.selected_ds_index);
                     }
                } else {
                    if(hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK && saqr_md.cluster_idle_count == 0) {
                        saqr_md.random_ds_qlen_1 = read_queue_len_list_1.execute(saqr_md.random_ds_index_1); // Read qlen for leafID1
                        saqr_md.random_ds_qlen_2 = read_queue_len_list_2.execute(saqr_md.random_ds_index_2); // Read qlen for leafID2
                    } else if (hdr.saqr.pkt_type == PKT_TYPE_QUEUE_SIGNAL) {
                        write_queue_len_list_1.execute(saqr_md.child_switch_index); // Write the qlen at corresponding index for the leaf in this cluster
                        write_queue_len_list_2.execute(saqr_md.child_switch_index); // Write the qlen at corresponding index for the leaf in this cluster
                    } else if (hdr.saqr.pkt_type == PKT_TYPE_IDLE_REMOVE) {
                        if (saqr_md.task_resub_hdr.ds_index_2 == INVALID_VALUE_16bit) {
                            saqr_md.task_resub_hdr.ds_index_1 = INVALID_VALUE_16bit;
                        }
                        ig_intr_dprsr_md.resubmit_type = RESUBMIT_TYPE_NEW_TASK; // Trigger resubmit for idle removal
                    }
                }
            }

            @stage(7) {
                compare_queue_len();
                get_larger_queue_len();
                if (ig_intr_md.resubmit_flag !=0) {
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK){
                        hdr.saqr.dst_id = saqr_md.random_id_1;
                    }
                } else {
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                    write_ingress_tstamp.execute(hdr.saqr.seq_num);
                    if (saqr_md.cluster_idle_count == 0) {
                        set_queue_len_unit_1.apply(); // Get 1/#workers in these two racks it is important since spine needs to keep track of the load *after* making a decision
                        set_queue_len_unit_2.apply();
                    }
                }
                }
            }

            @stage(8) {
                if (ig_intr_md.resubmit_flag == 0) {
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) { 
                        if (saqr_md.random_ds_index_1 == saqr_md.random_ds_index_2 || saqr_md.cluster_idle_count > 0) {
                                saqr_md.queue_len_diff = INVALID_VALUE_16bit; // No need to compare qlen diff with drift (in next stages) since rack was selected based on idle list
                        }
                        if (saqr_md.cluster_idle_count == 0 && saqr_md.cluster_num_valid_queue_signals > 0) {
                            if (saqr_md.random_ds_index_1 != saqr_md.random_ds_index_2){
                                calculate_queue_len_diff();    
                            }
                            if (saqr_md.selected_ds_qlen == saqr_md.random_ds_qlen_1) {
                                hdr.saqr.dst_id = saqr_md.random_id_1;
                                saqr_md.selected_ds_index = saqr_md.random_ds_index_1;
                                saqr_md.task_resub_hdr.ds_index_2 = saqr_md.random_ds_index_2;
                                saqr_md.selected_ds_qlen_unit = saqr_md.qlen_unit_1;
                                saqr_md.not_selected_ds_qlen_unit = saqr_md.qlen_unit_2;
                            } else {
                                hdr.saqr.dst_id = saqr_md.random_id_2;
                                saqr_md.selected_ds_index = saqr_md.random_ds_index_2;
                                saqr_md.task_resub_hdr.ds_index_2 = saqr_md.random_ds_index_1;
                                saqr_md.selected_ds_qlen_unit = saqr_md.qlen_unit_2;
                                saqr_md.not_selected_ds_qlen_unit = saqr_md.qlen_unit_1;
                            }
                        } else { // Idle selection
                            
                            saqr_md.selected_ds_qlen_unit = saqr_md.qlen_unit_1; // Put it selected_ds_qlen_unit so in next stages we increment the average load of this rack 
                            hdr.saqr.dst_id = saqr_md.idle_ds_id;
                            hdr.saqr.qlen = 1; // This is to indicate (to leaf) that spine's view is that leaf is idle and linked with this spine (so that if leaf is not idle anymore it will break the linkage)
                        }   
                    }
                }
            }

            @stage(9) {
                if (ig_intr_md.resubmit_flag != 0) {
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                        reset_deferred_queue_len_list_1.execute(saqr_md.selected_ds_index); // Just updated the queue_len_list so write 0 on deferred reg
                    }
                } else {
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK && saqr_md.cluster_idle_count == 0) {
                        
                        saqr_md.deferred_qlen_1 = check_deferred_queue_len_list_1.execute(saqr_md.selected_ds_index); // Returns Deferred[selected_index]
                        saqr_md.task_resub_hdr.ds_index_1 = saqr_md.selected_ds_index; // We keep the index of the node that initially selected in task_resub_hdr.ds_index_1
                         // Add (1/#workers) to the average load of selected rack (if selected this will be the new average load), 
                         // this is used in resubmission path for selecting the correct node
                        saqr_md.selected_ds_qlen = saqr_md.selected_ds_qlen + saqr_md.selected_ds_qlen_unit;
                        // Same as above but for the *other* node that initially (based on qlen list) had higher load
                        saqr_md.not_selected_ds_qlen = saqr_md.not_selected_ds_qlen + saqr_md.not_selected_ds_qlen_unit;
                    } else if (hdr.saqr.pkt_type == PKT_TYPE_QUEUE_SIGNAL) {

                        reset_deferred_queue_len_list_1.execute(saqr_md.child_switch_index); // Just updated the queue_len_list so write 0 on deferred reg
                    }   
                }
            }

            @stage(10) {
                if (ig_intr_md.resubmit_flag != 0) {
                    if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                        reset_deferred_queue_len_list_2.execute(saqr_md.selected_ds_index); // Just updated the queue_len_list so write 0 on deferred reg
                    }
                } else {
                    if (hdr.saqr.pkt_type==PKT_TYPE_QUEUE_SIGNAL){
                        reset_deferred_queue_len_list_2.execute(saqr_md.child_switch_index); // Just updated the queue_len_list so write 0 on deferred reg
                        _drop();
                    } else if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK && saqr_md.cluster_idle_count == 0) {
                        saqr_md.task_resub_hdr.qlen_1 = saqr_md.deferred_qlen_1 + saqr_md.selected_ds_qlen;
                        if(saqr_md.deferred_qlen_1 == 0) { // This return value means that we do not need to check deffered qlens, difference between samples were large enough that our decision is still valid
                            inc_deferred_queue_len_list_2.execute(saqr_md.selected_ds_index); // increment the second copy to be consistent with first one
                        } else { // This means our decision might be invalid, need to check the deffered queue lens and resubmit
                            ig_intr_dprsr_md.resubmit_type = RESUBMIT_TYPE_NEW_TASK;
                            saqr_md.task_resub_hdr.qlen_2 = read_deferred_queue_len_list_2.execute(saqr_md.task_resub_hdr.ds_index_2);
                        }
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
         
    Resubmit() resubmit;

    apply {  
        if (ig_intr_dprsr_md.resubmit_type == RESUBMIT_TYPE_NEW_TASK) {
            resubmit.emit(saqr_md.task_resub_hdr);
        } 

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
        pkt.extract(hdr.ethernet);
        pkt.extract(hdr.ipv4);
        pkt.extract(hdr.udp);
        pkt.extract(hdr.saqr);
        transition accept;
    }
}

control SpineEgressDeparser(
        packet_out pkt,
        inout saqr_header_t hdr,
        in eg_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.saqr);
    }
}

control SpineEgress(
        inout saqr_header_t hdr,
        inout eg_metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
    Register<bit<32>, _>(1) stat_count_task; 
    RegisterAction<bit<32>, _, bit<32>>(stat_count_task) inc_stat_count_task  = {
        void apply(inout bit<32> value, out bit<32> rv) {
            rv = value;
            value = value + 1;
        }
    };
    Register<bit<32>, _>(65536) egress_tstamp; 
        RegisterAction<bit<32>, _, bit<32>>(egress_tstamp) write_egress_tstamp  = {
            void apply(inout bit<32> value, out bit<32> rv) {
                value = eg_md.egress_tstamp_clipped;
            }
    };
    apply {
        if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
            eg_md.task_counter = inc_stat_count_task.execute(0);
            eg_md.egress_tstamp_clipped = (bit<32>)eg_intr_md_from_prsr.global_tstamp[31:0];
            write_egress_tstamp.execute(hdr.saqr.seq_num);
        }
    }
}