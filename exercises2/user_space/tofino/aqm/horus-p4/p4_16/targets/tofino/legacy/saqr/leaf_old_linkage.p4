#include <core.p4>
#include <tna.p4>

#include "../common/headers.p4"
#include "../common/util.p4"
#include "../headers.p4"


// TODO: Remove linked spine iq when new task comes and idlecount is 1
/*
 * Note: Difference with simulations
 * In python when taking samples from qlen lists, we just used random.sample() and took K *distinct* samples 
 * In hardware it is not possible to ensure that these values are distinct and two samples might point to same position
 * TODO: Modify the python simulations to reflect this (This is also same for racksched system might affect their performance)
 *
 * Notes:
 *  An action can be called directly without a table (from apply{} block)
 *  Here multiple calls to action from the apply{} block is allowed (e.g in different if-else branches)
 *  Limitations: 
 *    If multiple operations (simple arith +,-,...) done in a single action results in error. "Action Require multiple stages for 
 *    a single action. We currently support only single stage actions."
 * 
 *    Multiple operations in a single branch of apply{} block not allowed. The operations must be done in seperate actions (That 
 *     translates to parallel ALU blocks in hardware?)
 *    Index of reg (passed to .execute()) can not be computed in the same block of apply{}. But index can be computed in an action. 
 *    
 *   (As far as we know) Multiple accesses to the same register is not allowed. To overcome the restriction define the RegActions in a way that we can handle the different conditions inside them.
 *   
 *   Random generator only accepts constant upper bound. This causes problem for when we have different number of workers in the rack 
 *    and to select from them we need the random number to be in that specific range. 
 *
 *   Only one RegisterAction may be executed per packet for a given Register. This is a significant limitation for our algorithm.
 *   Switch can not read n random registers and then increment the selected worker's register after comparison. We need to rely on worker to update the qlen later. 
 *   
 *   Comparing two metadeta feilds (with < >) in apply{} blcok resulted in error. (Too complex). Only can use == on two meta feilds!
*/


control LeafIngress(
        inout saqr_header_t hdr,
        inout saqr_metadata_t saqr_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

            /* *** Register definitions *** */ 
            // TODO: Check Reg definition is _ correct?
            // idle list should be allocated to store MAX_VCLUSTER_RACK * MAX_IDLES_RACK
            Register<worker_id_t, _>(MAX_WORKERS_IN_RACK) idle_list;
                RegisterAction<bit<16>, _, bit<16>>(idle_list) add_to_idle_list = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        value = hdr.saqr.src_id;
                        rv = value;
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(idle_list) read_idle_list = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                    }
                };

            Register<bit<8>, _>(MAX_VCLUSTERS) idle_count; // Stores idle count for each vcluster
                RegisterAction<bit<8>, _, bit<8>>(idle_count) read_idle_count = { 
                    void apply(inout bit<8> value, out bit<8> rv) {
                        rv = value; // Retruns val    
                    }
                };
                
                /* TODO: Compare current value to avoid overflow
                 * Tofino Bug(?): Comparing value > MAX_VALUE(0xffff) returns always false!
                 * Possible Reason: https://community.intel.com/t5/Intel-Connectivity-Research/Modify-a-register-based-on-a-single-bit-of-its-current-value/m-p/1258877
                 * Comparison in SALU is converted to comparison with 0!
                 * e.g value < 0xffff translates to value - 0xffff < 0 which for unsigned bits results in issue!
                */ 
                RegisterAction<bit<8>, _, bit<8>>(idle_count) read_and_inc_idle_count = { 
                    void apply(inout bit<8> value, out bit<8> rv) {
                        rv = value; // Retruns val before modificaiton
                        value = value + 1; 
                    }
                };
                RegisterAction<bit<8>, _, bit<8>>(idle_count) read_and_dec_idle_count = { 
                    void apply(inout bit<8> value, out bit<8> rv) {
                        if (value > 0) { 
                            rv = value;
                            value = value - 1;
                        }
                    }
                };
            
            Register<queue_len_t, _>(MAX_WORKERS_IN_RACK) queue_len_list_1; // List of queue lens for all vclusters
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


            Register<queue_len_t, _>(MAX_WORKERS_IN_RACK) queue_len_list_2; // List of queue lens for all vclusters
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

            Register<queue_len_t, _>(MAX_WORKERS_IN_RACK) deferred_queue_len_list_1; // List of queue lens for all vclusters
                RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_1) check_deferred_queue_len_list_1 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        if (value <= saqr_md.queue_len_diff) { // Queue len drift is not large enough to invalidate the decision
                            value = value + 1;
                            rv = 0;
                        } else {
                            rv = value + saqr_md.selected_ds_qlen; // to avoid using another stage for this calculation
                        }
                    }
                };
                 RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_1) reset_deferred_queue_len_list_1 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        value = 0;
                        rv = value;
                    }
                };
                RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_1) inc_deferred_queue_len_list_1 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                            value = value + 1;
                    }
                };

            Register<queue_len_t, _>(MAX_WORKERS_IN_RACK) deferred_queue_len_list_2; // List of queue lens for all vclusters
                RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_2) inc_deferred_queue_len_list_2 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        value = value + 1;
                        rv = value;
                    }
                };
                RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_2) read_deferred_queue_len_list_2 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        rv = value + saqr_md.not_selected_ds_qlen;
                    }
                };
                 RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_2) reset_deferred_queue_len_list_2 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        value = 0;
                        rv = value;
                    }
                };

            Register<queue_len_t, _>(MAX_VCLUSTERS) aggregate_queue_len_list; // One for each vcluster
                RegisterAction<bit<8>, _, bit<8>>(aggregate_queue_len_list) update_read_aggregate_queue_len = {
                    void apply(inout bit<8> value, out bit<8> rv) {
                        if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                            value = value + saqr_md.queue_len_unit;
                        } else {
                            value = value - saqr_md.queue_len_unit;
                        }
                        rv = value;
                    }
                };
                RegisterAction<bit<8>, _, bit<8>>(aggregate_queue_len_list) read_aggregate_queue_len = {
                    void apply(inout bit<8> value, out bit<8> rv) {
                        rv = value;
                    }
                };
            
            Register<switch_id_t, _>(MAX_VCLUSTERS) linked_iq_sched; // Spine that ToR has sent last IdleSignal (1 for each vcluster).
                RegisterAction<bit<16>, _, bit<16>>(linked_iq_sched) read_reset_linked_iq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                        value = 0xFFFF;
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(linked_iq_sched) write_linked_iq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                        value = saqr_md.spine_to_link_iq;
                    }
                };
            

            Register<switch_id_t, _>(MAX_VCLUSTERS) linked_sq_sched; // Spine that ToR has sent last QueueSignal (1 for each vcluster).
                RegisterAction<bit<16>, _, bit<16>>(linked_sq_sched) read_update_linked_sq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                        if (value == INVALID_VALUE_16bit && hdr.saqr.pkt_type == PKT_TYPE_SCAN_QUEUE_SIGNAL) { // Not linked before and new SCAN request arrived
                            value = hdr.saqr.src_id;
                        }
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(linked_sq_sched) remove_linked_sq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        value = INVALID_VALUE_16bit;
                        rv = value;
                    }
                };
            Register<bit<16>, _>(MAX_VCLUSTERS) linked_view_drift; // Spine that ToR has sent last QueueSignal (1 for each vcluster).
                RegisterAction<bit<16>, _, bit<16>>(linked_view_drift) inc_read_linked_view_drift  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        if (value == saqr_md.cluster_num_valid_ds - 1) {
                            rv = 0;
                            value = 0;
                        } else {
                            rv = 1;
                            value = value + 1;
                        }
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(linked_view_drift) reset_linked_view_drift  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        value = 0;
                    }
                };

            // Below are registers to hold state in middle of probing Idle list proceess. 
            // So we can compare them when second switch responds.
            // IMPORTANT: Tofino Bug: comparison with largest value (e.g value==0xFF) in stateful ALU returns false even when value is 0xFF!
            // However comparison with 0 works fine! Maybe this is a bug in *tofino model* only. But no posts in community forums stated such bug! 
            // TODO: Report to community
            Register<queue_len_t, _>(MAX_VCLUSTERS) spine_iq_len_1; // Length of Idle list for first probed spine (1 for each vcluster).
                RegisterAction<queue_len_t, _, queue_len_t>(spine_iq_len_1) read_update_spine_iq_len_1  = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        rv = value;
                        if (value == INVALID_VALUE_8bit) { // Value==INVALID, So this is the first probe and we store the data
                            value = hdr.saqr.qlen;
                        } else { // Value found so this is the second probe and we load the data
                            value = INVALID_VALUE_8bit;
                        }
                    }
                };
                Register<bit<16>, _>(MAX_VCLUSTERS) spine_probed_id; // ID of the first probed spine (1 for each vcluster)
                RegisterAction<bit<16>, _, bit<16>>(spine_probed_id) read_update_spine_probed_id  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                        if (value == INVALID_VALUE_16bit) { // Value==INVALID, So this is the first probe and we store the data.
                            value = hdr.saqr.src_id;
                        } else { // Value was valid so this is the second probe and we load the data
                            value = INVALID_VALUE_16bit;
                        }
                    }
                };
            
            /* 
              As a workaround since we can't select the random range in runtime. 
              We get() a random variables and shift it depending on number of workers in rack.
              TODO: This enforces the num workers in rack to be pow of 2. Also, is this biased random?
            */
            Random<bit<16>>() random_worker_id_16;

            action get_worker_start_idx () {
                saqr_md.cluster_ds_start_idx = (bit <16>) (hdr.saqr.cluster_id * MAX_WORKERS_PER_CLUSTER);
            }

            // Calculates the index of next idle worker in idle_list array.
            action get_idle_index () {
                saqr_md.idle_ds_index = saqr_md.cluster_ds_start_idx + (bit <16>) saqr_md.cluster_idle_count;
            }

            action get_curr_idle_index() {
                saqr_md.idle_ds_index = saqr_md.idle_ds_index -1;
            }

            // action get_worker_index () {
            //     saqr_md.worker_index = (bit<16>) hdr.saqr.src_id + saqr_md.cluster_worker_start_idx;
            // }

            action _drop() {
                ig_intr_dprsr_md.drop_ctl = 0x1; // Drop packet.
            }

            action act_set_queue_len_unit(len_fixed_point_t cluster_unit) {
                saqr_md.queue_len_unit = cluster_unit;
            }
            table set_queue_len_unit {
                key = {
                    hdr.saqr.cluster_id: exact;
                }
                actions = {
                    act_set_queue_len_unit;
                    NoAction;
                }
                    size = HDR_CLUSTER_ID_SIZE;
                    default_action = NoAction;
            }

            action gen_random_probe_group() { // For probing two out of n spine schedulers
                ig_intr_tm_md.mcast_grp_a = (MulticastGroupId_t) 1; // Assume all use a single grp level 1
                /* 
                  Limitation: Casting the output of Random instance and assiging it directly to mcast_grp_b did not work. 
                  Had to assign it to a 16 bit meta field and then assign to mcast_group. 
                */
                // Different out ports for level 2 randomly generated
                // Here we use the same random 16 bit number generated for downstream ID to save resources
                ig_intr_tm_md.mcast_grp_b = saqr_md.random_id_1; 
            }
        
            // action set_mirror_type_worker_response() {
            //     ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE;
            // }

            action act_forward_saqr(PortId_t port, mac_addr_t dst_mac) {
                ig_intr_tm_md.ucast_egress_port = port;
                hdr.ethernet.dst_addr = dst_mac;
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

            action act_get_cluster_num_valid(bit<16> num_ds_elements, bit<16> num_us_elements) {
                saqr_md.cluster_num_valid_ds = num_ds_elements;
                saqr_md.cluster_num_valid_us = num_us_elements;
            }
            table get_cluster_num_valid {
                key = {
                    hdr.saqr.cluster_id : exact;
                }
                actions = {
                    act_get_cluster_num_valid;
                    NoAction;
                }
                size = HDR_CLUSTER_ID_SIZE;
                default_action = NoAction;
            }

            action gen_random_workers_16() {
                saqr_md.random_id_1 = (bit<16>) random_worker_id_16.get();
                saqr_md.random_id_2 = (bit<16>) random_worker_id_16.get();
            }
            
            action adjust_random_worker_range_8() {
                saqr_md.random_id_1 = saqr_md.random_id_1 >> 8;
                saqr_md.random_id_2 = saqr_md.random_id_2 >> 8;
            }

            action adjust_random_worker_range_4() {
                saqr_md.random_id_1 = saqr_md.random_id_1 >> 12;
                saqr_md.random_id_2 = saqr_md.random_id_2 >> 12;
            }

            action adjust_random_worker_range_2() {
                saqr_md.random_id_1 = saqr_md.random_id_1 >> 14;
                saqr_md.random_id_2 = saqr_md.random_id_2 >> 14;
            }

            action adjust_random_worker_range_1() {
                saqr_md.random_id_1 = saqr_md.random_id_1 >> 15;
                saqr_md.random_id_2 = saqr_md.random_id_2 >> 15;
            }

            table adjust_random_range_ds { // Reduce the random generated number (16 bit) based on number of workers in rack
                key = {
                    saqr_md.cluster_num_valid_ds: exact; 
                }
                actions = {
                    adjust_random_worker_range_8(); // == 8
                    adjust_random_worker_range_4(); // == 4
                    adjust_random_worker_range_2(); // == 2
                    adjust_random_worker_range_1(); // == 1
                    NoAction; // == 16
                }
                size = 16;
                default_action = NoAction;
            }
            
            table adjust_random_range_us { // Reduce the random generated number (16 bit) based on number of workers in rack
                key = {
                    saqr_md.cluster_num_valid_us: exact; 
                }
                actions = {
                    adjust_random_worker_range_8(); // == 8
                    adjust_random_worker_range_4(); // == 4
                    adjust_random_worker_range_2(); // == 2
                    adjust_random_worker_range_1(); // == 1
                    NoAction; // == 16
                }
                size = 16;
                default_action = NoAction;
            }

            action act_get_spine_dst_id(bit <16> spine_dst_id){
                hdr.saqr.dst_id = spine_dst_id;
            }
            table get_spine_dst_id {
                key = {
                    saqr_md.random_id_1: exact;
                }
                actions = {
                    act_get_spine_dst_id();
                    NoAction;
                }
                size = 16;
                default_action = NoAction;
            }
            action offset_random_ids() {
                saqr_md.random_id_1 = saqr_md.random_id_1 + saqr_md.cluster_ds_start_idx;
                saqr_md.random_id_2 = saqr_md.random_id_2 + saqr_md.cluster_ds_start_idx;
            }

            action compare_queue_len() {
                saqr_md.selected_ds_qlen = min(saqr_md.random_ds_qlen_1, saqr_md.random_ds_qlen_2);
            }
            action compare_correct_queue_len() {
                saqr_md.min_correct_qlen = min(saqr_md.task_resub_hdr.qlen_1, saqr_md.task_resub_hdr.qlen_2);
            }
            action get_larger_queue_len() {
                saqr_md.not_selected_ds_qlen = max(saqr_md.random_ds_qlen_1, saqr_md.random_ds_qlen_2);
            }
            
            action calculate_queue_len_diff(){
                saqr_md.queue_len_diff = saqr_md.not_selected_ds_qlen - saqr_md.selected_ds_qlen;
            }

            action compare_spine_iq_len() {
                saqr_md.selected_spine_iq_len = min(saqr_md.last_iq_len, hdr.saqr.qlen);
            }

            

            // action convert_pkt_to_sq_signal() {
                // hdr.saqr.pkt_type = PKT_TYPE_QUEUE_SIGNAL; // This should be set in last stage because effects other stages (if conditions)
                // hdr.saqr.qlen = saqr_md.aggregate_queue_len;
            // }
            
            apply {
                if (hdr.saqr.isValid()) {  // saqr packet
                    if (ig_intr_md.resubmit_flag != 0) { // Special case: packet is resubmitted just update the indexes
                        @stage(0){
                            compare_correct_queue_len();
                        }
                        @stage(1){
                            if (saqr_md.min_correct_qlen == saqr_md.task_resub_hdr.qlen_1) {
                                hdr.saqr.dst_id = saqr_md.task_resub_hdr.ds_index_1;
                                saqr_md.selected_ds_qlen = saqr_md.task_resub_hdr.qlen_1 + 1;
                            } else {
                                hdr.saqr.dst_id = saqr_md.task_resub_hdr.ds_index_2;
                                saqr_md.selected_ds_qlen = saqr_md.task_resub_hdr.qlen_2 + 1;
                            }
                        }
                        @stage(4) {
                            update_queue_len_list_1.execute(hdr.saqr.dst_id);
                            update_queue_len_list_2.execute(hdr.saqr.dst_id);
                        }
                        @stage(7) {
                            reset_deferred_queue_len_list_1.execute(hdr.saqr.dst_id); // Just updated the queue_len_list so write 0 on deferred reg
                        }
                        @stage(8) {
                            reset_deferred_queue_len_list_2.execute(hdr.saqr.dst_id);
                        }
                    } else {
                        /**Stage 0
                         * get_worker_start_idx
                         * queue_len_unit
                         Registers:
                         * idle_count
                         * linked_sq_sched
                        */
                        get_worker_start_idx(); // Get start index of workers for this vcluster
                        set_queue_len_unit.apply();
                        if (hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE) {
                                // @st st_idle_count = 0
                            saqr_md.cluster_idle_count = read_and_inc_idle_count.execute(hdr.saqr.cluster_id);
                        } else if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                            saqr_md.cluster_idle_count = read_and_dec_idle_count.execute(hdr.saqr.cluster_id); // Read last idle count for vcluster
                        } else if (hdr.saqr.pkt_type == PKT_TYPE_PROBE_IDLE_RESPONSE) {
                            saqr_md.cluster_idle_count = read_idle_count.execute(hdr.saqr.cluster_id); // Read last idle count for vcluster
                        }

                        if (hdr.saqr.pkt_type == PKT_TYPE_QUEUE_REMOVE) {
                            remove_linked_sq.execute(hdr.saqr.cluster_id); 
                        } else {
                            saqr_md.linked_sq_id = read_update_linked_sq.execute(hdr.saqr.cluster_id); // Get ID of the Spine that the leaf reports to   

                        }

                        /**Stage 1
                         * get_idle_index, dep: get_worker_start_idx @st0, idle_count @st 0
                         * get_cluster_num_valid
                         * gen_random_workers_16
                         Registers:
                         * aggregate_queue_len, dep: queue_len_unit @st0
                         * iq_len_1
                         * probed_id
                        */
                        // INFO: Compiler bug, if calculate the index for reg action here, compiler complains but if in action its okay!
                        @stage(1){
                            get_idle_index();
                            get_cluster_num_valid.apply();
                            gen_random_workers_16();
                            if (hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE || hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                                //@st st_aggregate_queue = 0
                                saqr_md.aggregate_queue_len = update_read_aggregate_queue_len.execute(hdr.saqr.cluster_id);
                                saqr_md.spine_view_ok = inc_read_linked_view_drift.execute(hdr.saqr.cluster_id);
                            } else if(hdr.saqr.pkt_type == PKT_TYPE_PROBE_IDLE_RESPONSE) {
                                saqr_md.last_iq_len = read_update_spine_iq_len_1.execute(hdr.saqr.cluster_id);
                                saqr_md.last_probed_id = read_update_spine_probed_id.execute(hdr.saqr.cluster_id);
                            } else if (hdr.saqr.pkt_type == PKT_TYPE_SCAN_QUEUE_SIGNAL) {
                                reset_linked_view_drift.execute(hdr.saqr.cluster_id);
                            }
                        }
                        
                        /** Stage 2
                         * get_curr_idle_index, dep: get_idle_index() @st1
                         * compare_spine_iq_len
                         * adjust_random_range
                         Registers:
                         * All of the worker qlen related regs, deps: resource limit of prev stage  
                        */
                        @stage(2) {
                            saqr_md.mirror_dst_id = hdr.saqr.dst_id; // We want the original packet to reach its destination
                            
                            if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK){
                                get_curr_idle_index(); // decrement the index so we read the correct idle worker id
                                adjust_random_range_ds.apply(); // move the random indexes to be in range of num workers in rack
                            } else {
                                adjust_random_range_us.apply();
                            }

                            if (hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE) {
                                if (saqr_md.linked_sq_id != INVALID_VALUE_16bit && saqr_md.spine_view_ok == 0) { // Need to send a new load signal to spine
                                        /* 
                                        Desired behaviour: Mirror premitive (emit invoked in ingrdeparser) will send the original response
                                        Here we modify the original packet and send it as a ctrl pkt to the linked spine.
                                        TODO: Might not work as we expect.
                                        */
                                        hdr.saqr.pkt_type = PKT_TYPE_QUEUE_SIGNAL; // This should be set in last stage because effects other stages (if conditions)
                                        hdr.saqr.qlen = saqr_md.aggregate_queue_len;
                                        hdr.saqr.dst_id = saqr_md.linked_sq_id;
                                        ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE; 
                                }
                            } else if (hdr.saqr.pkt_type == PKT_TYPE_SCAN_QUEUE_SIGNAL) {
                                if (saqr_md.linked_sq_id == INVALID_VALUE_16bit) { // Not linked to another spine so we reply back with queue signal
                                    hdr.saqr.pkt_type = PKT_TYPE_QUEUE_SIGNAL_INIT; // This should be set in last stage because effects other stages (if conditions)
                                    hdr.saqr.qlen = saqr_md.aggregate_queue_len;
                                    hdr.saqr.dst_id = hdr.saqr.src_id;
                                }
                            } else if (hdr.saqr.pkt_type == PKT_TYPE_PROBE_IDLE_RESPONSE){
                                compare_spine_iq_len();   
                            }

                        } 
                        /** Stage 3
                         * Register:
                         * idle_list, dep: idle_count @st0, get_idle_index() @st 1, get_curr_idle_index() @st 2
                        */ 
                        @stage(3) {
                            if (hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE) { 
                                add_to_idle_list.execute(saqr_md.idle_ds_index);

                            } else if(hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                                if (saqr_md.cluster_idle_count > 0) {
                                    saqr_md.idle_ds_id = read_idle_list.execute(saqr_md.idle_ds_index);
                                } else {
                                    offset_random_ids();
                                }
                            } else if (hdr.saqr.pkt_type == PKT_TYPE_PROBE_IDLE_RESPONSE) {
                                if (saqr_md.last_probed_id != INVALID_VALUE_16bit) { // This is the second probe response
                                    if (saqr_md.selected_spine_iq_len == saqr_md.last_iq_len) { // last spine selected
                                        saqr_md.spine_to_link_iq = saqr_md.last_probed_id;
                                        hdr.saqr.dst_id = saqr_md.last_probed_id;
                                    } else {
                                        saqr_md.spine_to_link_iq = hdr.saqr.src_id;
                                        hdr.saqr.dst_id = hdr.saqr.src_id;
                                    }
                                }
                            }
                        } 
                        
                        /** Stage 4
                         * 
                        */
                        @stage(4) {
                            if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK && saqr_md.cluster_idle_count == 0) {
                                saqr_md.random_ds_qlen_1 = read_queue_len_list_1.execute(saqr_md.random_id_1);
                                saqr_md.random_ds_qlen_2 = read_queue_len_list_2.execute(saqr_md.random_id_2);
                            } else if(hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE || hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE) {
                                write_queue_len_list_1.execute(hdr.saqr.src_id);
                                write_queue_len_list_2.execute(hdr.saqr.src_id);
                            } 
                            
                            // if (hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE) {
                            //     if (saqr_md.linked_sq_id != INVALID_VALUE_16bit) {
                                    
                            //         // Set different mirror types for different headers if needed
                            //         ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE; 
                            //     }
                            // }  
                            if (hdr.saqr.pkt_type == PKT_TYPE_PROBE_IDLE_RESPONSE || (hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE && saqr_md.cluster_idle_count == 0)) { // Only update linkage if this is the leaf have just became idle
                                if (saqr_md.last_probed_id != INVALID_VALUE_16bit) { // This is the second probe response
                                    write_linked_iq.execute(hdr.saqr.cluster_id);
                                    hdr.saqr.pkt_type = PKT_TYPE_IDLE_SIGNAL; // Now change to idle signal to notify the selected spine
                                } else {
                                    if(hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE){
                                        ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE; 
                                    }
                                    hdr.saqr.pkt_type = PKT_TYPE_PROBE_IDLE_QUEUE; // Change packet type to probe
                                    get_spine_dst_id.apply();
                                }
                            }
                        }

                        /** Stage 5
                         * 
                        */
                        @stage(5){
                        // packet is resubmitted
                        
                        if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK && saqr_md.cluster_idle_count == 0) {
                            compare_queue_len();
                            get_larger_queue_len();
                        }
                        
                        }

                        /* Stage 6
                         *
                        */
                        @stage(6) {
                            // packet is in first pass
                            calculate_queue_len_diff();
                            if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                                if (saqr_md.cluster_idle_count == 0) {
                                    if (saqr_md.selected_ds_qlen == saqr_md.random_ds_qlen_1) {
                                        hdr.saqr.dst_id = saqr_md.random_id_1;
                                        saqr_md.task_resub_hdr.ds_index_2 = saqr_md.random_id_2;
                                    } else {
                                        hdr.saqr.dst_id = saqr_md.random_id_2;
                                        saqr_md.task_resub_hdr.ds_index_2 = saqr_md.random_id_1;
                                    }
                                } else {
                                    hdr.saqr.dst_id = saqr_md.idle_ds_id;
                                }
                            } 
                        }
                        @stage(7) {
                            if (hdr.saqr.pkt_type==PKT_TYPE_TASK_DONE_IDLE || hdr.saqr.pkt_type==PKT_TYPE_TASK_DONE){
                                reset_deferred_queue_len_list_1.execute(hdr.saqr.src_id); // Just updated the queue_len_list so write 0 on deferred reg
                            } else if(hdr.saqr.pkt_type==PKT_TYPE_NEW_TASK) { //TODO: check this if condition!
                                if (saqr_md.random_id_2 != saqr_md.random_id_1) {
                                    saqr_md.task_resub_hdr.qlen_1 = check_deferred_queue_len_list_1.execute(hdr.saqr.dst_id); // Returns QL[dst_id] + Deferred[dst_id]
                                    saqr_md.task_resub_hdr.ds_index_1 = hdr.saqr.dst_id;
                                } else { // In case two samples point to the same cell, we do not need to resubmit just increment deferred list
                                    inc_deferred_queue_len_list_1.execute(hdr.saqr.dst_id);
                                }
                            }
                        }
                        @stage(8){
                            if (hdr.saqr.pkt_type==PKT_TYPE_TASK_DONE_IDLE || hdr.saqr.pkt_type==PKT_TYPE_TASK_DONE){
                                reset_deferred_queue_len_list_2.execute(hdr.saqr.src_id); // Just updated the queue_len_list so write 0 on deferred reg
                            } else if(hdr.saqr.pkt_type==PKT_TYPE_NEW_TASK) {//TODO: check this if condition!
                                if(saqr_md.task_resub_hdr.qlen_1 == 0) { // This return value means that we do not need to check deffered qlens, difference between samples were large enough that our decision is still valid
                                    inc_deferred_queue_len_list_2.execute(hdr.saqr.dst_id); // increment the second copy to be consistent with first one
                                } else { // This means our decision might be invalid, need to check the deffered queue lens and resubmit
                                    ig_intr_dprsr_md.resubmit_type = RESUBMIT_TYPE_NEW_TASK;
                                    saqr_md.task_resub_hdr.qlen_2 = read_deferred_queue_len_list_2.execute(saqr_md.task_resub_hdr.ds_index_2);
                                }
                            }
                        }
                    }
                    forward_saqr_switch_dst.apply();
                    
                }  else if (hdr.ipv4.isValid()) { // Regular switching procedure
                    // TODO: Not ported the ip matching tables for now, do we need them?
                    _drop();
                } else {
                    _drop();
                }
            }
        }

control LeafIngressDeparser(
        packet_out pkt,
        inout saqr_header_t hdr,
        in saqr_metadata_t saqr_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {
         
    Mirror() mirror;
    Resubmit() resubmit;

    apply {
        if (ig_intr_dprsr_md.mirror_type == MIRROR_TYPE_WORKER_RESPONSE) {
            
            /* 
             See page 58: P4_16 Tofino Native Architecture
             Application Note – (Public Version Mar 2021)
             In summary: this should replicate the initial received packet *Before any modifications* to the configured ports.
             Here we are using the dst_id as mirror Session ID
             Control plane needs to add mapping between session ID (we use dst_id aas key) and 
             output port (value) (same table as saqr forward in ingress)
            */
            // TODO: Bug Report to community. emit() should support single param interface when no header is needed. But gets compiler internal error! 
            mirror.emit<empty_t>((MirrorId_t) saqr_md.mirror_dst_id, {}); 
        }  else if (ig_intr_dprsr_md.mirror_type == MIRROR_TYPE_NEW_TASK) {

        }
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
parser saqrEgressParser(
        packet_in pkt,
        out saqr_header_t hdr,
        out eg_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

control saqrEgressDeparser(
        packet_out pkt,
        inout saqr_header_t hdr,
        in eg_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md) {
    apply {}
}

control saqrEgress(
        inout saqr_header_t hdr,
        inout eg_metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
    apply {}
}