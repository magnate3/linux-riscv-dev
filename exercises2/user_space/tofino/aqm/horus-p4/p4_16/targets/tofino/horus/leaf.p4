#include <core.p4>
#include <tna.p4>

#include "../common/headers.p4"
#include "../common/util.p4"
#include "../headers.p4"

/* Implementation of Horus Leaf 
 * Comments with the tag <TESTBEDONLY> mark the parts of the code that were modified for emulating multiple leaf schedulers using one switch.
 * These lines should be changed for normal operation (instructions for the changes are also provided in the commments)
 * 
 * Summary of a few lessons learned for P4 programming on TNA arch:
 *  An action can be called directly without a table (from apply{} block)
 *  Here multiple calls to action from the apply{} block is allowed (e.g in different if-else branches)
 *  Limitations: 
 *    If multiple operations (simple arith +,-,...) done in a single action results in error. "Action Require multiple stages for 
 *    a single action. We currently support only single stage actions."
 * 
 *    Multiple operations in a single branch of apply{} block not allowed. The operations must be done in seperate actions (That 
 *     translates to parallel ALU resources in hardware?)
 *    Index of reg (passed to .execute()) can not be computed in the same block of apply{}. But index can be computed in an action. 
 *    
 *   Multiple accesses to the same register is not allowed. We can define the RegActions in a way that we can handle the different conditions inside the RegAction (one access).
 *   
 *   Random generator only accepts constant upper bound (bit len). This causes problem for when number of nodes are not power of two!
 *   
 *   Only one RegisterAction may be executed per packet for a given Register. This is a significant limitation for our algorithm.
 *   Switch can not read n random registers and then increment the selected worker's register after comparison. We need to rely on worker to update the qlen later. 
 *   
 *   Comparing two metadeta feilds (with < >) in apply{} blcok resulted in error. (Too complex). Only can use == on two meta feilds!
 *   Therfore, we calculate min between two values and then in next stage check if that min value was the first sample or the second sample
*/

control LeafIngress(
        inout horus_header_t hdr,
        inout horus_metadata_t horus_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

        /* 
         **** Register definitions ***
         * All of the registers have seperate boundaries for each vcluster. 
         * We calculate the base/start index for the cluster_id and use the dedicated range for that cluster     
        */ 
            /* idle_list: holds the ID of the idle nodes */
            Register<worker_id_t, _>(MAX_WORKERS_IN_RACK) idle_list;
                RegisterAction<bit<16>, _, bit<16>>(idle_list) add_to_idle_list = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        value = hdr.horus.src_id;
                        rv = value;
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(idle_list) read_idle_list = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                    }
                };
            
            /* idle_count: Works as the pointer which points to the *next free slot* in the idle list. 
             * When writing new idle node, we write to the (idle_count) index and when reading the last valid idle node, 
               we read (idle_count-1) index.
            */
            Register<bit<16>, _>(MAX_VCLUSTERS) idle_count; 
                RegisterAction<bit<16>, _, bit<16>>(idle_count) read_idle_count = { 
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value; // Retruns val    
                    }
                };
                
                /* 
                 * Side Note:
                 * Possible Tofino Bug: Comparing value > MAX_VALUE(0xffff) returns always false!
                 * Possible Reason: https://community.intel.com/t5/Intel-Connectivity-Research/Modify-a-register-based-on-a-single-bit-of-its-current-value/m-p/1258877
                 * Comparison in SALU is converted to comparison with 0!
                 * e.g value < 0xffff translates to value - 0xffff < 0 which for unsigned bits results in a Bug!
                */ 
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
            /* queue_len_list_X: Holds the load (queue length) of each lower-layer node. 
             * We store two identical copies of this list so we can access the load values twice (for two samples)
            */
            Register<queue_len_t, _>(MAX_WORKERS_IN_RACK) queue_len_list_1; // List of queue lens for all vclusters
                RegisterAction<queue_len_t, _, queue_len_t>(queue_len_list_1) update_queue_len_list_1 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        value = value + 1;
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
                        value = hdr.horus.qlen;
                        rv = value;
                    }
                };
            
            Register<queue_len_t, _>(MAX_WORKERS_IN_RACK) queue_len_list_2; // List of queue lens for all vclusters
                RegisterAction<queue_len_t, _, queue_len_t>(queue_len_list_2) update_queue_len_list_2 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        value = value + 1;
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
                        value = hdr.horus.qlen;
                        rv = value;
                    }
                };
            
            /* 
             * deferred_queue_len_list_X: Holds the difference between the value in queue_len_list_X and the actual 
              load value (for each node).
             * This is because actual load is increased as we make scheduling decisions and we can not write the 
              incremented value to queue_len_list_X, so we keep them here and use these values for our 
              selective resubmission algorithm.
            */
            Register<queue_len_t, _>(MAX_WORKERS_IN_RACK) deferred_queue_len_list_1; // List of queue lens for all vclusters
                RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_1) check_deferred_queue_len_list_1 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        if (value <= horus_md.queue_len_diff) { // Queue len drift is not large enough to invalidate the decision
                            value = value + 1;
                            rv = 0;
                        } else {
                            rv = value + horus_md.selected_ds_qlen; // to avoid using another stage for this calculation
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
                        rv = value + horus_md.not_selected_ds_qlen;
                    }
                };
                 RegisterAction<queue_len_t, _, queue_len_t>(deferred_queue_len_list_2) reset_deferred_queue_len_list_2 = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        value = 0;
                        rv = value;
                    }
                };
            /* 
             * aggregate_queue_len_list: Maintains the average queue length of workers in rack. 
              This is used for reporting to spine.
             * In summary, we use horus_md.queue_len_unit which is the fixed point representation for 1/#workers in rack,
               and use inc/dec to maintain avg. instead of using division or floating point operations.
            */
            Register<queue_len_t, _>(MAX_VCLUSTERS) aggregate_queue_len_list; // One for each vcluster
                RegisterAction<bit<16>, _, bit<16>>(aggregate_queue_len_list) update_read_aggregate_queue_len = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                            value = value + horus_md.queue_len_unit;
                        } else {
                            if (value >= horus_md.queue_len_unit) {
                               value = value - horus_md.queue_len_unit; 
                            }
                        }
                        rv = value;
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(aggregate_queue_len_list) read_aggregate_queue_len = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                    }
                };
            
            /* 
             * linked_iq_sched: Maintains the Idle linkage status. 
             * Value INVALID_VALUE_16bit means that leaf is not linked with any spine.
             * Otherwise, it holds the ID of the spine switch that this leaf is linked with. 
             * When doesn't have anymore idle workers (or less than 2 idle worker), 
              it uses this info to send idleRemove to the linked spine
            */
            Register<switch_id_t, _>(MAX_VCLUSTERS) linked_iq_sched; // Spine that ToR has sent last IdleSignal (1 for each vcluster).
                RegisterAction<bit<16>, _, bit<16>>(linked_iq_sched) reset_linked_iq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value; // Return value before reset so leaf can unlink (send idle remove to the linked spine)
                        value = INVALID_VALUE_16bit;
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(linked_iq_sched) write_linked_iq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                        if (value == INVALID_VALUE_16bit){
                            value = horus_md.spine_to_link_iq;
                        }
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(linked_iq_sched) update_linked_iq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {    
                        value = hdr.horus.src_id;
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(linked_iq_sched) read_linked_iq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {    
                        rv = value;
                    }
                };
            
            /*
             * linked_sq_sched: maintains the ID of the spine scheduler for load linkage.
             * Leaf sends average load updates to the linked_sq. 
             * Side Note: In the current version we use static linkage (linkage is decided at initialization by controller)
              Therfore, a table could be used to store this info instead of register.
              The register however allows us to remove linkage or add new linkage in the dataplane (might be useful for dynamic load state updates).
            */
            Register<switch_id_t, _>(MAX_VCLUSTERS) linked_sq_sched; // Spine that ToR has sent last QueueSignal (1 for each vcluster).
                RegisterAction<bit<16>, _, bit<16>>(linked_sq_sched) read_update_linked_sq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                        if (value == INVALID_VALUE_16bit && hdr.horus.pkt_type == PKT_TYPE_SCAN_QUEUE_SIGNAL) { // Not linked before and new SCAN request arrived
                            value = hdr.horus.src_id;
                        }
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(linked_sq_sched) remove_linked_sq  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        value = INVALID_VALUE_16bit;
                        rv = value;
                    }
                };
            
            /*
             * linked_view_drift: Maintains the drift of the view of linked spine scheduler about the average load of this rack
               (since last time this leaf sent a state update message).
             * We use this to *selectively* send update message only when *all* of the workers have one less/more task in their queues.
             * How it works? : When spine selects a rack it internally increments the average load state (by 1/#workers).
             * But this load value changes as workers in this rack finish executing their tasks (it decreases).
             * We only send an update if all of the workers have finished one task since last update (actual average is decreased by 1).
            */
            Register<bit<16>, _>(MAX_VCLUSTERS) linked_view_drift; 
                RegisterAction<bit<16>, _, bit<16>>(linked_view_drift) inc_read_linked_view_drift  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        if (value == horus_md.cluster_num_valid_ds - 1) {
                            rv = 0;
                            value = 0;
                        } else {
                            rv = 1;
                            value = value + 1;
                        }
                    }
                };

            Register<bit<16>, _>(MAX_VCLUSTERS) idle_link_spine_view; 
                RegisterAction<bit<16>, _, bit<16>>(idle_link_spine_view) write_idle_link_spine_view  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        value = hdr.horus.qlen;
                    }
                };
                RegisterAction<bit<16>, _, bit<16>>(idle_link_spine_view) read_idle_link_spine_view  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        rv = value;
                        if (value == 1 && horus_md.cluster_idle_count == 0){
                            value = 0;
                        } else {
                            value = 1;
                        }
                    }
                };

             /*
              Backoff counters used to implement a buffer for delayed idle signals to avoid oscilation between states
              which results in high msg overheads and impacts the scheduling performance.
             */ 
             Register<bit<16>, _>(MAX_VCLUSTERS) backoff1; 
                RegisterAction<bit<16>, _, bit<16>>(backoff1) inc_read_backoff1  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        if (value == horus_md.num_valid_half) {
                            rv = 0;
                            value = 0;
                        } else {
                            rv = 1;
                            value = value + 1;
                        }
                    }
                };

                RegisterAction<bit<16>, _, bit<16>>(backoff1) reset_backoff1  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        value = 0;
                    }
                };

            Register<bit<16>, _>(MAX_VCLUSTERS) backoff3; 
                RegisterAction<bit<16>, _, bit<16>>(backoff3) inc_read_backoff3  = {
                    void apply(inout bit<16> value, out bit<16> rv) {
                        if (value == horus_md.cluster_num_valid_ds) {
                            rv = 0;
                            value = 0;
                        } else {
                            rv = 1;
                            value = value + 1;
                        }
                    }
                };

            /* 
             * TESTBEDONLY: Regs below only used for getting statistcs about the resubmission and update messages 
             * Remove them when used in production or anlyzing the resource usage of code!
            */
            Register<bit<32>, _>(MAX_VCLUSTERS) stat_count_resub; // Spine that ToR has sent last QueueSignal (1 for each vcluster).
                RegisterAction<bit<32>, _, bit<32>>(stat_count_resub) inc_stat_count_resub  = {
                    void apply(inout bit<32> value, out bit<32> rv) {
                        value = value + 1;
                    }
            };
            Register<bit<32>, _>(MAX_VCLUSTERS) stat_count_idle_signal; // Spine that ToR has sent last QueueSignal (1 for each vcluster).
                RegisterAction<bit<32>, _, bit<32>>(stat_count_idle_signal) inc_stat_count_idle_signal  = {
                    void apply(inout bit<32> value, out bit<32> rv) {
                        value = value + 1;
                    }
            };
            Register<bit<32>, _>(MAX_VCLUSTERS) stat_count_load_signal; // Spine that ToR has sent last QueueSignal (1 for each vcluster).
                RegisterAction<bit<32>, _, bit<32>>(stat_count_load_signal) inc_stat_count_load_signal  = {
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
                    value = horus_md.ingress_tstamp_clipped;
                }
            };
            
            /* 
             * Random number generator: 
             * Used for taking samples in pow-of-two choices. 
             * Also, for sampling in idle linkage process (sample from upper-layer spine schedulers).
            */
            Random<bit<16>>() random_worker_id_16;

            /* 
             * Calculates the base/start index for the register arrays (qlen_list, deferred_qlen_list, etc.) 
              based on the cluster_id of the coming packet.
            */
            action get_worker_start_idx () {
                horus_md.cluster_ds_start_idx = (bit <16>) (hdr.horus.cluster_id * MAX_WORKERS_PER_CLUSTER);
            }

            // Calculates the index of next idle worker in idle_list array. Base index + cluster_idle_count (pointer)
            action get_idle_index () {
                horus_md.idle_ds_index = horus_md.cluster_ds_start_idx + (bit <16>) horus_md.cluster_idle_count;
            }

            // Calculates the mirror dst id based on the actual dst_id and the emulated leaf index (TESTBEDONLY)
            action calc_mirror_dst_id() {
                horus_md.mirror_dst_id = hdr.horus.dst_id + hdr.horus.cluster_id;
            }

            // Since pointer points to *next available slot* in idle list. We decrement it when reading an idle node.
            action get_curr_idle_index() {
                horus_md.idle_ds_index = horus_md.idle_ds_index -1;
            }

            action _drop() { // Drop packet.
                ig_intr_dprsr_md.drop_ctl = 0x1; 
            }

            /* 
             * Gets the fixed point representation for (1/#workers) based on the cluster_id 
             * (how many workers belong to this cluster in this rack). 
             * Used for calculating average load.
            */ 
            action act_set_queue_len_unit(len_fixed_point_t cluster_unit) {
                horus_md.queue_len_unit = cluster_unit;
            }
            table set_queue_len_unit {
                key = {
                    hdr.horus.cluster_id: exact;
                }
                actions = {
                    act_set_queue_len_unit;
                    NoAction;
                }
                    size = HDR_CLUSTER_ID_SIZE;
                    default_action = NoAction;
            }

            /* For probing two out of n spine schedulers in idle linkage. 
              Not used in current experiments as we had one spine.
            */
            action gen_random_probe_group() { 
                ig_intr_tm_md.mcast_grp_a = (MulticastGroupId_t) 1; // Assume all use a single grp level 1
                /* 
                  Limitation: Casting the output of Random instance and assiging it directly to mcast_grp_b did not work. 
                  Had to assign it to a 16 bit meta field and then assign to mcast_group. 
                */
                // Different out ports for level 2 randomly generated
                // Here we use the same random 16 bit number generated for downstream ID to save resources
                ig_intr_tm_md.mcast_grp_b = horus_md.random_id_1; 
            }

            /* 
             * Forwards the packet based on the destination ID 
             * In our prototype, we directly use the node ID to assign the port and MAC address.
             * In production, the dst_id can be used as the key for routing tables (another stage).
            */
            action act_forward_horus(PortId_t port, mac_addr_t dst_mac) {
                ig_intr_tm_md.ucast_egress_port = port;
                hdr.ethernet.dst_addr = dst_mac;
            }
            table forward_horus_switch_dst {
                key = {
                    hdr.horus.dst_id: exact;
                    // TESTBEDONLY: to diffrentiate ports of virtual leaves
                    hdr.horus.cluster_id: exact;
                }
                actions = {
                    act_forward_horus;
                    NoAction;
                }
                size = 1024;
                default_action = NoAction;
            }
            /*
             * 1. Gets number of available workers (in this rack) for the vcluster.
             * 2. Gets number of spine schedulers for the vcluster.
             * (1) is used to generate random samples within the correct ranges when reading load registers and
              (2) is used for idle linkage to a spine scheduler.
            */
            action act_get_cluster_num_valid(bit<16> num_ds_elements, bit<16> num_us_elements) {
                horus_md.cluster_num_valid_ds = num_ds_elements;
                horus_md.cluster_num_valid_us = num_us_elements;
            }
            table get_cluster_num_valid {
                key = {
                    hdr.horus.cluster_id : exact;
                }
                actions = {
                    act_get_cluster_num_valid;
                    NoAction;
                }
                size = HDR_CLUSTER_ID_SIZE;
                default_action = NoAction;
            }

            /*
             * Since we can't select the random range in runtime, 
              we get() a 16bit random variable and shift it depending on desired range.
             * The table entries are constant and map number of bits to correct shift action
             * E.g Key: horus_md.cluster_num_valid_us=16. Action: adjust_random_worker_range_4 (shift 12 bits)
            */
            action gen_random_workers_16() { // Random for 65536 Nodes
                horus_md.random_id_1 = (bit<16>) random_worker_id_16.get();
                horus_md.random_id_2 = (bit<16>) random_worker_id_16.get();
            }
            
            action adjust_random_worker_range_8() { // Random for 256 Nodes
                horus_md.random_id_1 = horus_md.random_id_1 >> 8;
                horus_md.random_id_2 = horus_md.random_id_2 >> 8;
            }

             action adjust_random_worker_range_5() { // Random for 16 Nodes
                horus_md.random_id_1 = horus_md.random_id_1 >> 11;
                horus_md.random_id_2 = horus_md.random_id_2 >> 11;
            }

            action adjust_random_worker_range_4() { // Random for 16 Nodes
                horus_md.random_id_1 = horus_md.random_id_1 >> 12;
                horus_md.random_id_2 = horus_md.random_id_2 >> 12;
            }

            action adjust_random_worker_range_3() { // Random for 8 Nodes
                horus_md.random_id_1 = horus_md.random_id_1 >> 13;
                horus_md.random_id_2 = horus_md.random_id_2 >> 13;
            }

            action adjust_random_worker_range_2() { // Random for 4 Nodes
                horus_md.random_id_1 = horus_md.random_id_1 >> 14;
                horus_md.random_id_2 = horus_md.random_id_2 >> 14;
            }

            action adjust_random_worker_range_1() { // Random for 2 Nodes
                horus_md.random_id_1 = horus_md.random_id_1 >> 15;
                horus_md.random_id_2 = horus_md.random_id_2 >> 15;
            }

            table adjust_random_range_ds { // Reduce the random generated number (16 bit) based on number of workers in rack
                key = {
                    horus_md.cluster_num_valid_ds: exact; 
                }
                actions = {
                    adjust_random_worker_range_8(); // 256 Nodes == 8 bits
                    adjust_random_worker_range_5();
                    adjust_random_worker_range_4(); // 
                    adjust_random_worker_range_3(); //
                    adjust_random_worker_range_2(); // 
                    adjust_random_worker_range_1(); // 
                    NoAction; // == 16
                }
                size = 16;
                default_action = NoAction;
            }
            
            /* 
             Same logic as previous table, this time we use this for adjusting the random number within 
             range of number of spines (upstream). Used for idle linkage. 
            */
            table adjust_random_range_us { // Reduce the random generated number (16 bit) based on number of workers in rack
                key = {
                    horus_md.cluster_num_valid_us: exact; 
                }
                actions = {
                    adjust_random_worker_range_8(); // == 8 bits
                    adjust_random_worker_range_5();
                    adjust_random_worker_range_4(); // == 4
                    adjust_random_worker_range_3(); //
                    adjust_random_worker_range_2(); // == 2
                    adjust_random_worker_range_1(); // == 1
                    NoAction; // == 16
                }
                size = 16;
                default_action = NoAction;
            }

            /* 
             * Maps the generated random number to the ID of the spine switch (for idle linkage). 
             * Sine each vcluster might have different spines we use both random_id_1 and cluster_id as keys.
              E.g spine 0 for vcluster X is the spine with ID 100. While spine 0 for vcluster Y is the spine with ID 200. 
            */
            action act_get_spine_dst_id(bit <16> spine_dst_id){
                horus_md.spine_to_link_iq = spine_dst_id;
            }
            table get_spine_dst_id {
                key = {
                    horus_md.random_id_1: exact;
                    hdr.horus.cluster_id: exact;
                }
                actions = {
                    act_get_spine_dst_id();
                    NoAction;
                }
                size = 16;
                default_action = NoAction;
            }

            action send_pkt_to_cpu() {
                //hdr.horus.dst_id = PORT_PCI_CPU; Don't change dst_id, needed by upper-layer (controller logic)
                ig_intr_tm_md.ucast_egress_port = PORT_PCI_CPU;
            }

            /* Add the random generated index with the base index of cluster */ 
            action offset_random_ids() {
                horus_md.random_id_1 = horus_md.random_id_1 + horus_md.cluster_ds_start_idx;
                horus_md.random_id_2 = horus_md.random_id_2 + horus_md.cluster_ds_start_idx;
            }
            
            
            action dec_repeated_rand() { // Used for breaking ties when two randomly generated numbers point to same index
                horus_md.random_id_2 = horus_md.random_id_2-1;
            }
            action inc_repeated_rand() {
                horus_md.random_id_2 = horus_md.random_id_2+1;
            }

            action compare_queue_len() { // Find min qlen between the sampled qlen values and put result in selected_ds_qlen
                horus_md.selected_ds_qlen = min(horus_md.random_ds_qlen_1, horus_md.random_ds_qlen_2);
            }
            
            /* Used in resubmission path for scheduling: 
             Compares the actual/correct qlen (including load and drift) for samples */
            action compare_correct_queue_len() {
                horus_md.min_correct_qlen = min(horus_md.task_resub_hdr.qlen_1, horus_md.task_resub_hdr.qlen_2);
            }
            action get_larger_queue_len() {
                horus_md.not_selected_ds_qlen = max(horus_md.random_ds_qlen_1, horus_md.random_ds_qlen_2);
            }
            
            action calculate_queue_len_diff(){
                horus_md.queue_len_diff = horus_md.not_selected_ds_qlen - horus_md.selected_ds_qlen;
            }
            
            apply {
                if (hdr.horus.isValid()) {  // horus packet
                    if (ig_intr_md.resubmit_flag != 0) { // Special case: packet is resubmitted just update the indexes
                        @stage(0){
                            compare_correct_queue_len();
                            inc_stat_count_resub.execute(hdr.horus.cluster_id);
                        }
                        @stage(1){
                            if (horus_md.min_correct_qlen == horus_md.task_resub_hdr.qlen_1) {
                                hdr.horus.dst_id = horus_md.task_resub_hdr.ds_index_1;
                                horus_md.selected_ds_qlen = horus_md.task_resub_hdr.qlen_1 + 1;
                            } else {
                                hdr.horus.dst_id = horus_md.task_resub_hdr.ds_index_2;
                                horus_md.selected_ds_qlen = horus_md.task_resub_hdr.qlen_2 + 1;
                            }
                        }
                        @stage(5) {
                            update_queue_len_list_1.execute(hdr.horus.dst_id);
                            update_queue_len_list_2.execute(hdr.horus.dst_id);
                        }
                        // @stage(8) {
                        //     reset_deferred_queue_len_list_1.execute(hdr.horus.dst_id); // Just updated the queue_len_list so write 0 on deferred reg
                        // }
                        @stage(9) {
                            // reset_deferred_queue_len_list_2.execute(hdr.horus.dst_id);
                            hdr.horus.qlen = 0;
                        }
                    } else {
                        
                        get_worker_start_idx(); // Get start index (base address) for reg arrays for this vcluster
                        set_queue_len_unit.apply(); // Get (1/#workers in this rack) for this cluster (used for avg. calculation)
                        
                        if (hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE) {  // worker reply and its IDLE, read increment counter (pointer to idle list)
                            horus_md.cluster_idle_count = read_and_inc_idle_count.execute(hdr.horus.cluster_id);
                        } else if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) { // new task arrives, read decrement counter (cluster_idle_count points to next available slot in list)
                            horus_md.cluster_idle_count = read_and_dec_idle_count.execute(hdr.horus.cluster_id); // Read last idle count for vcluster
                        } else if (hdr.horus.pkt_type == PKT_TYPE_TASK_DONE) { // worker reply but still not idle, we just read the pointer (we use this later to send updates about idle linkage)
                            horus_md.cluster_idle_count = read_idle_count.execute(hdr.horus.cluster_id);
                        } else if (hdr.horus.pkt_type == PKT_TYPE_KEEP_ALIVE || hdr.horus.pkt_type == PKT_TYPE_WORKER_ID_ACK) {
                            send_pkt_to_cpu();
                        }

                        // Lines below are not used in current experiments. 
                        // Useful for removing load linkage in case of failure or dynamic linkage
                        if (hdr.horus.pkt_type == PKT_TYPE_QUEUE_REMOVE) { 
                            remove_linked_sq.execute(hdr.horus.cluster_id); 
                        } else {
                            horus_md.linked_sq_id = read_update_linked_sq.execute(hdr.horus.cluster_id); // Get ID of the Spine that the leaf reports to   
                        }

                        
                        // Side note: Compiler bug, if we calculate the index for reg action here, compiler complains but if in action its okay!

                        @stage(1) {
                            get_idle_index(); // Adds the base index to the pointer for idle list
                            get_cluster_num_valid.apply(); // Get number of workers in rack for this vcluster (used for adjusting random samples) 
                            gen_random_workers_16(); // Generate a 16 bit random number
                            if (hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.horus.pkt_type == PKT_TYPE_TASK_DONE || hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                                horus_md.aggregate_queue_len = update_read_aggregate_queue_len.execute(hdr.horus.cluster_id);
                            } 
                            horus_md.ingress_tstamp_clipped = (bit<32>)ig_intr_md.ingress_mac_tstamp[31:0];
                        }
                        
                        @stage(2) {
                            calc_mirror_dst_id(); // We want the original pkt to reach its destination (done later by mirroring the orignial pkt)
                            horus_md.received_dst_id = hdr.horus.dst_id; //  Keep received dst_id so that we can swap src_id dst_id to reply to other switches
                            if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK){
                                horus_md.num_valid_half = horus_md.cluster_num_valid_ds >> 1;
                                horus_md.task_counter = inc_stat_count_task.execute(0);
                                get_curr_idle_index(); // decrement the pointer so we read the correct idle worker id
                                adjust_random_range_ds.apply(); // shift the random indexes to be in range of num workers in rack
                            } else {
                                adjust_random_range_us.apply(); // shift the random indexes to be in range of num spine switches for vcluster
                            }
                        } 
                         
                        @stage(3) {
                            if (hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.horus.pkt_type == PKT_TYPE_TASK_DONE) { 
                                get_spine_dst_id.apply(); // Convert the random spine index (vcluster-based) to a global spine switch ID
                                if (hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE) { 
                                    add_to_idle_list.execute(horus_md.idle_ds_index); // If worker reply and its idle, add it to idle list
                                }
                                
                            } else if(hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                                write_ingress_tstamp.execute(hdr.horus.seq_num);
                                if (hdr.horus.qlen == 1) { // Spine thinks this leaf is idle
                                    horus_md.backoff_counter1 = inc_read_backoff1.execute(hdr.horus.cluster_id);
                                    
                                } else {
                                    reset_backoff1.execute(hdr.horus.cluster_id);
                                }
                                if (horus_md.cluster_idle_count > 0) { // If a new task arrives and idle workers available read idle_list to get ID of idle worker
                                    horus_md.idle_ds_id = read_idle_list.execute(horus_md.idle_ds_index); 
                                } else {
                                    if(horus_md.random_id_1 == horus_md.random_id_2) { 
                                        if (horus_md.random_id_2 == 0){ // Break ties in case two random numbers point to same index
                                            inc_repeated_rand();
                                        } else {
                                            dec_repeated_rand();
                                        }
                                    }
                                }
                            }
                        }
                        
                        @stage(4) { 
                            if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK ){
                                if (horus_md.cluster_idle_count==0){
                                    offset_random_ids(); // sum up the random numbers with base index to access the load list indices belonging to this vcluster    
                                }
                            } else if(hdr.horus.pkt_type == PKT_TYPE_TASK_DONE || hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE) {
                                horus_md.backoff_counter3 = inc_read_backoff3.execute(hdr.horus.cluster_id);
                            }
                        }

                        @stage(5) {
                            if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                                // Read two qlen values from list
                                horus_md.random_ds_qlen_1 = read_queue_len_list_1.execute(horus_md.random_id_1);
                                horus_md.random_ds_qlen_2 = read_queue_len_list_2.execute(horus_md.random_id_2);
                                
                                /* Here we check whether the spine that send us this task, thinks we are idle or not!
                                 * This is useful to detect if an idle add or idle remove procedure was not succesfull.
                                 * leaf updates the linkage reg based on the view of spine so that later it re-sends idle add or idle remove
                                 * If spine selected this rack based on idle assignment, it sets qlen field to 1
                                 * Otherwise, it sets qlen to 0.
                                */
                                if (hdr.horus.qlen == 1 && horus_md.backoff_counter1 == 0) { // Spine thinks this rack is idle so update the idle linkage (leaf is still linked)
                                    update_linked_iq.execute(hdr.horus.cluster_id);
                                } else if (hdr.horus.qlen == 0) { // Spine thinks rack is not idle
                                    reset_linked_iq.execute(hdr.horus.cluster_id);
                                }
                            } else if(hdr.horus.pkt_type == PKT_TYPE_TASK_DONE || hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE) {
                                write_queue_len_list_1.execute(hdr.horus.src_id); // Reply pkts contain the latest qlen of workers, so update the load lists
                                write_queue_len_list_2.execute(hdr.horus.src_id);
                                if (horus_md.cluster_idle_count > 1 && horus_md.backoff_counter3 ==0) { 
                                    // If there are idle workers available, leaf tries to initiate idle linkage:
                                    // Checks whether leaf is already linked with a spine, if not writes the new linkage (we send an idleAdd in next stage to this spine)
                                    horus_md.idle_link = write_linked_iq.execute(hdr.horus.cluster_id);
                                } else if (horus_md.cluster_idle_count <= 1) {
                                    // If the idle workers are not available, reset the linkage (read previous value so that we send idleRemove in next stage)
                                    horus_md.idle_link = read_linked_iq.execute(hdr.horus.cluster_id); 
                                } 
                            }  
                        }

                        @stage(6) {
                            if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                                compare_queue_len(); // Compare the sampled qlen values (for pow-of-two)
                                get_larger_queue_len(); 
                            } else if (hdr.horus.pkt_type == PKT_TYPE_TASK_DONE || hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE) { // Received a reply from worker and rack is not idle (removed the idle link)
                                if (horus_md.cluster_idle_count <= 1) { // Breaking the linkage
                                    if (horus_md.idle_link != INVALID_VALUE_16bit) { // Send a IDLE_REMOVE packet to the linked spine, mirror original reply to client
                                        hdr.horus.dst_id = horus_md.idle_link; // Send idle remove packet to the linked spine
                                        /* 
                                         * TESTBEDONLY: Here we should set the ID of the leaf switch as src_id, 
                                          but in our experimetns we used cluster_ids to diffrentiate the emulated leaf switches.
                                          So we use the cluster_id when sending reply to spine. 
                                          In real world it should be a constant SWITCH_ID. 
                                        */
                                        //hdr.horus.src_id = SWITCH_ID; 
                                        hdr.horus.src_id = hdr.horus.cluster_id;
                                        hdr.horus.qlen = horus_md.aggregate_queue_len; 
                                        hdr.horus.pkt_type = PKT_TYPE_IDLE_REMOVE;
                                        inc_stat_count_idle_signal.execute(hdr.horus.cluster_id);
                                        ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE;
                                    }
                                } else {
                                    if(horus_md.idle_link == INVALID_VALUE_16bit) { // Send IDLE_SIGNAL to the linked leaf if previously not linked with a spine
                                        /* 
                                         * TESTBEDONLY: Here we should set the ID of the leaf switch as src_id, 
                                          but in our experimetns we used cluster_ids to diffrentiate the emulated leaf switches.
                                          So we use the cluster_id when sending reply to spine. 
                                          In real world it should be a constant SWITCH_ID. 
                                        */
                                        hdr.horus.dst_id = horus_md.spine_to_link_iq;
                                        hdr.horus.qlen = horus_md.aggregate_queue_len;
                                        //hdr.horus.src_id = SWITCH_ID; 
                                        hdr.horus.src_id = hdr.horus.cluster_id; // Only for the virtual leaf in testbed experimetns TODO: Constant value switch id for the production
                                        hdr.horus.pkt_type = PKT_TYPE_IDLE_SIGNAL; // Now change to idle signal to notify the selected spine
                                        inc_stat_count_idle_signal.execute(hdr.horus.cluster_id);
                                        ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE;
                                    }
                                }
                            }
                        }

                        @stage(7) {
                            calculate_queue_len_diff();
                            if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                                if (horus_md.cluster_idle_count == 0) {
                                    if (horus_md.selected_ds_qlen == horus_md.random_ds_qlen_1) { // If minimum qlen belongs to our first sample
                                        hdr.horus.dst_id = horus_md.random_id_1; // Set dst_id to id of first sample
                                        // line below is used in case of resubmission, we always keep the id of *not selected* in task_resub_hdr.ds_index_2
                                        horus_md.task_resub_hdr.ds_index_2 = horus_md.random_id_2; 
                                    } else { // If minimum qlen belongs to our second sample
                                        hdr.horus.dst_id = horus_md.random_id_2;
                                        horus_md.task_resub_hdr.ds_index_2 = horus_md.random_id_1;
                                    }
                                } else { // If idle workers available dst_id is the one we read from idle_list
                                    hdr.horus.dst_id = horus_md.idle_ds_id;
                                }
                            } else if (hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.horus.pkt_type == PKT_TYPE_TASK_DONE) {
                                horus_md.spine_view_ok = inc_read_linked_view_drift.execute(hdr.horus.cluster_id); // Check if drift of load of the spine is larger than threshold
                            } 
                        }

                        @stage(8) {
                            if (hdr.horus.pkt_type==PKT_TYPE_TASK_DONE_IDLE || hdr.horus.pkt_type==PKT_TYPE_TASK_DONE){
                                reset_deferred_queue_len_list_1.execute(hdr.horus.src_id); // Just updated the queue_len_list so write 0 on deferred qlen reg (aka. drift)

                            } else if(hdr.horus.pkt_type==PKT_TYPE_NEW_TASK && horus_md.cluster_idle_count == 0) {
                                if (horus_md.random_id_2 != horus_md.random_id_1) {
                                    // Action below Returns QL[dst_id] + Deferred[dst_id] if resubmission needed o.t it will return 0
                                    // task_resub_hdr.qlen_1  will contain the complete load info about the node we initially selected
                                    // and its used in resub path
                                    horus_md.task_resub_hdr.qlen_1 = check_deferred_queue_len_list_1.execute(hdr.horus.dst_id); 
                                    // keep the ID of selected dst in ds_index_1
                                    horus_md.task_resub_hdr.ds_index_1 = hdr.horus.dst_id;
                                } else { // In case two samples point to the same cell, we do not need to resubmit just increment deferred list
                                    inc_deferred_queue_len_list_1.execute(hdr.horus.dst_id);
                                }
                            }
                        }
                        @stage(9){
                            if (hdr.horus.pkt_type==PKT_TYPE_TASK_DONE_IDLE || hdr.horus.pkt_type==PKT_TYPE_TASK_DONE) {
                                reset_deferred_queue_len_list_2.execute(hdr.horus.src_id); // Just updated the queue_len_list so write 0 on deferred reg
                                if (horus_md.spine_view_ok == 0) { // Need to send a new load signal to spine 
                                    /* 
                                    Desired behaviour: Mirror premitive (emit invoked in ingrdeparser) will send the original reply to the client
                                    Here we modify the packet and send it as a ctrl pkt to the linked spine.
                                    */
                                    hdr.horus.pkt_type = PKT_TYPE_QUEUE_SIGNAL;
                                    /* 
                                     * TESTBEDONLY: Here we should set the ID of the leaf switch as src_id, 
                                      but in our experimetns we used cluster_ids to diffrentiate the emulated leaf switches.
                                      So we use the cluster_id when sending reply to spine. 
                                      In real world it should be a constant SWITCH_ID. 
                                    */
                                    //hdr.horus.src_id = SWITCH_ID; 
                                    hdr.horus.src_id = hdr.horus.cluster_id; 
                                    hdr.horus.qlen = horus_md.aggregate_queue_len;
                                    inc_stat_count_load_signal.execute(hdr.horus.cluster_id);
                                    hdr.horus.dst_id = horus_md.linked_sq_id;
                                    ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE; 
                                }
                            } else if(hdr.horus.pkt_type==PKT_TYPE_NEW_TASK) {
                                if (horus_md.cluster_idle_count == 0){
                                    if(horus_md.task_resub_hdr.qlen_1 == 0) { // This return value means that we do not need to check deffered qlens, difference between samples were larger than drift so our decision is still valid
                                        inc_deferred_queue_len_list_2.execute(hdr.horus.dst_id); // increment the second copy to be consistent with first one
                                    } else { // our decision might be invalid, need to check the deffered queue lens for the sample that was greater (not_selected) and resubmit
                                        ig_intr_dprsr_md.resubmit_type = RESUBMIT_TYPE_NEW_TASK;
                                        horus_md.task_resub_hdr.qlen_2 = read_deferred_queue_len_list_2.execute(horus_md.task_resub_hdr.ds_index_2);
                                    }
                                    hdr.horus.qlen = 0;
                                } else { // If there were idle workers avilable, we poped that worker from the idle list. 
                                    /* 
                                     * Important: we set qlen field to 1 when sending the task to worker to tell the monitoring agent
                                     * that this worker were deleted from the idle list of leaf scheduler
                                     * each worker will send PKT_TYPE_TASK_DONE_IDLE when it becomes idle iff it had 
                                     * recieved one packet with qlen==1 before. 
                                     * This ensures that idle list of leaf stays unique: one worker will be added only if it was previously removed. 
                                     */
                                    hdr.horus.qlen = 1;
                                }
                            } 
                        }
                    }
                    if (hdr.horus.pkt_type != PKT_TYPE_WORKER_ID_ACK && hdr.horus.pkt_type != PKT_TYPE_KEEP_ALIVE){
                        forward_horus_switch_dst.apply(); // Forwarding tables...     
                    }
                    
                }  else if (hdr.ipv4.isValid()) { // Regular switching procedure
                    // TODO: Not ported the ip matching tables for now
                    _drop();
                } else {
                    _drop();
                }
            }
        }

control LeafIngressDeparser(
        packet_out pkt,
        inout horus_header_t hdr,
        in horus_metadata_t horus_md,
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
             output port (value) (same table as horus forward in ingress)
            */
            // TODO: Bug Report to community. emit() should support single argument call when no header is needed. 
            // But gets compiler internal error! So we add {} (empty mirror hdr)
            mirror.emit<empty_t>((MirrorId_t) horus_md.mirror_dst_id, {}); 
        }  
        if (ig_intr_dprsr_md.resubmit_type == RESUBMIT_TYPE_NEW_TASK) { // Resubmit triggered for schedling a task
            resubmit.emit(horus_md.task_resub_hdr);
        }
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.horus);
    }
}

// Empty egress parser/control blocks
parser HorusEgressParser(
        packet_in pkt,
        out horus_header_t hdr,
        out eg_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        pkt.extract(hdr.ethernet);
        pkt.extract(hdr.ipv4);
        pkt.extract(hdr.udp);
        pkt.extract(hdr.horus);
        transition accept;
    }
}

control HorusEgressDeparser(
        packet_out pkt,
        inout horus_header_t hdr,
        in eg_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.horus);
    }
}

control HorusEgress(
        inout horus_header_t hdr,
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
        if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
            eg_md.task_counter = inc_stat_count_task.execute(0);
            eg_md.egress_tstamp_clipped = (bit<32>)eg_intr_md_from_prsr.global_tstamp[31:0];
            write_egress_tstamp.execute(hdr.horus.seq_num);
        }
    }
}