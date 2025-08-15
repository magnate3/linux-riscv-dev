

```
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
```