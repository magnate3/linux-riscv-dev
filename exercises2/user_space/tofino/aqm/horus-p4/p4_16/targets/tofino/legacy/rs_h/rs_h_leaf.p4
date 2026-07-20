#include <core.p4>
#include <tna.p4>

#include "../common/headers.p4"
#include "../common/util.p4"
#include "../headers.p4"

control LeafIngress(
        inout saqr_header_t hdr,
        inout saqr_metadata_t saqr_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

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

             Register<queue_len_t, _>(MAX_VCLUSTERS) aggregate_queue_len_list; // One for each vcluster
                RegisterAction<queue_len_t, _, queue_len_t>(aggregate_queue_len_list) update_read_aggregate_queue_len = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                            value = value + saqr_md.queue_len_unit;
                        } else {
                            value = value - saqr_md.queue_len_unit;
                        }
                        rv = value;
                    }
                };
                RegisterAction<queue_len_t, _, queue_len_t>(aggregate_queue_len_list) read_aggregate_queue_len = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        rv = value;
                    }
                };

            Register<switch_id_t, _>(MAX_VCLUSTERS) linked_sq_sched; // ID of Spine that leaf sends load signal
                RegisterAction<switch_id_t, _, switch_id_t>(linked_sq_sched) read_linked_sq  = {
                    void apply(inout switch_id_t value, out switch_id_t rv) {
                        rv = value;
                        
                    }
                };
            
            Random<bit<16>>() random_worker_id_16;

            action get_worker_start_idx () {
                saqr_md.cluster_ds_start_idx = (bit <16>) (hdr.saqr.cluster_id * MAX_WORKERS_PER_CLUSTER);
            }

            action _drop() {
                ig_intr_dprsr_md.drop_ctl = 0x1; // Drop packet.
            }

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
                size = 1024;
                default_action = NoAction;
            }

            action act_get_cluster_num_valid(bit<16> num_ds_elements) {
                saqr_md.cluster_num_valid_ds = num_ds_elements;
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
 
            action gen_random_workers_16() {
                saqr_md.random_id_1 = (bit<16>) random_worker_id_16.get();
                saqr_md.random_id_2 = (bit<16>) random_worker_id_16.get();
            }
            
            action adjust_random_worker_range_8() {
                saqr_md.random_id_1 = saqr_md.random_id_1 >> 8;
                saqr_md.random_id_2 = saqr_md.random_id_2 >> 8;
            }

            action adjust_random_worker_range_5() {
                saqr_md.random_id_1 = saqr_md.random_id_1 >> 11;
                saqr_md.random_id_2 = saqr_md.random_id_2 >> 11;
            }

            action adjust_random_worker_range_4() {
                saqr_md.random_id_1 = saqr_md.random_id_1 >> 12;
                saqr_md.random_id_2 = saqr_md.random_id_2 >> 12;
            }

            action adjust_random_worker_range_3() {
                saqr_md.random_id_1 = saqr_md.random_id_1 >> 13;
                saqr_md.random_id_2 = saqr_md.random_id_2 >> 13;
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
                    adjust_random_worker_range_8(); // #== 256
                    adjust_random_worker_range_5();
                    adjust_random_worker_range_4(); 
                    adjust_random_worker_range_3();
                    adjust_random_worker_range_2(); 
                    adjust_random_worker_range_1(); // == 2
                    NoAction; // == 2^16
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
            apply {
                if (hdr.saqr.isValid()) {  // Saqr packet
                    
                    get_worker_start_idx(); // Get start index of workers for this vcluster
                    set_queue_len_unit.apply();
                    
                    @stage(1){
                        get_cluster_num_valid.apply();
                        gen_random_workers_16();
                        if (hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE || hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                            saqr_md.aggregate_queue_len = update_read_aggregate_queue_len.execute(hdr.saqr.cluster_id);
                            saqr_md.linked_sq_id = read_linked_sq.execute(hdr.saqr.cluster_id);
                        }
                    }
                    

                    @stage(2) {
                        saqr_md.mirror_dst_id = hdr.saqr.dst_id; // We want the original packet to reach its destination
                        if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                            adjust_random_range_ds.apply(); // move the random indexes to be in range of num workers in rack
                        }
                    } 
                    /** Stage 3
                     * Register:
                     * idle_list, dep: idle_count @st0, get_idle_index() @st 1, get_curr_idle_index() @st 2
                    */ 
                    @stage(3) {
                        if(hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {    
                            offset_random_ids();
                        } 
                    } 
                    
                    /** Stage 4
                     * 
                    */
                    @stage(4) {
                        if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                            saqr_md.random_ds_qlen_1 = read_queue_len_list_1.execute(saqr_md.random_id_1);
                            saqr_md.random_ds_qlen_2 = read_queue_len_list_2.execute(saqr_md.random_id_2);
                        } else if(hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE || hdr.saqr.pkt_type == PKT_TYPE_TASK_DONE_IDLE) {
                            write_queue_len_list_1.execute(hdr.saqr.src_id);
                            write_queue_len_list_2.execute(hdr.saqr.src_id);

                        } 
                    }

                    /** Stage 5
                     * 
                    */
                    @stage(5){
                    // packet is resubmitted
                        if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                            compare_queue_len();
                        } else if (hdr.saqr.pkt_type==PKT_TYPE_TASK_DONE_IDLE || hdr.saqr.pkt_type==PKT_TYPE_TASK_DONE) {
                            hdr.saqr.pkt_type = PKT_TYPE_QUEUE_SIGNAL;
                            // TESTBEDONLY: See comments in Saqr leaf. Uncomment below and comment the next line for real world.
                            //hdr.saqr.src_id = SWITCH_ID; 
                            hdr.saqr.src_id = hdr.saqr.cluster_id; 
                            hdr.saqr.qlen = saqr_md.aggregate_queue_len;
                            hdr.saqr.dst_id = saqr_md.linked_sq_id;
                            ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE; 
                        }
                    }

                    /* Stage 6
                     *
                    */
                    @stage(6) {
                        if (hdr.saqr.pkt_type == PKT_TYPE_NEW_TASK) {
                            if (saqr_md.selected_ds_qlen == saqr_md.random_ds_qlen_1) {
                                hdr.saqr.dst_id = saqr_md.random_id_1;
                            } else {
                                hdr.saqr.dst_id = saqr_md.random_id_2;
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
    apply {
        if (ig_intr_dprsr_md.mirror_type == MIRROR_TYPE_WORKER_RESPONSE) {
            mirror.emit<empty_t>((MirrorId_t) saqr_md.mirror_dst_id, {}); 
        } 
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.saqr);
    }
}

// Empty egress parser/control blocks
parser LeafEgressParser(
        packet_in pkt,
        out saqr_header_t hdr,
        out eg_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

control LeafEgressDeparser(
        packet_out pkt,
        inout saqr_header_t hdr,
        in eg_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md) {
    apply {}
}

control LeafEgress(
        inout saqr_header_t hdr,
        inout eg_metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
    apply {}
}