#include <core.p4>
#include <tna.p4>

#include "../common/headers.p4"
#include "../common/util.p4"
#include "../headers.p4"

control LeafIngress(
        inout horus_header_t hdr,
        inout horus_metadata_t horus_md,
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
                        value = hdr.horus.qlen;
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
                        value = hdr.horus.qlen;
                        rv = value;
                    }
                };

             Register<queue_len_t, _>(MAX_VCLUSTERS) aggregate_queue_len_list; // One for each vcluster
                RegisterAction<queue_len_t, _, queue_len_t>(aggregate_queue_len_list) update_read_aggregate_queue_len = {
                    void apply(inout queue_len_t value, out queue_len_t rv) {
                        if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                            value = value + horus_md.queue_len_unit;
                        } else {
                            value = value - horus_md.queue_len_unit;
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
                horus_md.cluster_ds_start_idx = (bit <16>) (hdr.horus.cluster_id * MAX_WORKERS_PER_CLUSTER);
            }

            action _drop() {
                ig_intr_dprsr_md.drop_ctl = 0x1; // Drop packet.
            }

            action act_forward_horus(PortId_t port, mac_addr_t dst_mac) {
                ig_intr_tm_md.ucast_egress_port = port;
                hdr.ethernet.dst_addr = dst_mac;
            }
            table forward_horus_switch_dst {
                key = {
                    hdr.horus.dst_id: exact;
                }
                actions = {
                    act_forward_horus;
                    NoAction;
                }
                size = 1024;
                default_action = NoAction;
            }

            action act_get_cluster_num_valid(bit<16> num_ds_elements) {
                horus_md.cluster_num_valid_ds = num_ds_elements;
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
 
            action gen_random_workers_16() {
                horus_md.random_id_1 = (bit<16>) random_worker_id_16.get();
                horus_md.random_id_2 = (bit<16>) random_worker_id_16.get();
            }
            
            action adjust_random_worker_range_8() {
                horus_md.random_id_1 = horus_md.random_id_1 >> 8;
                horus_md.random_id_2 = horus_md.random_id_2 >> 8;
            }

            action adjust_random_worker_range_5() {
                horus_md.random_id_1 = horus_md.random_id_1 >> 11;
                horus_md.random_id_2 = horus_md.random_id_2 >> 11;
            }

            action adjust_random_worker_range_4() {
                horus_md.random_id_1 = horus_md.random_id_1 >> 12;
                horus_md.random_id_2 = horus_md.random_id_2 >> 12;
            }

            action adjust_random_worker_range_3() {
                horus_md.random_id_1 = horus_md.random_id_1 >> 13;
                horus_md.random_id_2 = horus_md.random_id_2 >> 13;
            }

            action adjust_random_worker_range_2() {
                horus_md.random_id_1 = horus_md.random_id_1 >> 14;
                horus_md.random_id_2 = horus_md.random_id_2 >> 14;
            }

            action adjust_random_worker_range_1() {
                horus_md.random_id_1 = horus_md.random_id_1 >> 15;
                horus_md.random_id_2 = horus_md.random_id_2 >> 15;
            }

            table adjust_random_range_ds { // Reduce the random generated number (16 bit) based on number of workers in rack
                key = {
                    horus_md.cluster_num_valid_ds: exact; 
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
                horus_md.random_id_1 = horus_md.random_id_1 + horus_md.cluster_ds_start_idx;
                horus_md.random_id_2 = horus_md.random_id_2 + horus_md.cluster_ds_start_idx;
            }

            action compare_queue_len() {
                horus_md.selected_ds_qlen = min(horus_md.random_ds_qlen_1, horus_md.random_ds_qlen_2);
            }
            apply {
                if (hdr.horus.isValid()) {  // Horus packet
                    
                    get_worker_start_idx(); // Get start index of workers for this vcluster
                    set_queue_len_unit.apply();
                    
                    @stage(1){
                        get_cluster_num_valid.apply();
                        gen_random_workers_16();
                        if (hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.horus.pkt_type == PKT_TYPE_TASK_DONE || hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                            horus_md.aggregate_queue_len = update_read_aggregate_queue_len.execute(hdr.horus.cluster_id);
                            horus_md.linked_sq_id = read_linked_sq.execute(hdr.horus.cluster_id);
                        }
                    }
                    

                    @stage(2) {
                        horus_md.mirror_dst_id = hdr.horus.dst_id; // We want the original packet to reach its destination
                        if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                            adjust_random_range_ds.apply(); // move the random indexes to be in range of num workers in rack
                        }
                    } 
                    /** Stage 3
                     * Register:
                     * idle_list, dep: idle_count @st0, get_idle_index() @st 1, get_curr_idle_index() @st 2
                    */ 
                    @stage(3) {
                        if(hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {    
                            offset_random_ids();
                        } 
                    } 
                    
                    /** Stage 4
                     * 
                    */
                    @stage(4) {
                        if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                            horus_md.random_ds_qlen_1 = read_queue_len_list_1.execute(horus_md.random_id_1);
                            horus_md.random_ds_qlen_2 = read_queue_len_list_2.execute(horus_md.random_id_2);
                        } else if(hdr.horus.pkt_type == PKT_TYPE_TASK_DONE || hdr.horus.pkt_type == PKT_TYPE_TASK_DONE_IDLE) {
                            write_queue_len_list_1.execute(hdr.horus.src_id);
                            write_queue_len_list_2.execute(hdr.horus.src_id);

                        } 
                    }

                    /** Stage 5
                     * 
                    */
                    @stage(5){
                    // packet is resubmitted
                        if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                            compare_queue_len();
                        } else if (hdr.horus.pkt_type==PKT_TYPE_TASK_DONE_IDLE || hdr.horus.pkt_type==PKT_TYPE_TASK_DONE) {
                            hdr.horus.pkt_type = PKT_TYPE_QUEUE_SIGNAL;
                            // TESTBEDONLY: See comments in Horus leaf. Uncomment below and comment the next line for real world.
                            //hdr.horus.src_id = SWITCH_ID; 
                            hdr.horus.src_id = hdr.horus.cluster_id; 
                            hdr.horus.qlen = horus_md.aggregate_queue_len;
                            hdr.horus.dst_id = horus_md.linked_sq_id;
                            ig_intr_dprsr_md.mirror_type = MIRROR_TYPE_WORKER_RESPONSE; 
                        }
                    }

                    /* Stage 6
                     *
                    */
                    @stage(6) {
                        if (hdr.horus.pkt_type == PKT_TYPE_NEW_TASK) {
                            if (horus_md.selected_ds_qlen == horus_md.random_ds_qlen_1) {
                                hdr.horus.dst_id = horus_md.random_id_1;
                            } else {
                                hdr.horus.dst_id = horus_md.random_id_2;
                            }
                        }
                    }
                    
                    forward_horus_switch_dst.apply();
                    
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
        inout horus_header_t hdr,
        in horus_metadata_t horus_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {
    Mirror() mirror;
    apply {
        if (ig_intr_dprsr_md.mirror_type == MIRROR_TYPE_WORKER_RESPONSE) {
            mirror.emit<empty_t>((MirrorId_t) horus_md.mirror_dst_id, {}); 
        } 
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.horus);
    }
}

// Empty egress parser/control blocks
parser LeafEgressParser(
        packet_in pkt,
        out horus_header_t hdr,
        out eg_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

control LeafEgressDeparser(
        packet_out pkt,
        inout horus_header_t hdr,
        in eg_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md) {
    apply {}
}

control LeafEgress(
        inout horus_header_t hdr,
        inout eg_metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
    apply {}
}