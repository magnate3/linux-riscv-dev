#include <core.p4>
#include <v1model.p4>

#include "header.p4"
#include "parser.p4"

#define NUM_VCLUSTERS_PER_RACK 8

// Used by spine schedulers, currently hardcoded (can be set from ctrl plane)
#define SWITCH_ID 1

typedef bit<8> queue_len_t;
typedef bit<9> port_id_t;
typedef bit<16> worker_id_t;

typedef bit<QUEUE_LEN_FIXED_POINT_SIZE> len_fixed_point_t;

control egress(inout headers hdr, inout metadata meta, inout standard_metadata_t standard_metadata) {
    action act_set_src_id(){
        hdr.falcon.src_id = SWITCH_ID;
    }

    table set_src_id {
        actions = {act_set_src_id;}
        default_action = act_set_src_id;
    }

    apply {  
        set_src_id.apply();
    }
}

control ingress(inout headers hdr, inout metadata meta, inout standard_metadata_t standard_metadata) {
    register<bit<16>>((bit<32>) NUM_VCLUSTERS_PER_RACK) linked_iq_sched; // Spine that ToR has sent last IdleSignal.
    register<bit<16>>((bit<32>) NUM_VCLUSTERS_PER_RACK) linked_sq_sched; // Spine that ToR has sent last QueueSignal.
    
    // List of idle workers up to 16 (idle workers) * 64 (clusters) 
    // Value 0x00 means Non-valid (NULL)
    register<worker_id_t>((bit<32>) 1024) idle_list; 
    register<bit<HDR_SRC_ID_SIZE>>((bit<32>) NUM_VCLUSTERS_PER_RACK) idle_count; // Idle count for each cluster, acts as pointer going frwrd and backwrd to point to idle worker list

    register<queue_len_t>((bit <32>) 1024) queue_len_list; // List of queue lens 8 (workers) * 128 (clusters)
    register<queue_len_t>((bit <32>) NUM_VCLUSTERS_PER_RACK) aggregate_queue_len_list;

    register<queue_len_t>((bit<32>) NUM_VCLUSTERS_PER_RACK) spine_iq_len_1;
    //register<queue_len_t>((bit<32>) NUM_VCLUSTERS_PER_RACK) spine_iq_len_2;

    register<worker_id_t>((bit<32>) NUM_VCLUSTERS_PER_RACK) spine_probed_id;
    //register<worker_id_t>((bit<32>) NUM_VCLUSTERS_PER_RACK) spine_sw_id_2;

    //register<bit<16>>((bit <32>) 1024) workers_per_cluster;
    //register<bit<16>>((bit <32>) 1024) spines_per_cluster;

    action _drop() {
        mark_to_drop(standard_metadata);
    }
    
    action act_gen_rand_probe_group() {
        /* 
        TODO: Use modify_field_rng_uniform instead of random<> for hardware targets
        This is not implemented in bmv but available in Hardware. 
        */
        //modify_field_rng_uniform(meta.falcon_meta.rand_probe_group, 0, RAND_MCAST_RANGE);   
        random<bit<HDR_FALCON_RAND_GROUP_SIZE>>(meta.falcon_meta.rand_probe_group, 0, RAND_MCAST_RANGE);
    }

    action act_set_queue_len_unit(len_fixed_point_t cluster_unit){
        meta.falcon_meta.queue_len_unit = cluster_unit;
    }

    action mac_forward(port_id_t port) {
        standard_metadata.egress_spec = port;
    }

    action act_gen_random_worker_id_1() {
        //modify_field_rng_uniform(meta.falcon_meta.random_downstream_id_1, 0, meta.falcon_meta.cluster_num_valid_ds);
        random<bit<HDR_SRC_ID_SIZE>>(meta.falcon_meta.random_downstream_id_1, 0, meta.falcon_meta.cluster_num_valid_ds - 1);
        meta.falcon_meta.random_downstream_id_1 = meta.falcon_meta.random_downstream_id_1 + meta.falcon_meta.cluster_worker_start_idx;
    }

    action act_gen_random_worker_id_2() {
        //modify_field_rng_uniform(meta.falcon_meta.random_downstream_id_2, 0, meta.falcon_meta.cluster_num_valid_ds);
        random<bit<HDR_SRC_ID_SIZE>>(meta.falcon_meta.random_downstream_id_2, 0, meta.falcon_meta.cluster_num_valid_ds - 1);
        meta.falcon_meta.random_downstream_id_2 = meta.falcon_meta.random_downstream_id_2 + meta.falcon_meta.cluster_worker_start_idx;
    }

    action act_get_cluster_num_valid_ds(bit<16> num_ds_elements) {
        meta.falcon_meta.cluster_num_valid_ds = num_ds_elements;
    }

    action act_read_idle_count() {
        idle_count.read(meta.falcon_meta.cluster_idle_count, (bit<32>) hdr.falcon.local_cluster_id);
        meta.falcon_meta.cluster_worker_start_idx = (bit <16>) (hdr.falcon.local_cluster_id * MAX_WORKERS_PER_CLUSTER);
        /* TODO: use "add_to_field()" for hardware targets, simply "+" in bvm */
        //meta.falcon_meta.cluster_idle_count = meta.falcon_meta.cluster_idle_count + 1;
        meta.falcon_meta.idle_worker_index = (bit <16>) meta.falcon_meta.cluster_idle_count + meta.falcon_meta.cluster_worker_start_idx;
    
        //add_to_field(meta.falcon_meta.idle_worker_index, meta.falcon_meta.cluster_idle_count);
        //add_to_field(meta.falcon_meta.idle_worker_index, hdr.falcon.local_cluster_id);
    }

    action act_add_to_idle_list() {
        idle_list.write((bit<32>) meta.falcon_meta.idle_worker_index, hdr.falcon.src_id);
        meta.falcon_meta.idle_worker_index = meta.falcon_meta.idle_worker_index + 1;
        idle_count.write((bit<32>) hdr.falcon.local_cluster_id, meta.falcon_meta.idle_worker_index);
    }

    action act_pop_from_idle_list () {
        meta.falcon_meta.idle_worker_index = meta.falcon_meta.idle_worker_index - 1;
        idle_list.read(hdr.falcon.dst_id, (bit<32>) meta.falcon_meta.idle_worker_index);
        idle_count.write((bit<32>) hdr.falcon.local_cluster_id, meta.falcon_meta.idle_worker_index);
        //add_to_field(meta.falcon_meta.idle_worker_index, -1);
    }

    action act_decrement_queue_len() {
        // Update queue len
        meta.falcon_meta.worker_index = (bit<16>) hdr.falcon.src_id + meta.falcon_meta.cluster_worker_start_idx;
        queue_len_list.read(meta.falcon_meta.qlen_curr, (bit<32>)meta.falcon_meta.worker_index);
        meta.falcon_meta.qlen_curr = meta.falcon_meta.qlen_curr - meta.falcon_meta.queue_len_unit;
        queue_len_list.write((bit<32>)meta.falcon_meta.worker_index, meta.falcon_meta.qlen_curr);

        aggregate_queue_len_list.read(meta.falcon_meta.qlen_agg, (bit<32>) hdr.falcon.local_cluster_id);
        meta.falcon_meta.qlen_agg = meta.falcon_meta.qlen_agg - meta.falcon_meta.queue_len_unit;
        aggregate_queue_len_list.write((bit<32>) hdr.falcon.local_cluster_id, meta.falcon_meta.qlen_agg);
    }

    action act_forward_falcon(bit<9> port) {
        standard_metadata.egress_spec = port;
    }

    action act_cmp_random_qlen() {
        // if (meta.falcon_meta.random_downstream_id_1 == meta.falcon_meta.random_downstream_id_2){
        //     meta.falcon_meta.selected_downstream_id = meta.falcon_meta.random_downstream_id_1;
        // }
        queue_len_list.read(meta.falcon_meta.qlen_rand_1, (bit<32>) meta.falcon_meta.random_downstream_id_1);
        queue_len_list.read(meta.falcon_meta.qlen_rand_2, (bit<32>) meta.falcon_meta.random_downstream_id_2);
        if (meta.falcon_meta.qlen_rand_1 >= meta.falcon_meta.qlen_rand_2) {
            hdr.falcon.dst_id = meta.falcon_meta.random_downstream_id_2;
        } else {
            hdr.falcon.dst_id = meta.falcon_meta.random_downstream_id_1;
        }
    }

    action act_increment_queue_len() {
        queue_len_list.read(meta.falcon_meta.qlen_curr, (bit<32>)hdr.falcon.dst_id);
        meta.falcon_meta.qlen_curr = meta.falcon_meta.qlen_curr + meta.falcon_meta.queue_len_unit;
        queue_len_list.write((bit<32>)hdr.falcon.dst_id, meta.falcon_meta.qlen_curr);  

        aggregate_queue_len_list.read(meta.falcon_meta.qlen_agg, (bit<32>) hdr.falcon.local_cluster_id);
        meta.falcon_meta.qlen_agg = meta.falcon_meta.qlen_agg + meta.falcon_meta.queue_len_unit;
        aggregate_queue_len_list.write((bit<32>) hdr.falcon.local_cluster_id, meta.falcon_meta.qlen_agg);       
    }

    action act_check_last_probe() {
        spine_iq_len_1.read(meta.falcon_meta.last_idle_list_len, (bit<32>) hdr.falcon.local_cluster_id);
        spine_probed_id.read(meta.falcon_meta.last_idle_probe_id, (bit<32>) hdr.falcon.local_cluster_id);
    }

    action act_update_probe() {
        spine_iq_len_1.write((bit<32>)hdr.falcon.local_cluster_id, hdr.falcon.qlen);
        spine_probed_id.write((bit<32>)hdr.falcon.local_cluster_id, hdr.falcon.src_id);
    }

    action act_reset_probe_state() {
        linked_iq_sched.write((bit<32>) hdr.falcon.local_cluster_id, meta.falcon_meta.shortest_idle_queue_id);
        spine_iq_len_1.write((bit<32>) hdr.falcon.local_cluster_id, 0xFF);
    }

    action act_read_linked_sq() {
        linked_sq_sched.read(meta.falcon_meta.linked_sq_id, (bit<32>) hdr.falcon.local_cluster_id);
    }

    action act_update_linked_sq() {
        linked_sq_sched.write((bit<32>) hdr.falcon.local_cluster_id, hdr.falcon.src_id);
    }

    action broadcast() {
        standard_metadata.mcast_grp = 1;
        meta.ingress_metadata.nhop_ipv4 = hdr.ipv4.dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl + 8w255;
    }

    table set_queue_len_unit {
        key = {
            hdr.falcon.local_cluster_id: exact;
        }
        actions = {
            act_set_queue_len_unit;
            _drop;
        }
        size = HDR_CLUSTER_ID_SIZE;
        default_action = _drop;
    }

    table gen_random_probe_group {
        actions = {act_gen_rand_probe_group;}
        default_action = act_gen_rand_probe_group;
    }

    table gen_random_downstream_id_1 {
        actions = {act_gen_random_worker_id_1;}
        default_action = act_gen_random_worker_id_1;
    }

    table gen_random_downstream_id_2 {
        actions = {act_gen_random_worker_id_2;}
        default_action = act_gen_random_worker_id_2;
    }

    // Gets the actual number of downstream elements (workers or tor schedulers) for vcluster (passed by ctrl plane)
    table get_cluster_num_valid_ds {
        key = {
            hdr.falcon.cluster_id : exact;
        }
        actions = {
            act_get_cluster_num_valid_ds;
            NoAction;
        }
        size = HDR_CLUSTER_ID_SIZE;
        default_action = NoAction;
    }

    table read_idle_count {
        actions = {act_read_idle_count;}
        default_action = act_read_idle_count;
    }

    table add_to_idle_list {
        actions = {act_add_to_idle_list;}
        default_action = act_add_to_idle_list;
    }

    table pop_from_idle_list {
        actions = {act_pop_from_idle_list;}
        default_action = act_pop_from_idle_list;
    }

    // table get_worker_index {
    //     actions = {act_get_worker_index;}
    //     default_action = act_get_worker_index;
    // }

    table decrement_queue_len {
        actions = {act_decrement_queue_len;}
        default_action = act_decrement_queue_len;
    }

    table cmp_random_qlen {
        actions = {act_cmp_random_qlen;}
        default_action = act_cmp_random_qlen;
    }
    // Currently uses the ID of workers to forward downstream.
    // Mapping from worker IDs for each vcluster to physical port passed by control plane tables. 
    table forward_falcon {
        key = {
            hdr.falcon.dst_id: exact;
            hdr.falcon.cluster_id: exact;
        }
        actions = {
            act_forward_falcon;
            NoAction;
        }
        size = HDR_SRC_ID_SIZE;
        default_action = NoAction;
    }

    table increment_queue_len {
        actions = {act_increment_queue_len;}
        default_action = act_increment_queue_len;
    }

    table check_last_probe {
        actions = {act_check_last_probe;}
        default_action = act_check_last_probe;
    }

    table update_probe {
        actions = {act_update_probe;}
        default_action = act_update_probe;
    }

    table reset_probe_state {
        actions = {act_reset_probe_state;}
        default_action = act_reset_probe_state;
    }

    table read_linked_sq {
        actions = {act_read_linked_sq;}
        default_action = act_read_linked_sq;
    }

    table update_linked_sq {
        actions = {act_update_linked_sq;}
        default_action = act_update_linked_sq;
    }

    action set_nhop(bit<32> nhop_ipv4, bit<9> port) {
        meta.ingress_metadata.nhop_ipv4 = nhop_ipv4;
        standard_metadata.egress_spec = port;
        hdr.ipv4.ttl = hdr.ipv4.ttl + 8w255;
    }
    action set_dmac(bit<48> dmac) {
        hdr.ethernet.dstAddr = dmac;
    }

    table ipv4_lpm {
        actions = {
            _drop;
            set_nhop;
            broadcast;
            NoAction;
        }
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        size = 1024;
        default_action = NoAction();
    }
    table forward {
        actions = {
            set_dmac;
            _drop;
            NoAction;
        }
        key = {
            meta.ingress_metadata.nhop_ipv4: exact;
        }
        size = 512;
        default_action = NoAction();
    }

    apply {
        if (hdr.falcon.isValid()) {
            // TODO: Optimization: these tables do not apply for packets upstream links 
            
            read_idle_count.apply();
            read_linked_sq.apply();
            set_queue_len_unit.apply();
            if (hdr.falcon.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.falcon.pkt_type == PKT_TYPE_TASK_DONE) {
                decrement_queue_len.apply();
                if (meta.falcon_meta.linked_sq_id != 0xFF) { // not Null. TODO: fix Null value 0xFF port is valid
                    hdr.falcon.pkt_type = PKT_TYPE_QUEUE_SIGNAL;
                    hdr.falcon.qlen = meta.falcon_meta.qlen_agg; // Reporting agg qlen to Spine
                    // TODO: Fix forwarding (use layer 2 routing)
                    standard_metadata.egress_spec = (bit<9>)meta.falcon_meta.linked_sq_id;
                }

                if (hdr.falcon.pkt_type == PKT_TYPE_TASK_DONE_IDLE) {
                    if (meta.falcon_meta.cluster_idle_count < MAX_IDLE_WORKERS_PER_CLUSTER) {
                        add_to_idle_list.apply();
                    }
                    if (meta.falcon_meta.cluster_idle_count == 1) { // Just became Idle, need to anounce to a spine scheduler
                        gen_random_probe_group.apply();
                        hdr.falcon.pkt_type = PKT_TYPE_PROBE_IDLE_QUEUE; // Send probes
                        /* TODO: use "modify_field()" for hardware targets */ 
                        standard_metadata.mcast_grp = (bit <16>) meta.falcon_meta.rand_probe_group;
                        //modify_field(standard_metadata.mcast_grp, meta.falcon_meta.rand_probe_group);
                    }
                }
            } else if(hdr.falcon.pkt_type == PKT_TYPE_NEW_TASK) {
                /*
                 TODO @parham: Remove the ToR from linked spine if the packet is coming based on random decision
                 This needs a copy of packet (original task) to go to the server and another copy (ctrl msg) to send to linked_iq
                */
                if (meta.falcon_meta.cluster_idle_count > 0) { //Idle workers available
                    pop_from_idle_list.apply();
                    if (meta.falcon_meta.cluster_idle_count == 1) { // No more idle after this assignment
                        linked_iq_sched.write(0, 0); // Set to NULL
                    }
                } else {
                    get_cluster_num_valid_ds.apply();
                    gen_random_downstream_id_1.apply();
                    gen_random_downstream_id_2.apply();
                    cmp_random_qlen.apply();
                }
                increment_queue_len.apply(); // TODO: Optimize this, needs to read the current queue lenght again (once read by previous actions)
                forward_falcon.apply();
            } else if (hdr.falcon.pkt_type == PKT_TYPE_PROBE_IDLE_RESPONSE) {
                if (meta.falcon_meta.cluster_idle_count > 0) { // Still idle workers available
                    check_last_probe.apply();
                    if (meta.falcon_meta.last_idle_list_len == 0xFF) { // Not set yet, this is first probe response
                        update_probe.apply();
                    } else { // This is the the second probe response
                        if (meta.falcon_meta.last_idle_list_len >= hdr.falcon.qlen) {
                            spine_probed_id.read(meta.falcon_meta.shortest_idle_queue_id, (bit<32>) hdr.falcon.local_cluster_id);
                            } else { // The spine that just sent its idle len is target
                                meta.falcon_meta.shortest_idle_queue_id = hdr.falcon.src_id;
                            }
                            // Send idle signal to spine sw
                            standard_metadata.egress_spec = (bit<9>) meta.falcon_meta.shortest_idle_queue_id;
                            hdr.falcon.pkt_type = PKT_TYPE_IDLE_SIGNAL;
                            reset_probe_state.apply();
                        }
                    }
            } else if (hdr.falcon.pkt_type == PKT_TYPE_QUEUE_REMOVE) {
                linked_sq_sched.write((bit<32>) hdr.falcon.local_cluster_id, 0xFF);
            } else if (hdr.falcon.pkt_type == PKT_TYPE_SCAN_QUEUE_SIGNAL) {
                if (meta.falcon_meta.linked_sq_id == 0xFF) {
                    update_linked_sq.apply();
                    hdr.falcon.pkt_type = PKT_TYPE_QUEUE_SIGNAL;
                    hdr.falcon.qlen = meta.falcon_meta.qlen_agg; // Reporting agg qlen to Spine
                    standard_metadata.egress_spec = (bit<9>) meta.falcon_meta.linked_sq_id;
                }
            }
            else {
                mark_to_drop(standard_metadata);
            }
            
        } else {
            // Apply regular switch procedure
            if (hdr.ipv4.isValid()) {
                ipv4_lpm.apply();
                forward.apply();
            }
        }
    }
}

V1Switch(ParserImpl(), verifyChecksum(), ingress(), egress(), computeChecksum(), DeparserImpl()) main;
