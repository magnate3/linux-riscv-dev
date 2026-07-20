#include <core.p4>
#include <tna.p4>

#include "header.p4"
#include "parser.p4"

// Define constants for types of packets
// Source: https://carolinafernandez.github.io/development/2019/08/06/Recurrent-processing-in-P4
#define PKT_INSTANCE_TYPE_NORMAL 0
#define PKT_INSTANCE_TYPE_INGRESS_CLONE 1
#define PKT_INSTANCE_TYPE_EGRESS_CLONE 2
#define PKT_INSTANCE_TYPE_COALESCED 3
#define PKT_INSTANCE_TYPE_INGRESS_RECIRC 4
#define PKT_INSTANCE_TYPE_REPLICATION 5
#define PKT_INSTANCE_TYPE_RESUBMIT 6

// Use this ID when populating tables for broadcast to all (downstream) ports
#define MCAST_ID_BROADCAST 255

// Note: Set this according to switch layer. Larger values for spine as switch is in path of more vclusters.
#define MAX_VCLUSTERS 8

// TODO: Check this approach, using combinations of 2 out of #spine schedulers (16) to PROBE_IDLE_QUEUE
#define RAND_MCAST_RANGE 1

#define LEAF_SPINE_RATIO 10

typedef bit<8> queue_len_t;
typedef bit<9> port_id_t;
typedef bit<16> worker_id_t;

typedef bit<QUEUE_LEN_FIXED_POINT_SIZE> len_fixed_point_t;

control egress(inout headers hdr, inout metadata meta, inout standard_metadata_t standard_metadata) {
    action act_set_src_id(bit <16> self_id){
        hdr.falcon.src_id = self_id;
    }

    table set_src_id {
        actions = {act_set_src_id;}
        default_action = act_set_src_id(0);
    }

    apply {  
        set_src_id.apply();
    }
}

control ingress(inout headers hdr, inout metadata meta, inout standard_metadata_t standard_metadata) {
    register<bit<1>>((bit<32>) 1) switch_type; // To select switch behaviour (spine/leaf) from ctrl plane
    
    // List of idle workers up to 16 (idle workers) * 64 (clusters) 
    // Value 0x00 means Not-valid (NULL)
    register<worker_id_t>((bit<32>) 1024) idle_list; 
    register<bit<HDR_SRC_ID_SIZE>>((bit<32>) MAX_VCLUSTERS) idle_count; // Idle count for each cluster, acts as pointer going frwrd and backwrd to point to idle worker list

    // Spine only registers:
    // Holds length of the queue lengths vector 
    register<bit<HDR_SRC_ID_SIZE>>((bit<32>) MAX_VCLUSTERS) queue_len_count;
    // To map between queue_lens avaialable and id of the switch
    register<bit<HDR_SRC_ID_SIZE>>((bit<32>) 1024) queue_len_switch_id;

    register<queue_len_t>((bit <32>) 1024) queue_len_list; // List of queue lens 8 (workers) * 128 (clusters)
    register<queue_len_t>((bit <32>) MAX_VCLUSTERS) aggregate_queue_len_list;

    // Leaf switch registers
    // These registers are used for stateful probing, ToR sends first probes fills these
    register<bit<16>>((bit<32>) MAX_VCLUSTERS) linked_iq_sched; // Spine that ToR has sent last IdleSignal.
    register<bit<16>>((bit<32>) MAX_VCLUSTERS) linked_sq_sched; // Spine that ToR has sent last QueueSignal.
    register<queue_len_t>((bit<32>) MAX_VCLUSTERS) spine_iq_len_1;
    register<worker_id_t>((bit<32>) MAX_VCLUSTERS) spine_probed_id;
    //register<worker_id_t>((bit<32>) MAX_VCLUSTERS) spine_sw_id_2;

    //register<bit<16>>((bit <32>) 1024) workers_per_cluster;
    //register<bit<16>>((bit <32>) 1024) spines_per_cluster;

    action clone_packet() {
        // Clone from ingress to egress pipeline
        // secnod param (mirror session ID) maps to a specific output port
        clone(CloneType.I2E, (bit<32>)standard_metadata.egress_spec);

    }

    action _drop() {
        mark_to_drop(standard_metadata);
    }

    action act_get_switch_type() {
        switch_type.read(meta.falcon_meta.switch_type, (bit<32>) 0);
    }
    
    action act_gen_random_probe_group() {
        /* 
        TODO: Use modify_field_rng_uniform instead of random<> for hardware targets
        This is not implemented in bmv but available in Hardware. 
        //modify_field_rng_uniform(meta.falcon_meta.rand_probe_group, 0, RAND_MCAST_RANGE);
        */
        

        // Note: mcast group 0 means "do not multicast" (in bmv2)
        random<bit<HDR_FALCON_RAND_GROUP_SIZE>>(meta.falcon_meta.rand_probe_group, 1, RAND_MCAST_RANGE);
    }

    action act_set_queue_len_unit(len_fixed_point_t cluster_unit){
        meta.falcon_meta.queue_len_unit = cluster_unit;
    }

    action mac_forward(port_id_t port) {
        standard_metadata.egress_spec = port;
    }

    action act_gen_random_worker_id_1() {
        //modify_field_rng_uniform(meta.falcon_meta.random_downstream_id_1, 0, meta.falcon_meta.cluster_num_valid_ds);
        // Note: ranges for random in bmv2 are inclusive (reason to decreament upper bound) 
        random<bit<HDR_SRC_ID_SIZE>>(meta.falcon_meta.random_downstream_id_1, 0, meta.falcon_meta.cluster_num_valid_ds - 1);
        meta.falcon_meta.random_downstream_id_1 = meta.falcon_meta.random_downstream_id_1 + meta.falcon_meta.cluster_worker_start_idx;
    }

    action act_gen_random_worker_id_2() {
        //modify_field_rng_uniform(meta.falcon_meta.random_downstream_id_2, 0, meta.falcon_meta.cluster_num_valid_ds);
        // Note: ranges for random in bmv2 are inclusive (reason to decreament upper bound)
        random<bit<HDR_SRC_ID_SIZE>>(meta.falcon_meta.random_downstream_id_2, 0, meta.falcon_meta.cluster_num_valid_ds - 1);
        meta.falcon_meta.random_downstream_id_2 = meta.falcon_meta.random_downstream_id_2 + meta.falcon_meta.cluster_worker_start_idx;
    }

    action act_spine_gen_random_downstream_id_1() {
        bit<HDR_SRC_ID_SIZE> random_index;
        random<bit<HDR_SRC_ID_SIZE>>(random_index, 0, meta.falcon_meta.cluster_num_avail_queue - 1);
        // Map index to switch ID 
        queue_len_switch_id.read(meta.falcon_meta.random_downstream_id_1, (bit<32>) (random_index + meta.falcon_meta.cluster_worker_start_idx));
    }

    action act_spine_gen_random_downstream_id_2() {
        bit<HDR_SRC_ID_SIZE> random_index;
        random<bit<HDR_SRC_ID_SIZE>>(random_index, 0, meta.falcon_meta.cluster_num_avail_queue - 1);
        // Map index to switch ID 
        queue_len_switch_id.read(meta.falcon_meta.random_downstream_id_2, (bit<32>) (random_index + meta.falcon_meta.cluster_worker_start_idx));
    }

    action act_get_cluster_num_valid_ds(bit<16> num_ds_elements) {
        meta.falcon_meta.cluster_num_valid_ds = num_ds_elements;
    }

    // Get number of queue len signal available at spine
    action act_spine_get_cluster_num_avail_queues() {  
        queue_len_count.read(meta.falcon_meta.cluster_num_avail_queue, (bit<32>) hdr.falcon.cluster_id);
    }

    action act_spine_add_queue_len () {
        // Read, increement write back the register (stateful ALU operation in Tofino ?)
        queue_len_count.read(meta.falcon_meta.cluster_num_avail_queue, (bit<32>) hdr.falcon.cluster_id);
        queue_len_count.write((bit<32>) hdr.falcon.cluster_id, meta.falcon_meta.cluster_num_avail_queue + 1);
        // Calculate index to write new queue len
        // meta.falcon_meta.cluster_worker_start_idx = (bit <16>) (hdr.falcon.local_cluster_id * MAX_WORKERS_PER_CLUSTER);
        // Write new queue len info
        queue_len_list.write((bit<32>)(meta.falcon_meta.cluster_worker_start_idx + meta.falcon_meta.cluster_num_avail_queue), hdr.falcon.qlen);
        queue_len_switch_id.write((bit <32>) (meta.falcon_meta.cluster_worker_start_idx + meta.falcon_meta.cluster_num_avail_queue), hdr.falcon.src_id);
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

    action act_spine_read_idle_count() {
        idle_count.read(meta.falcon_meta.cluster_idle_count, (bit<32>) hdr.falcon.cluster_id);
        meta.falcon_meta.cluster_worker_start_idx = (bit <16>) (hdr.falcon.cluster_id * MAX_WORKERS_PER_CLUSTER);
        /* TODO: use "add_to_field()" for hardware targets, simply "+" in bvm */
        //meta.falcon_meta.cluster_idle_count = meta.falcon_meta.cluster_idle_count + 1;
        meta.falcon_meta.idle_worker_index = (bit <16>) meta.falcon_meta.cluster_idle_count + meta.falcon_meta.cluster_worker_start_idx - 1;
    
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
        
        idle_count.read(meta.falcon_meta.cluster_idle_count, (bit<32>) hdr.falcon.local_cluster_id);
        meta.falcon_meta.cluster_idle_count = meta.falcon_meta.cluster_idle_count - 1;
        idle_count.write((bit<32>) hdr.falcon.local_cluster_id, meta.falcon_meta.cluster_idle_count);
        //add_to_field(meta.falcon_meta.idle_worker_index, -1);
    }

    action act_spine_pick_top_idle_list () {
        idle_list.read(hdr.falcon.dst_id, (bit<32>) meta.falcon_meta.idle_worker_index);
    }

    action act_spine_pop_from_idle_list () {
        meta.falcon_meta.idle_worker_index = meta.falcon_meta.idle_worker_index - 1;

        idle_count.write((bit<32>) hdr.falcon.cluster_id, meta.falcon_meta.cluster_idle_count - 1);
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

    action act_get_random_qlen() {
        // if (meta.falcon_meta.random_downstream_id_1 == meta.falcon_meta.random_downstream_id_2){
        //     meta.falcon_meta.selected_downstream_id = meta.falcon_meta.random_downstream_id_1;
        // }
        queue_len_list.read(meta.falcon_meta.qlen_rand_1, (bit<32>) meta.falcon_meta.random_downstream_id_1);
        queue_len_list.read(meta.falcon_meta.qlen_rand_2, (bit<32>) meta.falcon_meta.random_downstream_id_2);
    }

    action act_increment_queue_len() {
        queue_len_list.read(meta.falcon_meta.qlen_curr, (bit<32>)hdr.falcon.dst_id);
        meta.falcon_meta.qlen_curr = meta.falcon_meta.qlen_curr + meta.falcon_meta.queue_len_unit;
        queue_len_list.write((bit<32>)hdr.falcon.dst_id, meta.falcon_meta.qlen_curr);  

        aggregate_queue_len_list.read(meta.falcon_meta.qlen_agg, (bit<32>) hdr.falcon.local_cluster_id);
        meta.falcon_meta.qlen_agg = meta.falcon_meta.qlen_agg + meta.falcon_meta.queue_len_unit;
        aggregate_queue_len_list.write((bit<32>) hdr.falcon.local_cluster_id, meta.falcon_meta.qlen_agg);       
    }

    action act_get_aggregate_queue_len() {
        aggregate_queue_len_list.read(hdr.falcon.qlen, (bit<32>) hdr.falcon.local_cluster_id);
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

    action act_read_linked_iq() {
        linked_iq_sched.read(meta.falcon_meta.linked_iq_id, (bit<32>) hdr.falcon.local_cluster_id);
    }

    action act_update_linked_sq() {
        linked_sq_sched.write((bit<32>) hdr.falcon.local_cluster_id, hdr.falcon.src_id);
        meta.falcon_meta.linked_sq_id = hdr.falcon.src_id;
    }

    action broadcast() {
        standard_metadata.mcast_grp = MCAST_ID_BROADCAST;
        meta.ingress_metadata.nhop_ipv4 = hdr.ipv4.dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl + 8w255;
    }

    action act_spine_select_random_leaf() {
        random<bit<HDR_SRC_ID_SIZE>>(hdr.falcon.dst_id, 0, meta.falcon_meta.cluster_num_valid_ds - 1);
    }

    table get_switch_type {
        actions = {act_get_switch_type;}
        default_action = act_get_switch_type;
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
        actions = {act_gen_random_probe_group;}
        default_action = act_gen_random_probe_group;
    }

    table gen_random_downstream_id_1 {
        actions = {act_gen_random_worker_id_1;}
        default_action = act_gen_random_worker_id_1;
    }

    table gen_random_downstream_id_2 {
        actions = {act_gen_random_worker_id_2;}
        default_action = act_gen_random_worker_id_2;
    }

    table spine_gen_random_downstream_id_1 {
        actions = {act_spine_gen_random_downstream_id_1;}
        default_action = act_spine_gen_random_downstream_id_1;
    }

    table spine_gen_random_downstream_id_2 {
        actions = {act_spine_gen_random_downstream_id_2;}
        default_action = act_spine_gen_random_downstream_id_2;
    }

    table get_aggregate_queue_len {
        actions = {act_get_aggregate_queue_len;}
        default_action = act_get_aggregate_queue_len;
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

    table spine_get_cluster_num_valid_ds {
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

    table spine_get_cluster_num_avail_queues {
        actions = {act_spine_get_cluster_num_avail_queues;}
        default_action = act_spine_get_cluster_num_avail_queues;
    }

    table spine_add_queue_len {
        actions = {act_spine_add_queue_len;}
        default_action = act_spine_add_queue_len;
    }

    table read_idle_count {
        actions = {act_read_idle_count;}
        default_action = act_read_idle_count;
    }

    table spine_read_idle_count {
        actions = {act_spine_read_idle_count;}
        default_action = act_spine_read_idle_count;
    }

    table add_to_idle_list {
        actions = {act_add_to_idle_list;}
        default_action = act_add_to_idle_list;
    }

    table spine_add_to_idle_list {
        actions = {act_add_to_idle_list;}
        default_action = act_add_to_idle_list;
    }

    table pop_from_idle_list {
        actions = {act_pop_from_idle_list;}
        default_action = act_pop_from_idle_list;
    }

    table spine_pick_top_idle_list {
        actions = {act_spine_pick_top_idle_list;}
        default_action = act_spine_pick_top_idle_list;
    }

    table spine_pop_from_idle_list {
        actions = {act_spine_pop_from_idle_list;}
        default_action = act_spine_pop_from_idle_list;
    }

    // table get_worker_index {
    //     actions = {act_get_worker_index;}
    //     default_action = act_get_worker_index;
    // }

    table decrement_queue_len {
        actions = {act_decrement_queue_len;}
        default_action = act_decrement_queue_len;
    }

    table get_random_qlen {
        actions = {act_get_random_qlen;}
        default_action = act_get_random_qlen;
    }

    table spine_get_random_qlen {
        actions = {act_get_random_qlen;}
        default_action = act_get_random_qlen;
    }

    // Currently uses the ID of workers to forward.
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

    // Mapping between switch ID and port for each switch
    // TODO: Spine switch e.g Si could be in path of packets that are not destined for Si 
    //  Should check dst_id and then use this table to forward to correct destination
    table forward_falcon_switch_dst {
        key = {
            hdr.falcon.dst_id: exact;
        }
        actions = {
            act_forward_falcon;
            NoAction;
        }
        size = HDR_SRC_ID_SIZE;
        default_action = NoAction;
    }

    table forward_falcon_switch_dst_2 {
        key = {
            hdr.falcon.dst_id: exact;
        }
        actions = {
            act_forward_falcon;
            NoAction;
        }
        size = HDR_SRC_ID_SIZE;
        default_action = NoAction;
    }

    // Used when making a clone packet from ingress to egress
    // Maps dst_id to port
    table spine_forward_falcon_early {
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

    // Maps dst_id to port
    table spine_forward_falcon {
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

    table read_linked_iq {
        actions = {act_read_linked_iq;}
        default_action = act_read_linked_iq;
    }

    table update_linked_sq {
        actions = {act_update_linked_sq;}
        default_action = act_update_linked_sq;
    }

    table spine_select_random_leaf {
        actions = {act_spine_select_random_leaf;}
        default_action = act_spine_select_random_leaf;
    }

    table spine_gen_random_probe_group {
        actions = {act_gen_random_probe_group;}
        default_action = act_gen_random_probe_group;
    }

    // ********* Normal switch forwarding BEGIN ********
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
    // ********* END Normal switch forwarding ********


    apply {
        if (hdr.falcon.isValid()) {
            // TODO: Optimization (or is this done by Tofino compiler)? these tables do not apply for all of the packets
            
            get_switch_type.apply();
            if (meta.falcon_meta.switch_type == 0) { // Leaf Switch
                read_idle_count.apply();
            
                read_linked_sq.apply();
                set_queue_len_unit.apply();
            
                if (hdr.falcon.pkt_type == PKT_TYPE_TASK_DONE_IDLE || hdr.falcon.pkt_type == PKT_TYPE_TASK_DONE) { // Only leaf switch
                    decrement_queue_len.apply();

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

                    if (meta.falcon_meta.linked_sq_id != 0xFF) { // not Null. TODO: fix Null value 0xFF port is valid
                        clone_packet(); // Make a new clone for PKT_TYPE_QUEUE_SIGNAL 
                        hdr.falcon.pkt_type = PKT_TYPE_QUEUE_SIGNAL;
                        hdr.falcon.qlen = meta.falcon_meta.qlen_agg; // Reporting agg qlen to Spine
                        hdr.falcon.dst_id = meta.falcon_meta.linked_sq_id;
                        forward_falcon_switch_dst.apply(); // Set egress point based on switch ID
                    }
                    
                } else if(hdr.falcon.pkt_type == PKT_TYPE_NEW_TASK) {
                    /*
                     TODO @parham: Remove the ToR from linked_iq spine if the packet is coming based on random decision
                     This needs a copy of packet (original task) to go to the server and another copy (ctrl msg) to go to linked_iq
                    */
                    if (meta.falcon_meta.cluster_idle_count > 0) { //Idle workers available
                        pop_from_idle_list.apply();
                    } else {
                        get_cluster_num_valid_ds.apply();
                        gen_random_downstream_id_1.apply();
                        gen_random_downstream_id_2.apply();
                        get_random_qlen.apply();
                        if (meta.falcon_meta.qlen_rand_1 >= meta.falcon_meta.qlen_rand_2) {
                            hdr.falcon.dst_id = meta.falcon_meta.random_downstream_id_2;
                        } else {
                            hdr.falcon.dst_id = meta.falcon_meta.random_downstream_id_1;
                        }
                    }
                    // TODO: Optimize this, needs to read the current queue lenght again (once read by previous actions)
                    // TODO: How to increment queue_len at spine? Should know about vcluster unit for all racks?
                    increment_queue_len.apply(); 
                    forward_falcon.apply();
                    read_linked_iq.apply();

                    // Rack not idle anymore after this assignment
                    // TODO: Currently, leaf will only send PKT_TYPE_IDLE_REMOVE when there was some linked IQ 
                    //  This means that for task that is coming because of random decision we are not sending Idle remove
                    //  This limitation is because spine can not iterate over the idle list to remove a switch at the middle it has a stack and can pop only!
                    if (meta.falcon_meta.cluster_idle_count == 0 && meta.falcon_meta.linked_iq_id != 0xFF) { 
                        clone_packet(); // Clone task packet and send to egress port (map based on "dst_id" which is set by forward_falcon)
                        linked_iq_sched.write((bit <32>) hdr.falcon.local_cluster_id, (bit<16>) 0xFF); // Set to NULL
                        // Reply to the spine with Idle remove
                        hdr.falcon.pkt_type = PKT_TYPE_IDLE_REMOVE;
                        standard_metadata.egress_spec = standard_metadata.ingress_port;
                    }
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
                        get_aggregate_queue_len.apply(); // Reporting agg qlen to Spine
                        update_linked_sq.apply();
                        hdr.falcon.pkt_type = PKT_TYPE_QUEUE_SIGNAL;
                        hdr.falcon.dst_id = meta.falcon_meta.linked_sq_id;
                        forward_falcon_switch_dst_2.apply(); // Set egress point based on switch ID
                        //standard_metadata.egress_spec = (bit<9>) meta.falcon_meta.linked_sq_id;
                    }
                } else {
                    mark_to_drop(standard_metadata);
                    }
            } else { // Spine switch
                spine_read_idle_count.apply();
                if(hdr.falcon.pkt_type == PKT_TYPE_NEW_TASK) {
                    if (meta.falcon_meta.cluster_idle_count > 0) { //Idle workers available
                        spine_pick_top_idle_list.apply();
                    } else {
                        spine_get_cluster_num_valid_ds.apply();
                        spine_get_cluster_num_avail_queues.apply();
                        if (meta.falcon_meta.cluster_num_avail_queue < 2) { // Not enough info at spine
                            spine_select_random_leaf.apply(); // Get ID of a random leaf from all available leafs
                            spine_forward_falcon_early.apply(); // Map ID to egress port
                            clone_packet(); // Clone that packet based on egress port (set by spine_forward_falcon_early)
                            spine_gen_random_probe_group.apply();
                            hdr.falcon.pkt_type = PKT_TYPE_SCAN_QUEUE_SIGNAL; // Multicast Scan packet to k connected ToRs
                            standard_metadata.mcast_grp = (bit <16>) meta.falcon_meta.rand_probe_group;
                        } else {
                            spine_gen_random_downstream_id_1.apply();
                            spine_gen_random_downstream_id_2.apply();
                            spine_get_random_qlen.apply();
                            if (meta.falcon_meta.qlen_rand_1 >= meta.falcon_meta.qlen_rand_2) {
                                hdr.falcon.dst_id = meta.falcon_meta.random_downstream_id_2;
                            } else {
                                hdr.falcon.dst_id = meta.falcon_meta.random_downstream_id_1;
                            }
                        }
                    }
                    spine_forward_falcon.apply();
                } else if (hdr.falcon.pkt_type == PKT_TYPE_IDLE_SIGNAL) {
                    if (meta.falcon_meta.cluster_idle_count < MAX_IDLE_WORKERS_PER_CLUSTER) {
                        spine_add_to_idle_list.apply();
                        queue_len_count.write((bit<32>) hdr.falcon.cluster_id, 0); // Reset length mappings
                        hdr.falcon.pkt_type = PKT_TYPE_QUEUE_REMOVE; // packet to unpair the leaf switches
                        broadcast(); // Send to all downstream links. 
                        // TODO: Instead of broadcast we need to send this to the ones currently in queue list
                        // without broadcast, spine needs to maintain state about the ToRs in queue list (e.g their IDs or IP etc.)
                    }
                } else if (hdr.falcon.pkt_type == PKT_TYPE_QUEUE_SIGNAL) {
                    spine_add_queue_len.apply();
                } else if (hdr.falcon.pkt_type == PKT_TYPE_PROBE_IDLE_QUEUE) {
                    hdr.falcon.pkt_type = PKT_TYPE_PROBE_IDLE_RESPONSE;
                    hdr.falcon.qlen = (bit <8>) meta.falcon_meta.cluster_idle_count; // TODO: 8bit sufficient for cluster_idle_count 
                    hdr.falcon.dst_id = hdr.falcon.src_id;
                } else if (hdr.falcon.pkt_type == PKT_TYPE_IDLE_REMOVE) {
                    // TODO: For now, leaf will only send back IDLE_REMOVE to the spine as reply.
                    // Sometimes, We need to remove the switch with <src_id> from the idle list of the linked_iq as a result of random tasks.
                    //  but don't have access to its index at spine. 
                    spine_pop_from_idle_list.apply();
                }
        } 
    } else if (hdr.ipv4.isValid()) {
        // Apply regular switch procedure
        
        ipv4_lpm.apply();
        forward.apply();
        
        }
    }
}
V1Switch(ParserImpl(), verifyChecksum(), ingress(), egress(), computeChecksum(), DeparserImpl()) main;
