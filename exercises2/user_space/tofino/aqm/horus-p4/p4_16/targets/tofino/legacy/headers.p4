#ifndef __HEADER_P4__
#define __HEADER_P4__ 1

#define NUM_SW_PORTS   48
#define NUM_LEAF_US    24
#define NUM_LEAF_DS    24
#define NUM_SPINE_DS   24
#define NUM_SPINE_US   24

#define PKT_TYPE_NEW_TASK 0
#define PKT_TYPE_NEW_TASK_RANDOM 1
#define PKT_TYPE_TASK_DONE 2
#define PKT_TYPE_TASK_DONE_IDLE 3
#define PKT_TYPE_QUEUE_REMOVE 4
#define PKT_TYPE_SCAN_QUEUE_SIGNAL 5
#define PKT_TYPE_IDLE_SIGNAL 6
#define PKT_TYPE_QUEUE_SIGNAL 7
#define PKT_TYPE_PROBE_IDLE_QUEUE 8
#define PKT_TYPE_PROBE_IDLE_RESPONSE 9
#define PKT_TYPE_IDLE_REMOVE 10
#define PKT_TYPE_KEEP_ALIVE 11
#define PKT_TYPE_WORKER_ID 12
#define PKT_TYPE_WORKER_ID_ACK 13
#define PKT_TYPE_REMOVE_ACK 14
#define PKT_TYPE_QUEUE_SIGNAL_INIT 15

#define PORT_PCI_CPU 192

#define HDR_PKT_TYPE_SIZE 8
#define HDR_CLUSTER_ID_SIZE 16
#define HDR_SRC_ID_SIZE 16
#define HDR_QUEUE_LEN_SIZE 16
#define HDR_SEQ_NUM_SIZE 16
#define HDR_saqr_RAND_GROUP_SIZE 8
#define HDR_saqr_DST_SIZE 8

#define QUEUE_LEN_FIXED_POINT_SIZE 16


#define MAX_VCLUSTERS 32
#define MAX_WORKERS_PER_CLUSTER 16
#define MAX_LEAFS_PER_CLUSTER 16

#define MAX_WORKERS_IN_RACK 256 
#define MAX_LEAFS 256

// This defines the maximum queue length signals (for each vcluster) that a single spine would maintain (MAX_LEAFS/L_VALUE)
#define MAX_LINKED_LEAFS 64 
// This is the total length of array (shared between vclusters) for tracking leaf queue lengths
#define MAX_TOTAL_LEAFS  256 

//#define ARRAY_SIZE 573500
#define ARRAY_SIZE 65536

#define MIRROR_TYPE_WORKER_RESPONSE 1
#define MIRROR_TYPE_NEW_TASK 2

#define RESUBMIT_TYPE_NEW_TASK 1
#define RESUBMIT_TYPE_IDLE_REMOVE 2


const bit<8> INVALID_VALUE_8bit = 8w0x7F;
const bit<16> INVALID_VALUE_16bit = 16w0x7FFF;

typedef bit<HDR_QUEUE_LEN_SIZE> queue_len_t;
typedef bit<9> port_id_t;
typedef bit<16> worker_id_t;
typedef bit<16> leaf_id_t;
typedef bit<16> switch_id_t;
typedef bit<QUEUE_LEN_FIXED_POINT_SIZE> len_fixed_point_t;

header empty_t {
}

header saqr_h {
    bit<HDR_PKT_TYPE_SIZE> pkt_type;
    bit<HDR_CLUSTER_ID_SIZE> cluster_id;
    bit<16> src_id;                 // workerID for ToRs. ToRID for spines.
    bit<16> dst_id;
    bit<HDR_QUEUE_LEN_SIZE> qlen;   // Also used for reporting length of idle list (from spine sw to leaf sw) and indicating the decision type for a scheduled task (Idle selection or not)
    bit<HDR_SEQ_NUM_SIZE> seq_num;   
}

struct saqr_header_t {
    ethernet_h ethernet;
    ipv4_h ipv4;
    udp_h udp;
    saqr_h saqr;
}


// We use the same resub header format for removal to avoid using additional parser resources
header task_resub_hdr_t {
    bit<16> ds_index_1; 
    bit<16> ds_index_2; 
    bit<HDR_QUEUE_LEN_SIZE> qlen_1;
    bit<HDR_QUEUE_LEN_SIZE> qlen_2;
}


// Empty metadata struct for empty egress blocks
struct eg_metadata_t {
    bit<32> egress_tstamp_clipped;
    bit<32> task_counter;
}

struct saqr_metadata_t {
    bit<1> idle_remove_lock;
    bit<HDR_SRC_ID_SIZE> linked_sq_id;
    bit<HDR_SRC_ID_SIZE> linked_iq_id;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> queue_len_unit; // (1/num_worekrs) for each vcluster
    bit<QUEUE_LEN_FIXED_POINT_SIZE> queue_len_unit_sample_1; // (1/num_worekrs) for each vcluster
    bit<QUEUE_LEN_FIXED_POINT_SIZE> queue_len_unit_sample_2; // (1/num_worekrs) for each vcluster
    bit<32> task_counter;
    bit<32> ingress_tstamp_clipped;
    bit<16> cluster_idle_count;
    bit<16> idle_ds_index;
    bit<16> worker_index;
    bit<16> cluster_ds_start_idx;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> aggregate_queue_len;
    MulticastGroupId_t rand_probe_group;
    bit<16> cluster_num_valid_ds;
    bit<16> cluster_num_valid_us;
    bit<16> cluster_num_valid_queue_signals;
    bit<16> cluster_num_valid_queue_signals_copy;
    bit<16> random_id_1;
    bit<16> random_id_2;
    bit<16> random_ds_index_1;
    bit<16> random_ds_index_2;
    bit<16> t1_random_ds_index_1;
    bit<16> t1_random_ds_index_2;
    bit<16> t2_random_ds_index_1;
    bit<16> t2_random_ds_index_2;
    bit<16> child_switch_index;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> worker_qlen_1;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> worker_qlen_2;

    bit<QUEUE_LEN_FIXED_POINT_SIZE> random_ds_qlen_1;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> random_ds_qlen_2;

    bit<QUEUE_LEN_FIXED_POINT_SIZE> selected_correct_qlen;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> not_selected_correct_qlen;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> min_correct_qlen;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> selected_ds_qlen;
    bit<16> selected_ds_index;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> not_selected_ds_qlen;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> queue_len_diff;

    bit<16> deferred_qlen_1;
    bit<16> cluster_absolute_leaf_index;
    bit<16> idle_ds_id;
    bit<16> selected_spine_iq_len;
    bit<16> last_iq_len;
    bit<16> last_probed_id;
    bit<16> spine_to_link_iq;
    bit<16> idle_link;
    bit<16> received_dst_id;
    bit<16> received_src_id;
    bit<16> virtual_leaf_id; // Used for virtualizing the single switch only in testbed (not for actual implementation)
    bit<16> num_additional_signal_needed;
    bit<16> cluster_max_linked_leafs;
    bit<16> mirror_dst_id; // Usage similar to hdr.dst_id but this is for mirroring
    bit<16> lid_ds_index;
    bit<16> idle_id_to_write;
    task_resub_hdr_t task_resub_hdr;
    bit<8> idle_len_8bit;
    bit<16> spine_view_ok;
    bit<16> idle_rr_index;
    bit<16> selected_idle_index;
    bit<16> last_new_index;
    bit<16> idle_remove_min_id;
    bit<16> linked_iq_view;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> qlen_unit_1;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> qlen_unit_2;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> selected_ds_qlen_unit;
    bit<QUEUE_LEN_FIXED_POINT_SIZE> not_selected_ds_qlen_unit;
}


#endif // __HEADER_P4__
