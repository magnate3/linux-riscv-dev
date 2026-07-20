#include <core.p4>
#include <tna.p4>

#include "common/headers.p4"
#include "common/util.p4"
#include "headers.p4"

#define SAQR_PORT 1234

parser SaqrIngressParser (
        packet_in pkt,
        out saqr_header_t hdr,
        out saqr_metadata_t saqr_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        saqr_md.linked_sq_id = INVALID_VALUE_16bit;
        saqr_md.queue_len_unit = 0;
        saqr_md.cluster_idle_count = 0;   
        saqr_md.idle_ds_index = 0;   
        saqr_md.worker_index = 0;  
        saqr_md.cluster_ds_start_idx=0;
        saqr_md.rand_probe_group = 0;
        saqr_md.aggregate_queue_len = 0;
        saqr_md.random_id_1 = 0;
        saqr_md.random_id_2 = 0;
        saqr_md.task_resub_hdr.qlen_1 = 0;
        saqr_md.deferred_qlen_1 = 0;
        saqr_md.last_probed_id = INVALID_VALUE_16bit;
        saqr_md.last_iq_len = INVALID_VALUE_16bit;
        saqr_md.idle_link = 0;
        pkt.extract(ig_intr_md);
        transition parse_resub_meta;
    }

    state parse_resub_meta {
        transition select (ig_intr_md.resubmit_flag) { // Assume only one resubmission type for now
            0: parse_port_meta; // Not resubmitted
            1: parse_resub_hdr; // Resubmitted packet
        }
    }

    // Header format: ig_intrinsic_md + phase0 (we skipped this part) + ETH/IP... OR ig_intrinsic_md + resubmit + ETH/IP.
    // So no need to call .advance (or skip) when extracting resub_hdr as by extracting, we are moving the pointer so next state starts at correct index
    // Note: actual resubmitted header will be 8bytes regardless of our task_resub_hdr size (padded by 0s)
    state parse_resub_hdr {
        pkt.extract(saqr_md.task_resub_hdr); // Extract data from previous pas
        transition parse_ethernet;
    }

    state parse_port_meta {
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select (hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : reject;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select (hdr.ipv4.protocol) {
            IP_PROTOCOLS_UDP : parse_udp;
            default : reject; 
        }
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition select (hdr.udp.dst_port) {
            SAQR_PORT : parse_saqr;
            default: accept;
        }
    }

    state parse_saqr {
        pkt.extract(hdr.saqr);
        transition accept;
    }
    
}


parser SpineIngressParser (
        packet_in pkt,
        out saqr_header_t hdr,
        out saqr_metadata_t saqr_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    // TofinoIngressParser() tofino_parser;

    state start {
        saqr_md.linked_sq_id = INVALID_VALUE_16bit;
        saqr_md.queue_len_unit = 0;
        saqr_md.cluster_idle_count = 0;   
        saqr_md.idle_ds_index = 0;   
        saqr_md.worker_index = 0;  
        saqr_md.cluster_ds_start_idx=0;
        saqr_md.rand_probe_group = 0;
        saqr_md.aggregate_queue_len = 0;
        saqr_md.random_id_1 = 0;
        saqr_md.random_id_2 = 0;
        saqr_md.task_resub_hdr.qlen_1 = 0;
        saqr_md.deferred_qlen_1 = 0;
        saqr_md.last_probed_id = INVALID_VALUE_16bit;
        saqr_md.last_iq_len = INVALID_VALUE_16bit;
        pkt.extract(ig_intr_md);
        transition parse_resub_meta;
    }

    state parse_resub_meta {
        transition select (ig_intr_md.resubmit_flag) { // Assume only one resubmission type for now
            0: parse_port_meta; // Not resubmitted
            1: parse_resub_hdr; // Resubmitted packet
        }
    }

    // Header format: ig_intrinsic_md + phase0 (we skipped this part) + ETH/IP... OR ig_intrinsic_md + resubmit + ETH/IP.
    // So no need to call .advance (or skip) when extracting resub_hdr as by extracting, we are moving the pointer so next state starts at correct index
    // Note: actual resubmitted header will be 8bytes regardless of our task_resub_hdr size (padded by 0s)
    state parse_resub_hdr {
        pkt.extract(saqr_md.task_resub_hdr); // Extract data from previous pas
        transition parse_ethernet;
    }

    state parse_port_meta {
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select (hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : reject;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select (hdr.ipv4.protocol) {
            IP_PROTOCOLS_UDP : parse_udp;
            default : reject; 
        }
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition select (hdr.udp.dst_port) {
            SAQR_PORT : parse_saqr;
            default: accept;
        }
    }

    state parse_saqr {
        pkt.extract(hdr.saqr);
        transition accept;
    }
    
}