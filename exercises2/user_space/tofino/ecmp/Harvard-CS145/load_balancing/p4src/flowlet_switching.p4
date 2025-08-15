/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

//My includes
#include "include/headers.p4"
#include "include/parsers.p4"

#define FLOW_TABLE_SIZE 8192
#define FLOWLET_TIME_GAP 50000

/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}

/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
    register<bit<8>>(FLOW_TABLE_SIZE) flow_to_hash_index;
    register<bit<48>>(FLOW_TABLE_SIZE) flow_to_timestamp;

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action fill_metadata(bit<8> group_num, bit<8> buckets_in_group) {
        meta.group_num = group_num;
        meta.buckets_in_group = buckets_in_group;
    }

    action forward(bit<9> egress_port) {
        standard_metadata.egress_spec = egress_port;
    }

    table dipv4 {
        key = {
            hdr.ipv4.dstAddr: exact;
        }

        actions = {
            forward;
            fill_metadata;
            drop;
            NoAction;
        }
        size = 256;
        default_action = NoAction;
    }

    table group_info_to_port  {
        key = {
            meta.group_num: exact;
            meta.hash_index: exact;
        }
        actions = {
            forward;
            drop;
            NoAction;
        }
        size = 256;
        default_action = NoAction;
    }

    apply {
        dipv4.apply();
        if (meta.group_num != 0)  {
            // Find flow index from your incoming packet
            /* Add your hash function here */
            /* ....... */

            // Compare your timestamp to previous timestamp stored in the switch
            bit<48> previous_timestamp;
            // read switch's last seen timestamp 
            /* Add your code here */
            /* ....... */

            // if this is determinted as a new flowlet...
            if (standard_metadata.ingress_global_timestamp - previous_timestamp > FLOWLET_TIME_GAP)  {
                // Create new flowlet id to calculate the hash_index,
                // and write hash_index to the corresponding switch register
                /* Add your code here */
                /* ....... */
                
            } else {
                // Read stored hash_index from the corresponding switch register
                /* Add your code here */
                /* ....... */
                
            }

            // Write timestamp to the corresponding switch register
            /* Add your code here */
            /* ....... */

            // Forward packet
            group_info_to_port.apply();
        }
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply {

    }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            { hdr.ipv4.version,
	          hdr.ipv4.ihl,
              hdr.ipv4.dscp,
              hdr.ipv4.ecn,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
              hdr.ipv4.hdrChecksum,
              HashAlgorithm.csum16);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

//switch architecture
V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;