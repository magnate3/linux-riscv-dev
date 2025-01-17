/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

#include "include/headers.p4"
#include "include/parsers.p4"
#include "include/checksum.p4"
#include "include/sink.p4"
#include "include/transit.p4"
#include "include/source.p4"
#include "include/forward.p4"

/********************************************************************
**************** I N G R E S S   P R O C E S S I N G ****************
********************************************************************/

control MyIngress(inout headers hdr, 
                  inout local_metadata_t local_metadata, 
                  inout standard_metadata_t standard_metadata) {

    apply {
        if(hdr.ipv4.isValid()) {

            // Perform L2 forwarding based on IPv4 destination address
            l3_forward.apply(hdr, local_metadata, standard_metadata);

            if (hdr.udp.isValid() || hdr.tcp.isValid()) {
                    
                // Setting source and sink local metadata
                process_int_source_sink.apply(hdr, local_metadata, standard_metadata);

                // In case of source add the INT header
                if (local_metadata.int_meta.source == true) {
                    process_int_source.apply(hdr, local_metadata);
                }

                // Clone packet for Telemetry Report
                if (hdr.int_header.isValid()) {
                    local_metadata.perserv_meta.ingress_port = standard_metadata.ingress_port;
                    local_metadata.perserv_meta.egress_port = standard_metadata.egress_port;
                    local_metadata.perserv_meta.deq_qdepth = standard_metadata.deq_qdepth;
                    local_metadata.perserv_meta.deq_timedelta = standard_metadata.deq_timedelta;
                    local_metadata.perserv_meta.ingress_global_timestamp = standard_metadata.ingress_global_timestamp;
                    local_metadata.perserv_meta.egress_global_timestamp = standard_metadata.egress_global_timestamp;
                    clone_preserving_field_list(CloneType.I2E, REPORT_MIRROR_SESSION_ID, 1);

                }
            }
        }
    }
}

/********************************************************************
***************** E G R E S S   P R O C E S S I N G *****************
********************************************************************/

control MyEgress(inout headers hdr,
                 inout local_metadata_t local_metadata,
                 inout standard_metadata_t standard_metadata) {
    
    apply {
        // Insert old packet metadata into the cloned packet
        if(hdr.int_header.isValid() && standard_metadata.instance_type == CLONE_PKT) {
            standard_metadata.ingress_port = local_metadata.perserv_meta.ingress_port;
            standard_metadata.egress_port = local_metadata.perserv_meta.egress_port;
            standard_metadata.deq_qdepth = local_metadata.perserv_meta.deq_qdepth;
            standard_metadata.deq_timedelta = local_metadata.perserv_meta.deq_timedelta;
            standard_metadata.ingress_global_timestamp = local_metadata.perserv_meta.ingress_global_timestamp;
            standard_metadata.egress_global_timestamp = local_metadata.perserv_meta.egress_global_timestamp;
        }

        // In case of cloned packet, send telemetry report
        if (standard_metadata.instance_type == CLONE_PKT) {
            process_int_transit.apply(hdr, local_metadata, standard_metadata);
            process_int_report.apply(hdr, local_metadata, standard_metadata);
        }

        // In case of sink, remove INT header from original packet
        if (hdr.int_header.isValid() && local_metadata.int_meta.sink == true && standard_metadata.instance_type != CLONE_PKT) {
            process_int_sink.apply(hdr, local_metadata, standard_metadata);
        }
    }
}

/***********************************************************
*********************** S W I T C H ************************
***********************************************************/

V1Switch(
    MyIngressParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyEgressDeparser()
) main;