//Exactly one of the following must be defined when compiling this P4 program:
// VARIANT_PER_FLOW_VQ: Each flow has its own VQ
// VARIANT_PER_PORT_VQ: Each port has its own VQ
// VARIANT_NO_VQ: There is no VQ

#if defined(VARIANT_PER_FLOW_VQ) + defined(VARIANT_PER_PORT_VQ) + defined(VARIANT_NO_VQ) != 1
    #error "Exactly one source code variant must be defined"
#endif

control MyIngress(inout headers hdr, inout metadata meta, inout standard_metadata_t standard_metadata) {

    #if defined(VARIANT_PER_FLOW_VQ) || defined(VARIANT_PER_PORT_VQ)
        meter((bit<32>) (1 << VQ_ID_T_WIDTH), MeterType.packets) vq_packets;
    #endif

    action set_egress_port_and_mac(portId_t egress_port, macAddr_t dst_mac) {
        standard_metadata.egress_spec = egress_port;
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dst_mac;
        log_msg("Egress port set to {} and dst MAC to {}", {egress_port, dst_mac});
    }

    table l3_forward {
        key = { hdr.ipv4.dstAddr: exact; }
        actions = { set_egress_port_and_mac; }
        size = 256;
    }

    apply {
        if (hdr.ipv4.isValid()) {
            log_msg("Received IPv4: ingress={}; ttl={}; protocol={}; ecn={}; {}.{}.{}.{} -> {}.{}.{}.{}",
                    {standard_metadata.ingress_port, hdr.ipv4.ttl, hdr.ipv4.protocol, hdr.ipv4.ecn,
                    SLICE_IPV4_ADDRESS(hdr.ipv4.srcAddr), SLICE_IPV4_ADDRESS(hdr.ipv4.dstAddr)});
            if (!hdr.tcp.isValid() && !hdr.udp.isValid()) { log_msg("WARN: packet is neither TCP nor UDP"); }
        } else if (hdr.ethernet.etherType == ETHER_TYPE_IPV6) { //We sometimes receive IPv6 packets for some reason
            log_msg("Received IPv6; ingress={}; dropping", {standard_metadata.ingress_port});
            mark_to_drop(standard_metadata);
            return;
        } else { //We are unable to forward non-IPv4 Ethernet packets
            log_msg("ERROR: ether type {}; ingress={}; ethernet-src={}; ethernet-dst={}; dropping",
                    {hdr.ethernet.etherType, standard_metadata.ingress_port,
                    hdr.ethernet.srcAddr, hdr.ethernet.dstAddr});
            mark_to_drop(standard_metadata);
            return;
        }

        //Decrease TTL
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
        if (hdr.ipv4.ttl == 0) {
            log_msg("ERROR: TTL expired for destination {}", {SLICE_IPV4_ADDRESS(hdr.ipv4.dstAddr)});
            mark_to_drop(standard_metadata);
            return;
        }

        //Set the next hop
        if (l3_forward.apply().miss) {
            log_msg("FAIL: L3 forward table miss for destination {}", {SLICE_IPV4_ADDRESS(hdr.ipv4.dstAddr)});
            mark_to_drop(standard_metadata);
            return;
        }

        hash(meta.flow_id, HashAlgorithm.crc32, (bit<1>) 0, {
            hdr.ipv4.srcAddr,
            hdr.ipv4.dstAddr,
            hdr.ipv4.protocol,
            hdr.tcp.isValid() ? hdr.tcp.srcPort : (hdr.udp.isValid() ? hdr.udp.srcPort : 0),
            hdr.tcp.isValid() ? hdr.tcp.dstPort : (hdr.udp.isValid() ? hdr.udp.dstPort : 0)
        }, (bit<32>) (1 << FLOW_ID_T_WIDTH));

        #ifdef VARIANT_NO_VQ
            meta.vq_id = 0; //No VQ -> use a dummy value
        #else
            //Validate that the egress port is within the supported range
            if ((portId_t) ((small_port_t) standard_metadata.egress_spec) != standard_metadata.egress_spec) {
                log_msg("FAIL: Egress port is out of range: {}", {standard_metadata.egress_spec});
                mark_to_drop(standard_metadata);
                return;
            }

            //Calculate the VQ ID
            #ifdef VARIANT_PER_FLOW_VQ
                //Each (port, flow) pairs gets it own VQ
                meta.vq_id = ((small_port_t) standard_metadata.egress_spec) ++ meta.flow_id;
            #else
                //Each port gets its own VQ - the flow ID is ignored
                meta.vq_id = (vq_id_t) standard_metadata.egress_spec;
            #endif

            //Determine how congested the VQ is
            meter_color_t color = METER_INVALID;
            vq_packets.execute_meter((bit<32>) meta.vq_id, color);
            log_msg("Meter color of VQ={}: {}", {meta.vq_id, color});

            //Apply ECN if VQ is congested
            if (color == METER_GREEN) {
                //Do nothing: just let the packet be forwarded
            } else if (color == METER_YELLOW) {
                if (hdr.ipv4.ecn == 0) { log_msg("WARN: hosts don't support ECN"); }
                hdr.ipv4.ecn = 3; //Set ECN to 11
            } else if (color == METER_RED) {
                mark_to_drop(standard_metadata);
                return;
            } else {
                log_msg("ERROR: Unknown meter color={} for VQ={}", {color, meta.vq_id});
            }
        #endif
    }
}

control MyEgress(inout headers hdr, inout metadata meta, inout standard_metadata_t standard_metadata) {
    apply {
        //Log data that we can later use to create plots
        log_msg("Egress data: timestamp={}; ingress_port={}; egress_port={}; flow_id={}; vq_id={}; dequeue_timedelta={}; packet_length={}",
                {standard_metadata.egress_global_timestamp, standard_metadata.ingress_port, standard_metadata.egress_port,
                meta.flow_id, meta.vq_id, standard_metadata.deq_timedelta, standard_metadata.packet_length});
    }
}
