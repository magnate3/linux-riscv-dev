// Forward the packet based on the IPv4 destination address
control l3_forward(inout headers hdr,
                       inout local_metadata_t local_metadata,
                       inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(mac_t dstAddr, mac_t srcAddr, port_t port) {
        standard_metadata.egress_spec = port;
        standard_metadata.egress_port = port;
        hdr.ethernet.src_addr = srcAddr;
        hdr.ethernet.dst_addr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }   

    table ipv4_lpm {
        key = {
            hdr.ipv4.dst_addr : lpm;
        }
        actions = {
            ipv4_forward;
            drop;
        }
        size = 1024;
        const default_action = drop();
    }

    apply {
        if(hdr.ipv4.ttl == 0) {
            drop();
        }
        if(hdr.ipv4.isValid()) {
            ipv4_lpm.apply();
        }
            
    }
}