// Actions run in the egress of the INT Sink, before packet is forwarded to end-host
control process_int_sink (
    inout headers hdr,
    inout local_metadata_t local_metadata,
    inout standard_metadata_t standard_metadata) {
    
    action restore_header () {
        // Restore fields of IPv4 length and UDP length and IPv4 DSCP which was stored in the INT Shim Header
        hdr.ipv4.dscp = hdr.intl4_shim.original_dscp;
        hdr.ipv4.len = hdr.ipv4.len - INT_TOTAL_HEADER_SIZE;
        if(hdr.udp.isValid()) {
            hdr.udp.length_ = hdr.udp.length_ - INT_TOTAL_HEADER_SIZE;
        }
        
    }
    
    action int_sink() {
        // Remove all the INT information from the packet
        hdr.intl4_shim.setInvalid();
        hdr.int_header.setInvalid();
    }

    apply {
        restore_header();
        int_sink();
    }
}

// Action for the INT telemetry report, as it is encapsulated with new Ethernet, IP and UDP headers
control process_int_report (inout headers hdr,
                            inout local_metadata_t local_metadata,
                            inout standard_metadata_t standard_metadata) {

    action do_report_encapsulation(mac_t src_mac,
                                   mac_t mon_mac, 
                                   ip_address_t src_ip,
                                   ip_address_t mon_ip, 
                                   l4_port_t src_port, 
                                   l4_port_t mon_port) {
        
        // Report Ethernet Header
        hdr.report_ethernet.setValid();
        hdr.report_ethernet.dst_addr = mon_mac;
        hdr.report_ethernet.src_addr = src_mac;
        hdr.report_ethernet.ether_type = 0x0800;

        // Report IPV4 Header
        hdr.report_ipv4.setValid();
        hdr.report_ipv4.version = 4w4;
        hdr.report_ipv4.ihl = 4w5;
        hdr.report_ipv4.dscp = 6w0;
        hdr.report_ipv4.ecn = 2w0;
        
        if(hdr.udp.isValid()) {
            hdr.report_ipv4.len = (bit<16>) IPV4_MIN_HEAD_LEN + (bit<16>) UDP_HEADER_LEN + (bit<16>) REPORT_GROUP_HEADER_LEN + (bit<16>) REPORT_INDIVIDUAL_HEADER_LEN + 
            local_metadata.int_meta.new_bytes + (bit<16>) IPV4_MIN_HEAD_LEN + (bit<16>) UDP_HEADER_LEN + (bit<16>) INT_TOTAL_HEADER_SIZE;
        } else if (hdr.tcp.isValid()) {
            hdr.report_ipv4.len = (bit<16>) IPV4_MIN_HEAD_LEN + (bit<16>) UDP_HEADER_LEN + (bit<16>) REPORT_GROUP_HEADER_LEN + (bit<16>) REPORT_INDIVIDUAL_HEADER_LEN + 
            local_metadata.int_meta.new_bytes + (bit<16>) IPV4_MIN_HEAD_LEN + (bit<16>) TCP_HEADER_LEN + (bit<16>) INT_TOTAL_HEADER_SIZE;
        }

        hdr.report_ipv4.protocol = IP_PROTO_UDP;
        hdr.report_ipv4.identification = 0;
        hdr.report_ipv4.flags = 0;
        hdr.report_ipv4.frag_offset = 0;
        hdr.report_ipv4.ttl = 64;
        hdr.report_ipv4.src_addr = src_ip;
        hdr.report_ipv4.dst_addr = mon_ip;

        // Report UDP Header
        hdr.report_udp.setValid();
        hdr.report_udp.src_port = src_port;
        hdr.report_udp.dst_port = mon_port;
        if (hdr.udp.isValid()) {
            hdr.report_udp.length_ = (bit<16>) UDP_HEADER_LEN + (bit<16>) REPORT_GROUP_HEADER_LEN + (bit<16>) REPORT_INDIVIDUAL_HEADER_LEN + 
            local_metadata.int_meta.new_bytes + (bit<16>) IPV4_MIN_HEAD_LEN + (bit<16>) UDP_HEADER_LEN + (bit<16>) INT_TOTAL_HEADER_SIZE;
        } else if (hdr.tcp.isValid()) {
            hdr.report_udp.length_ = (bit<16>) UDP_HEADER_LEN + (bit<16>) REPORT_GROUP_HEADER_LEN + (bit<16>) REPORT_INDIVIDUAL_HEADER_LEN + 
            local_metadata.int_meta.new_bytes + (bit<16>) IPV4_MIN_HEAD_LEN + (bit<16>) TCP_HEADER_LEN + (bit<16>) INT_TOTAL_HEADER_SIZE;
        }

        // Telemetry Group Header 
        hdr.report_group_header.setValid();
        hdr.report_group_header.ver = 2;                // Version 2.0
        hdr.report_group_header.hw_id = HW_ID;          // Default Value 1
        hdr.report_group_header.seq_no = 0;             // Default Value 0
        hdr.report_group_header.node_id = local_metadata.int_meta.node_id;

        // Telemetry Report Individual Header
        hdr.report_individual_header.setValid();
        hdr.report_individual_header.rep_type = 1;      // INT Report
        hdr.report_individual_header.in_type = 4;       // Individual Report Inner Content is an IPv4 packet
        if (hdr.udp.isValid()) {
            hdr.report_individual_header.len = 2 + local_metadata.int_meta.new_words + 5 + 2 + 4;
        } else if (hdr.tcp.isValid()) {
            hdr.report_individual_header.len = 2 + local_metadata.int_meta.new_words + 5 + 5 + 4;
        }
        hdr.report_individual_header.rep_md_len = local_metadata.int_meta.new_words;
        hdr.report_individual_header.d = 0;
        hdr.report_individual_header.q = 0;
        hdr.report_individual_header.f = 1;
        hdr.report_individual_header.i = 0;
        hdr.report_individual_header.rsvd = 0;

        // Individual report main contents
        hdr.report_individual_header.rep_md_bits = hdr.int_header.instruction_mask_0003 ++ hdr.int_header.instruction_mask_0407 ++
                                                     hdr.int_header.instruction_mask_0811 ++ hdr.int_header.instruction_mask_1215;
        hdr.report_individual_header.domain_specific_id = 0;
        hdr.report_individual_header.domain_specific_md_bits = 0;
        hdr.report_individual_header.domain_specific_md_status = 0;

        hdr.ethernet.setInvalid();
    }

    table tb_generate_report {
        actions = {
            do_report_encapsulation;
            NoAction();
        }
        default_action = do_report_encapsulation(0, 0, 0, 0, 0, 0);
    }

    apply {
        tb_generate_report.apply();
    }
}