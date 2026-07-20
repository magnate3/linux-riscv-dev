



```
    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol, hdr.ipv4.frag_offset) {
            (IP_PROTOCOLS_ICMP, 0) : parse_icmp;
            (IP_PROTOCOLS_IGMP, 0) : parse_igmp;
            (IP_PROTOCOLS_TCP, 0) : parse_tcp;
            (IP_PROTOCOLS_UDP, 0) : parse_udp;
            // Do NOT parse the next header if IP packet is fragmented.
            default : accept;
        }
    }
```




```
    state accept_non_switchml {
        ig_md.switchml_md.setValid();
        ig_md.switchml_md = switchml_md_initializer;
        ig_md.switchml_md.packet_type = packet_type_t.IGNORE; // assume non-SwitchML packet
        transition accept;
    }
   state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        ipv4_checksum.add(hdr.ipv4);
        ig_md.checksum_err_ipv4 = ipv4_checksum.verify();
        ig_md.update_ipv4_checksum = false;
        
        // parse only non-fragmented IP packets with no options
        transition select(hdr.ipv4.ihl, hdr.ipv4.frag_offset, hdr.ipv4.protocol) {
            (5, 0, ip_protocol_t.ICMP) : parse_icmp;
            (5, 0, ip_protocol_t.UDP)  : parse_udp;
            default                    : accept_non_switchml;
        }
    }
```

# frag_offset

```
control IngressControlFrag(inout headers hdr,
                           inout ingress_metadata_t ig_md,
                           in ingress_intrinsic_metadata_t ig_intr_md) {


    apply {
        ig_md.layer3_frag = 0;
        if (hdr.ipv4.isValid()) {
            if ((hdr.ipv4.flags&1) != 0) ig_md.layer3_frag = 1;
            if (hdr.ipv4.frag_offset != 0) ig_md.layer3_frag = 1;
        } else if (hdr.ipv6.isValid()) {
            if (hdr.ipv6.next_hdr == IP_PROTOCOL_IPV6_FRAG) ig_md.layer3_frag = 1;
        }
#ifdef HAVE_SRV6
        if (hdr.ipv4b.isValid()) {
            if ((hdr.ipv4b.flags&1) != 0) ig_md.layer3_frag = 1;
            if (hdr.ipv4b.frag_offset != 0) ig_md.layer3_frag = 1;
        } else if (hdr.ipv6b.isValid()) {
            if (hdr.ipv6b.next_hdr == IP_PROTOCOL_IPV6_FRAG) ig_md.layer3_frag = 1;
        }
#endif
    }

}
```


```
    apply {
        ig_md.natted = 0;
        if (ig_md.ipv4_valid==1)  {
            if (!tbl_ipv4_nat_trns.apply().hit) {
                tbl_ipv4_nat_cfg.apply();
            }
        }
        if (ig_md.ipv6_valid==1)  {
            if (!tbl_ipv6_nat_trns.apply().hit) {
                tbl_ipv6_nat_cfg.apply();
            }
        }
        if ((ig_md.natted != 0) && (ig_md.layer3_frag != 0)) {
            ig_md.dropping = 2;
        }
    }
```

+ mark_to_drop and  ig_intr_md.egress_spec = CPU_PORT
```
        if (ig_md.dropping == 1) {
            mark_to_drop(ig_intr_md);
            return;
        }
        if (ig_md.dropping == 2) {
            hdr.cpu.setValid();
            hdr.cpu.port = ig_md.ingress_id;
            ig_intr_md.egress_spec = CPU_PORT;
            ig_md.punting = 1;
            return;
        }
        ig_ctl_qos_in.apply(hdr,ig_md,ig_intr_md);
        if (ig_md.dropping == 1) {
            mark_to_drop(ig_intr_md);
            return;
        }
        ig_ctl_vrf.apply(hdr,ig_md,ig_intr_md);
        ig_ctl_ipv4c.apply(hdr,ig_md,ig_intr_md);
        ig_ctl_ipv6c.apply(hdr,ig_md,ig_intr_md);
        if (ig_md.dropping == 1) {
            mark_to_drop(ig_intr_md);
            return;
        }
```

# hdr.ipv4.total_len mtu
```
    table set_opcodes {
        key = {
            eg_md.switchml_md.first_packet : exact;
            eg_md.switchml_md.last_packet : exact;
        }
        actions = {
            set_first;
            set_middle;
            set_last_immediate;
            set_only_immediate;
        }
        size = 4;
        const entries = {
            ( true, false) :          set_first(); // RDMA_WRITE_FIRST;
            (false, false) :         set_middle(); // RDMA_WRITE_MIDDLE;
            (false,  true) : set_last_immediate(); // RDMA_WRITE_LAST_IMMEDIATE;
            ( true,  true) : set_only_immediate(); // RDMA_WRITE_ONLY_IMMEDIATE;
        }
    }

    apply {
        // Get switch IP and switch MAC
        switch_mac_and_ip.apply();

        // Fill in headers for ROCE packet
        create_roce_packet.apply();

        // Add payload size
        if (eg_md.switchml_md.packet_size == packet_size_t.IBV_MTU_256) {
            hdr.ipv4.total_len = hdr.ipv4.total_len + 256;
            hdr.udp.length = hdr.udp.length + 256;
        }
        else if (eg_md.switchml_md.packet_size == packet_size_t.IBV_MTU_1024) {
            hdr.ipv4.total_len = hdr.ipv4.total_len + 1024;
            hdr.udp.length = hdr.udp.length + 1024;
        }

        // Fill in queue pair number and sequence number
        fill_in_qpn_and_psn.apply();

        // Fill in opcode based on pool index
        set_opcodes.apply();
    }
}
```

# l3_mtu_check 

ipv4_mtu_check   