

#   pseudo header
```
control SwitchEgressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in eg_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprsr) {
   	
    Checksum() ip_checksum;
    Checksum() tcp_checksum;    
    
    apply {
    
	   if(eg_md.redo_checksum == 1){	        
              hdr.ipv4.hdr_checksum = ip_checksum.update({
              hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.total_len,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.frag_offset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.src_addr,
              hdr.ipv4.dst_addr
          });

          hdr.tcp.checksum = tcp_checksum.update({
              //==pseudo header
                      hdr.ipv4.src_addr,
                      hdr.ipv4.dst_addr,
                      8w0,
                      hdr.ipv4.protocol,
                      eg_md.tcp_total_len,
              //==actual header
                      hdr.tcp.src_port,
                      hdr.tcp.dst_port,
                      hdr.tcp.seq_no,
                      hdr.tcp.ack_no,
                      hdr.tcp.data_offset,
                      hdr.tcp.res,
    		      hdr.tcp.flag_cwr, 
		      hdr.tcp.flag_ece, 
		      hdr.tcp.flag_urg, 
		      hdr.tcp.flag_ack, 
		      hdr.tcp.flag_psh, 
		      hdr.tcp.flag_rst, 
		      hdr.tcp.flag_syn, 
		      hdr.tcp.flag_fin, 
                      hdr.tcp.window,
                      hdr.tcp.urgent_ptr
                      //hdr.payload
          });

	}//endif redo_checksum 

        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.sip_meta);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
    }
}

```

# update checksum

```
// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------
control SwitchIngressDeparser(packet_out pkt,
                              inout header_t hdr,
                              in metadata_t ig_md,
                              in ingress_intrinsic_metadata_for_deparser_t 
                                ig_intr_dprsr_md
                              ) {

    Checksum() ipv4_checksum;
    Checksum() tcp_checksum;
    Checksum() udp_checksum;

    apply {
        // Updating and checking of the checksum is done in the deparser.
        // Checksumming units are only available in the parser sections of 
        // the program.
        if (ig_md.checksum_upd_ipv4) {
            hdr.ipv4.hdr_checksum = ipv4_checksum.update(
                {hdr.ipv4.version,
                 hdr.ipv4.ihl,
                 hdr.ipv4.diffserv,
                 hdr.ipv4.total_len,
                 hdr.ipv4.identification,
                 hdr.ipv4.flags,
                 hdr.ipv4.frag_offset,
                 hdr.ipv4.ttl,
                 hdr.ipv4.protocol,
                 hdr.ipv4.src_addr,
                 hdr.ipv4.dst_addr});
        }
        if (ig_md.checksum_upd_tcp) {
            hdr.tcp.checksum = tcp_checksum.update({
                hdr.ipv4.src_addr,
                hdr.tcp.src_port,
                ig_md.checksum_tcp_tmp
            });
        }
        if (ig_md.checksum_upd_udp) {
            hdr.udp.checksum = udp_checksum.update(data = {
                hdr.ipv4.src_addr,
                hdr.udp.src_port,
                ig_md.checksum_udp_tmp
            }, zeros_as_ones = true);
            // UDP specific checksum handling
        }

        pkt.emit(hdr);
    }
}

```