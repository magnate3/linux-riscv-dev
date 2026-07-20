/*************************************************************************
************   Adding INT to the Packet  *************
*************************************************************************/

control process_int (
    inout headers hdr,
    inout local_metadata_t metadata) {

    action int_a(){

      hdr.intl4_shim.setValid();                              // insert INT shim header
      hdr.intl4_shim.int_type = 3;                            // int_type: Hop-by-hop type (1) , destination type (2), MX-type (3)
      hdr.intl4_shim.npt = 0;                                 // next protocol type: 0
      hdr.intl4_shim.len = INT_HEADER_LEN_WORD;               // This is 3 from 0xC (INT_TOTAL_HEADER_SIZE >> 2)
      hdr.intl4_shim.udp_ip_dscp = hdr.ipv4.dscp;             // although should be first 6 bits of the second byte
      hdr.intl4_shim.udp_ip = 0;                              // although should be first 6 bits of the second byte

      // insert INT header
      hdr.int_header.setValid();
      hdr.int_header.ver = 2;
      hdr.int_header.d = 0;
      hdr.int_header.class = (bit<4>)metadata.classID;
      hdr.int_header.latency = metadata.latency;
      hdr.int_header.priority = metadata.priority;               // bit 8 is buffer related, rest are reserved



      // add the header len (3 words) to total len
      hdr.ipv4.len = hdr.ipv4.len + INT_TOTAL_HEADER_SIZE;
      hdr.udp.length_ = hdr.udp.length_ + INT_TOTAL_HEADER_SIZE;

      hdr.ipv4.dscp = DSCP_INT;

    }

    table int_t {

    actions = {
        int_a;
        NoAction;
    }
    const default_action = NoAction();
}

apply {
    int_t.apply();

}

}
