


#  digest    
```
const bit<3> HH_DIGEST = 0x03;
struct hh_digest_t {
    ipv4_addr_t src_addr;
    ipv4_addr_t dst_addr;
    bit<8>  protocol;
    l4_port_t src_port;
    l4_port_t dst_port;
}
control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    Digest <hh_digest_t>() hh_digest;

    apply {
        if(ig_dprsr_md.digest_type == HH_DIGEST) {
            hh_digest.pack({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.ipv4.protocol, meta.src_port, meta.dst_port});
        }

        pkt.emit(hdr);
    }
}
```