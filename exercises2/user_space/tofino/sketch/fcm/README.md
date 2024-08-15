
流量大小估计是每流测量中的一项关键任务。 在实际场景中，在数百万个流中，人们通常对给定流的一小部分特别感兴趣。 这些优先级高的流被称为重要流。 在这张海报中，我们提出了 Cuckoo sketch，它为重要流提供了更高的准确性。 Cuckoo sketch的关键思想是将重要流与不重要流分开，并将它们存储在不同的结构中：重要流存储在 Cuckoo 哈希表中，其中存储流的确切大小，而不重要流存储在count-min sketch。 Cuckoo哈希的重新分配机制是通过驱逐不太重要的流来在Cuckoo哈希表中为重要的流腾出一些空间，这会导致吞吐量的轻微损失。 实验结果表明，Cuckoo sketch 对重要流提供了更好的精度，将平均相对误差降低了 69%，而对不重要流的精度损失几乎可以忽略不计。     

# mirror   
```
control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    Mirror() mirror;
    Digest <hh_digest_t>() hh_digest;

    apply {
        if(ig_dprsr_md.mirror_type == 1) {
            // session 1, where it points to the recirculation port
            mirror.emit(10w1);
        }

        if(ig_dprsr_md.digest_type == HH_DIGEST) {
            hh_digest.pack({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.ipv4.protocol, meta.src_port, meta.dst_port});
        }

        pkt.emit(hdr);
    }
}
```