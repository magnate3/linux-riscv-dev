
# hdr.udp.checksum
`hdr.inner_ipv4,hdr.gtpu`   
```
 control SwitchComputeChecksum(
            inout Header hdr,
            inout UserMetadata user_md) {
    apply {
        update_checksum(hdr.ipv4.isValid() && hdr.ipv4.ihl == 5,
            { hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.diffserv,
                hdr.ipv4.totalLen,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.fragOffset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.srcAddr,
                hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum, HashAlgorithm.csum16);
          update_checksum_with_payload(hdr.udp.isValid(), { hdr.ipv4.srcAddr, hdr.ipv4.dstAddr,
                        8w0, hdr.ipv4.protocol, hdr.udp.length, hdr.udp.srcPort, hdr.udp.dstPort, hdr.udp.length,hdr.inner_ipv4,hdr.gtpu}, hdr.udp.checksum, HashAlgorithm.csum16);
    }
}
```


```
root@ubuntux86:# ip netns exec host1 ping 192.168.0.2
PING 192.168.0.2 (192.168.0.2) 56(84) bytes of data.
64 bytes from 192.168.0.2: icmp_seq=1 ttl=64 time=4.49 ms
64 bytes from 192.168.0.2: icmp_seq=2 ttl=64 time=5.36 ms
64 bytes from 192.168.0.2: icmp_seq=3 ttl=64 time=6.20 ms
64 bytes from 192.168.0.2: icmp_seq=4 ttl=64 time=5.28 ms
64 bytes from 192.168.0.2: icmp_seq=5 ttl=64 time=5.66 ms
^C
--- 192.168.0.2 ping statistics ---
5 packets transmitted, 5 received, 0% packet loss, time 4004ms
rtt min/avg/max/mdev = 4.487/5.395/6.195/0.556 ms
```