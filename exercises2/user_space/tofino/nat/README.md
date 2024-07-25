
# egress port  
ingress_intrinsic_metadata_t ig_intr_md
```
    action nat(ipv4_addr_t srcAddr, ipv4_addr_t dstAddr, PortId_t dst_port) {
        ig_tm_md.ucast_egress_port = dst_port;
        hdr.ipv4.dst_addr = dstAddr;
        hdr.ipv4.src_addr = srcAddr;
        ig_dprsr_md.drop_ctl = 0;
    }
```


# checksum

```
header tcp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<32>  seq_no;
    bit<32>  ack_no;
    bit<4>   data_offset;
    bit<4>   res;
    bit<8>   flags;
    bit<16>  window;
}

header tcp_checksum_h {
    bit<16>  checksum;
    bit<16>  urgent_ptr;
}

header udp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<16>  len;
}

header udp_checksum_h {
    bit<16>  checksum;
}
```
header分为两部分， checksum 和其他部分分开   

```
header tcp_checksum_h {
    bit<16>  checksum;
    bit<16>  urgent_ptr;
}
    tcp_checksum_h     tcp_ipv4_checksum;
    tcp_checksum_h     tcp_ipv6_checksum;
```

#  Conditions in an action must be simple comparisons of an action data parameter


```
    apply {
        //only if IPV4 the rule is applied. Therefore other packets will not be forwarded.
        if (hdr.ipv4.isValid()){
            ipv4_lpm.apply();
        }   
        //else if (hdr.ipv6.isValid()) {
        //    ipv6_lpm.apply();
        //} 
    }  
```

```
[--Werror=target-error] error: 
Conditions in an action must be simple comparisons of an action data parameter
Try moving the test out of the action and into a control apply block, or making it part of the table key
Number of errors exceeded set maximum of 1

```


# header


```
 p40f_result_hdr_t               p40f_result_hdr
```


```
    action set_result(bit<16> result, bit<1> is_generic_fuzzy) {
        // set result
        hdr.p40f_result_hdr.p0f_result.result = result;
        hdr.p40f_result_hdr.p0f_result.is_generic_fuzzy = is_generic_fuzzy;

        hdr.p40f_result_hdr.p0f_result.result = result;
    }
```



```
   // initaillize
        if (hdr.ipv4.isValid() && hdr.tcp.isValid()) { 
 
            hdr.p40f_result_hdr.setValid();
```

```
    action set_result(bit<16> result, bit<1> is_generic_fuzzy) {
        // set result
        hdr.p40f_result_hdr.p0f_result.result = result;
        hdr.p40f_result_hdr.p0f_result.is_generic_fuzzy = is_generic_fuzzy;

        hdr.p40f_result_hdr.p0f_result.result = result;
    }
```