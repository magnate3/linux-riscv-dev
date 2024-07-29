
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

# time

```
root@localhost:/sde/bf-sde-8.9.1/p4studio/build-test/nat64# echo "Take up `cat build/tofino-nat64/tofino/pipe/tofino-nat64.bfa | grep -c -E "stage.+ingress"` ingress stages"
Take up 5 ingress stages
root@localhost:/sde/bf-sde-8.9.1/p4studio/build-test/nat64# echo "Take up `cat build/tofino-nat64/tofino/pipe/tofino-nat64.bfa |   grep -c -E "stage.+egress"` egress stages"
Take up 0 egress stages
```

#  instruction


```
/sde/bf-sde-8.9.1/p4studio/build-test/nat64# echo "`cat build/tofino-nat64/tofino/pipe/tofino-nat64.bfa |   grep  -E "instruction.+action"`"
    instruction: ipv4_lpm_0(action, $DEFAULT)
    instruction: ipv6_lpm_0(action, $DEFAULT)
    instruction: tbl_tofinonat64l456$tind(action, $DEFAULT)
    instruction: tbl_tofinonat64l448$tind(action, $DEFAULT)
    instruction: ingress_reset_invalidated_checksum_fields_1$tind(action, $DEFAULT)
    instruction: ingress_reset_invalidated_checksum_fields_0$tind(action, $DEFAULT)
    instruction: ingresshdr.ipv4.hdr_checksum_encode_update_condition_3(action, $DEFAULT)
    instruction: ingresshdr.tcp_ipv4_checksum.checksum_encode_update_condition_4(action, $DEFAULT)
    instruction: ingresshdr.tcp_ipv6_checksum.checksum_encode_update_condition_5(action, $DEFAULT)
```
```
apply {
          if(ig_md.nat64 && hdr.tcp_ipv6_checksum.isValid())
            {
        	hdr.tcp_ipv4_checksum.urgent_ptr = hdr.tcp_ipv6_checksum.urgent_ptr;
 	        hdr.tcp_ipv6_checksum.setInvalid();
            }   
}
```

```
      - set hdr.tcp_ipv4_checksum.urgent_ptr, hdr.tcp_ipv6_checksum.urgent_ptr
      - set hdr.tcp_ipv6_checksum.$valid, 0
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

# bug


# bug1

```
error: This program violates action constraints imposed by Tofino.

  The following field slices must be allocated in the same container as they are present within the same byte of header ingress::hdr.ipv6:
        ingress::hdr.ipv6.traffic_class
        ingress::hdr.ipv6.version

  However, the program requires multiple instruction types for the same container in the same action (SwitchIngress.ipv4_translate):
        The following slice(s) are written using add instruction.
          ingress::hdr.ipv6.traffic_class
        The following slice(s) are written using assignment instruction.
          ingress::hdr.ipv6.version

Therefore, the program requires an action impossible to synthesize for Tofino ALU. Rewrite action SwitchIngress.ipv4_translate to use the same instruction for all the above field slices that must be in the same container.

Number of errors exceeded set maximum of 1
```

```
hdr.ipv6.version = zzz;
hdr.ipv6.traffic_class = xxx;
……
hdr.ipv6.hop_limit = yyyy;
hdr.ipv6.traffic_class = hdr.ipv6.hop_limit +64;
```

