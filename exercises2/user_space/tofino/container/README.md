

# bug


## bug1
```
error: invalid SuperCluster was formed: SUPERCLUSTER Uid: 166
    slice lists:
        [ ingress::hdr.ip6.version<4> ^4 ^bit[0..3] deparsed exact_containers [0:3] ]
    rotational clusters:
        [[ingress::hdr.ip6.version<4> ^4 ^bit[0..3] deparsed exact_containers [0:3]]]
because this slice list is not byte-sized: [ ingress::hdr.ip6.version<4> ^4 ^bit[0..3] deparsed exact_containers [0:3] ] has 4 bits.
This is either a compiler internal bug or you can introduce padding fields around them by @padding or @flexible
Number of errors exceeded set maximum of 1

```
将@pa_container_size("ingress", "hdr.ip6.traffic_class", 8)改成   
@pa_container_size("ingress", "hdr.ip6.traffic_class", 16)



## bug2

```
        hdr.ip6.version = 6;
        //hdr.ip6.traffic_class = 64;
        hdr.ip6.flow_label = 0;
        hdr.ip6.payload_len = hdr.ip4.total_len -  IPV4_HDR_SIZE;
        hdr.ip6.next_hdr = hdr.ip4.protocol;
        hdr.ip6.hop_limit = hdr.ip4.ttl + 64;
        hdr.ip6.traffic_class = hdr.ip6.hop_limit +32;
```

```
warning: Parser state min_parse_depth_accept_loop will be unrolled up to 4 times due to @pragma max_loop_depth.
warning: Parser state min_parse_depth_accept_loop will be unrolled up to 4 times due to @pragma max_loop_depth.
error: This program violates action constraints imposed by Tofino.

  The following field slices must be allocated in the same container as they are present within the same byte of header ingress::hdr.ip6:
        ingress::hdr.ip6.traffic_class
        ingress::hdr.ip6.version

  However, the program requires multiple instruction types for the same container in the same action (MyIngress.ipv4_forward):
        The following slice(s) are written using add instruction.
          ingress::hdr.ip6.traffic_class
        The following slice(s) are written using assignment instruction.
          ingress::hdr.ip6.version

Therefore, the program requires an action impossible to synthesize for Tofino ALU. Rewrite action MyIngress.ipv4_forward to use the same instruction for all the above field slices that must be in the same container.

Number of errors exceeded set maximum of 1
```

改成如下还是报同样的错误       
```
        hdr.ip6.version =   6;
        //hdr.ip6.traffic_class = 64;
        hdr.ip6.flow_label = 0;
        hdr.ip6.payload_len = hdr.ip4.total_len -  IPV4_HDR_SIZE;
        hdr.ip6.next_hdr = hdr.ip4.protocol;
        hdr.ip6.hop_limit = hdr.ip4.ttl + 64;
        hdr.ip6.traffic_class = hdr.ip6.hop_limit;
```



改成如下编译通过       
```
        hdr.ip6.version =   6;
        hdr.ip6.traffic_class = 64;
        hdr.ip6.flow_label = 0;
        hdr.ip6.payload_len = hdr.ip4.total_len -  IPV4_HDR_SIZE;
        hdr.ip6.next_hdr = hdr.ip4.protocol;
        hdr.ip6.hop_limit = hdr.ip4.ttl + 64;
```

## 强制转换bug2   

```
        hdr.ip6.version =   6;
        hdr.ip6.traffic_class = 64;
        //hdr.ip6.flow_label = 0;
        hdr.ip6.payload_len = hdr.ip4.total_len -  IPV4_HDR_SIZE;
        hdr.ip6.flow_label = (bit<20>)hdr.ip6.payload_len;
        hdr.ip6.next_hdr = hdr.ip4.protocol;
        hdr.ip6.hop_limit = hdr.ip4.ttl + 64
```
hdr.ip6.flow_label = (bit<20>)hdr.ip6.payload_len;    
```
warning: Parser state min_parse_depth_accept_loop will be unrolled up to 4 times due to @pragma max_loop_depth.
warning: Parser state min_parse_depth_accept_loop will be unrolled up to 4 times due to @pragma max_loop_depth.
error: invalid SuperCluster was formed: SUPERCLUSTER Uid: 141
    slice lists:
        [ ingress::hdr.ip4.dst_addr<32> ^0 ^bit[0..159] [0:31]
          ingress::hdr.ip4.src_addr<32> [0:31]
          ingress::hdr.ip4.hdr_checksum<16> [0:15]
          ingress::hdr.ip4.protocol<8> ^0 ^bit[0..79] [0:7]
          ingress::hdr.ip4.ttl<8> ^0 ^bit[0..71] no_split [0:7]
          ingress::hdr.ip4.frag_offset<13> [0:12]
          ingress::hdr.ip4.flags<3> [0:2]
          ingress::hdr.ip4.identification<16> [0:15]
          ingress::hdr.ip4.total_len<16> ^0 ^bit[0..31] no_split [0:15]
          ingress::hdr.ip4.diffserv<8> [0:7]
          ingress::hdr.ip4.ihl<4> [0:3]
          ingress::hdr.ip4.version<4> [0:3] ]
        [ ingress::hdr.ip6.next_hdr<8> ^0 ^bit[0..55] deparsed exact_containers [0:7] ]
        [ ingress::hdr.ip6.hop_limit<8> ^0 ^bit[0..63] deparsed solitary no_split exact_containers [0:7] ]
        [ ingress::hdr.ip6.payload_len<16> ^0 ^bit[0..47] deparsed solitary no_split exact_containers [0:15] ]
        [ ingress::hdr.ip6.flow_label<20> ^0 ^bit[0..31] deparsed solitary exact_containers [0:15]
          ingress::hdr.ip6.flow_label<20> ^0 ^bit[0..15] deparsed solitary exact_containers [16:19] ]
    rotational clusters:
        [[ingress::hdr.ip4.dst_addr<32> ^0 ^bit[0..159] [0:31]]]
        [[ingress::hdr.ip4.src_addr<32> [0:31]]]
        [[ingress::hdr.ip4.hdr_checksum<16> [0:15]]]
        [[ingress::hdr.ip6.next_hdr<8> ^0 ^bit[0..55] deparsed exact_containers [0:7]], [ingress::hdr.ip4.protocol<8> ^0 ^bit[0..79] [0:7]]]
        [[ingress::hdr.ip6.hop_limit<8> ^0 ^bit[0..63] deparsed solitary no_split exact_containers [0:7], ingress::hdr.ip4.ttl<8> ^0 ^bit[0..71] no_split [0:7]]]
        [[ingress::hdr.ip4.frag_offset<13> [0:12]]]
        [[ingress::hdr.ip4.flags<3> [0:2]]]
        [[ingress::hdr.ip4.identification<16> [0:15]]]
        [[ingress::hdr.ip6.payload_len<16> ^0 ^bit[0..47] deparsed solitary no_split exact_containers [0:15], ingress::hdr.ip4.total_len<16> ^0 ^bit[0..31] no_split [0:15], ingress::hdr.ip6.flow_label<20> ^0 ^bit[0..31] deparsed solitary exact_containers [0:15]]]
        [[ingress::hdr.ip4.diffserv<8> [0:7]]]
        [[ingress::hdr.ip4.ihl<4> [0:3]]]
        [[ingress::hdr.ip4.version<4> [0:3]]]
        [[ingress::hdr.ip6.flow_label<20> ^0 ^bit[0..15] deparsed solitary exact_containers [16:19]]]
because this slice list is not byte-sized: [ ingress::hdr.ip6.flow_label<20> ^0 ^bit[0..31] deparsed solitary exact_containers [0:15]
          ingress::hdr.ip6.flow_label<20> ^0 ^bit[0..15] deparsed solitary exact_containers [16:19] ] has 20 bits.
This is either a compiler internal bug or you can introduce padding fields around them by @padding or @flexible
Number of errors exceeded set maximum of 1
```

# bug3
````
@pa_container_size("ingress", "hdr.ip6.hop_limit", 8)
@pa_container_size("ingress", "hdr.ip4.ttl", 8)
```
放到同样的pa_container_size，编译没有出错

改成  

````
@pa_container_size("ingress", "hdr.ip6.hop_limit", 8)
@pa_container_size("ingress", "hdr.ip4.ttl", 16)
```

```
SUPERCLUSTER Uid: 141
    slice lists:
        [ ingress::hdr.ip4.dst_addr<32> ^0 ^bit[0..159] [0:31]
          ingress::hdr.ip4.src_addr<32> [0:31]
          ingress::hdr.ip4.hdr_checksum<16> [0:15]
          ingress::hdr.ip4.protocol<8> ^0 ^bit[0..79] [0:7]
          ingress::hdr.ip4.ttl<8> ^0 ^bit[0..71] no_split [0:7]
          ingress::hdr.ip4.frag_offset<13> [0:12]
          ingress::hdr.ip4.flags<3> [0:2]
          ingress::hdr.ip4.identification<16> [0:15]
          ingress::hdr.ip4.total_len<16> ^0 ^bit[0..31] no_split [0:15]
          ingress::hdr.ip4.diffserv<8> [0:7]
          ingress::hdr.ip4.ihl<4> [0:3]
          ingress::hdr.ip4.version<4> [0:3] ]
        [ ingress::hdr.ip6.next_hdr<8> ^0 ^bit[0..55] deparsed exact_containers [0:7] ]
        [ ingress::hdr.ip6.hop_limit<8> ^0 ^bit[0..63] deparsed solitary no_split exact_containers [0:7] ]
        [ ingress::hdr.ip6.payload_len<16> ^0 ^bit[0..47] deparsed solitary no_split exact_containers [0:15] ]
    rotational clusters:
        [[ingress::hdr.ip4.dst_addr<32> ^0 ^bit[0..159] [0:31]]]
        [[ingress::hdr.ip4.src_addr<32> [0:31]]]
        [[ingress::hdr.ip4.hdr_checksum<16> [0:15]]]
        [[ingress::hdr.ip6.next_hdr<8> ^0 ^bit[0..55] deparsed exact_containers [0:7]], [ingress::hdr.ip4.protocol<8> ^0 ^bit[0..79] [0:7]]]
        [[ingress::hdr.ip6.hop_limit<8> ^0 ^bit[0..63] deparsed solitary no_split exact_containers [0:7], ingress::hdr.ip4.ttl<8> ^0 ^bit[0..71] no_split [0:7]]]
        [[ingress::hdr.ip4.frag_offset<13> [0:12]]]
        [[ingress::hdr.ip4.flags<3> [0:2]]]
        [[ingress::hdr.ip4.identification<16> [0:15]]]
        [[ingress::hdr.ip6.payload_len<16> ^0 ^bit[0..47] deparsed solitary no_split exact_containers [0:15], ingress::hdr.ip4.total_len<16> ^0 ^bit[0..31] no_split [0:15]]]
        [[ingress::hdr.ip4.diffserv<8> [0:7]]]
        [[ingress::hdr.ip4.ihl<4> [0:3]]]
        [[ingress::hdr.ip4.version<4> [0:3]]]

Number of errors exceeded set maximum of 1
```


## bug4

```
        hdr.ip6.hop_limit = hdr.ip4.ttl + 64;
        hdr.ip6.hop_limit = hdr.ip6.hop_limit *2;
        hdr.ip6.hop_limit = hdr.ip6.hop_limit *2 -64;
        hdr.ip6.hop_limit = hdr.ip6.hop_limit *2 +64;
```

```
[--Werror=unsupported] error: shl: action spanning multiple stages. Operations on operand 2 ($tmp1[0..7]) in action ipv4_forward require multiple stages for a single action. We currently support only single stage actions. Please consider rewriting the action to be a single stage action.
        hdr.ip6.hop_limit = hdr.ip6.hop_limit *2;
                            ^^^^^^^^^^^^^^^^^^^^
/sde/bf-sde-9.7.1/p4studio/build-test/container/p4_16-tna-skel.p4(174)
    action ipv4_forward(bit<48> dstAddr,bit<9> port){
           ^^^^^^^^^^^^
Number of errors exceeded set maximum of 1
```



改成如下编译通过   
```
        hdr.ip6.version =   6;
        hdr.ip6.traffic_class = 64;
        hdr.ip6.flow_label = 0;
        hdr.ip6.payload_len = hdr.ip4.total_len -  IPV4_HDR_SIZE;
        hdr.ip6.next_hdr = hdr.ip4.protocol;
        hdr.ip6.hop_limit = hdr.ip4.ttl + 64;
        hdr.ip6.hop_limit = hdr.ip4.ttl *2;
```