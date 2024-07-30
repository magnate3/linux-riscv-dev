

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
```
@pa_container_size("ingress", "hdr.ip6.hop_limit", 8)
@pa_container_size("ingress", "hdr.ip4.ttl", 8)
```
放到同样的pa_container_size，编译没有出错

改成  

```
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


# bug4
斐波那契数列    
```
        ig_md.operand1 = hdr.ipv4.total_len*2;
        ig_md.operand2 = ig_md.operand1 *2 +64;
```


```
[--Werror=unsupported] error: add: action spanning multiple stages. Operations on operand 2 ($tmp2[0..15]) in action register_action require multiple stages for a single action. We currently support only single stage actions. Please consider rewriting the action to be a single stage action.
        ig_md.operand2 = ig_md.operand1 *2 +64;
                         ^^^^^^^^^^^^^^^^^^^^^
/sde/bf-sde-9.7.1/p4studio/build-test/nat64/tofino-nat64.p4(458)
    action register_action(bit<32> idx) {
           ^^^^^^^^^^^^^^^
Number of errors exceeded set maximum of 1

```
不能做多个逻辑远算，改成   

```
ig_md.operand1  +64;
```
仍然有bug    
```
        ig_md.operand1 = hdr.ipv4.total_len*2;
        ig_md.operand2 = ig_md.operand1  +64;
        ig_md.operand3 = ig_md.operand2 * ig_md.operand1;
        ig_md.operand4 = ig_md.operand2 * ig_md.operand3;
        ig_md.operand5 = ig_md.operand4 * ig_md.operand3;
        ig_md.operand6 = ig_md.operand4 * ig_md.operand5;
        ig_md.operand7 = ig_md.operand5 * ig_md.operand6;
        ig_md.operand8 = ig_md.operand7 * ig_md.operand6;
```
从求ig_md.operand3开始都有bug        

```
error: : source of modify_field invalid
        ig_md.operand3 = ig_md.operand2 * ig_md.operand1;
                       ^
Number of errors exceeded set maximum of 1

```

## bug5


不能做多个逻辑远算   
```
[--Werror=unsupported] error: add: action spanning multiple stages. Operations on operand 2 ($tmp2[0..15]) in action fib_action2 require multiple stages for a single action. We currently support only single stage actions. Please consider rewriting the action to be a single stage action.
        ig_md.operand2 = hdr.ipv4.total_len*2 +4;
                         ^^^^^^^^^^^^^^^^^^^^^^^
/sde/bf-sde-9.7.1/p4studio/build-test/nat64/tofino-nat64.p4(477)
    action fib_action2() {
           ^^^^^^^^^^^
```
 
 改成
 
 ```
   action fib_action2() {
        ig_md.operand2 = hdr.ipv4.total_len +4;
    }
 ```

## bug6

```
struct metadata_t {
    bit<1>        first_frag;
    bool        checksum_err_ipv4;
    bit<16>  l4_payload_checksum;
    bool     nat64;
    bit<16>  operand1;
    bit<16>  operand2;
    bit<16>  operand3;
    bit<16>  operand4;
    bit<16>  operand5;
    bit<16>  operand6;
    bit<16>  operand7;
    bit<16>  operand8;
}
```
操作数不能是同一个struct的成员

```
 error: : source of modify_field invalid
        ig_md.operand3 = ig_md.operand2 * ig_md.operand1;
                       ^
```


```
error: : source of modify_field invalid
        ig_md.operand4 = ig_md.operand2 * ig_md.operand3;
```

改成如下：

```
struct metadata_t {
    bit<16>  operand1;
    bit<16>  operand2;
    bit<16>  operand3;
    bit<16>  operand4;
    bit<16>  operand5;
    bit<16>  operand6;
    bit<16>  operand7;
    bit<16>  operand8;
}
```

```
    action fib_action1() {
        ig_md.operand1 = hdr.ipv4.total_len*2;
    }
    action fib_action2() {
        ig_md.operand2 = hdr.ipv4.total_len +4;
    }
    action fib_action3() {
        ig_md.operand3 = ig_md.operand2 * 2;
        //ig_md.operand3 = ig_md.operand2 * ig_md.operand1;
    }
    action fib_action4() {
        ig_md.operand4 = ig_md.operand3 * 2;
        //ig_md.operand4 = ig_md.operand2 * ig_md.operand3;
    }
    action fib_action5() {
        ig_md.operand5 = ig_md.operand4 * 2;
        //ig_md.operand5 = ig_md.operand3 * ig_md.operand4;
    }
    action fib_action6() {
        ig_md.operand6 = ig_md.operand5 * 2;
        //ig_md.operand6 = ig_md.operand4 * ig_md.operand5;
    }
    action fib_action7() {
        ig_md.operand7 = ig_md.operand6 * 2;
        //ig_md.operand7 = ig_md.operand5 * ig_md.operand6;
    }
    action fib_action8() {
        ig_md.operand8 = ig_md.operand7 * 2;
        //ig_md.operand8 = ig_md.operand6 * ig_md.operand7;
    }
        table ipv4_lpm {
        key = {
            hdr.ipv4.dst_addr: exact;
        }
        actions = {
            ipv4_translate;
            register_action;
            fib_action1;
            fib_action2;
            fib_action3;
            fib_action4;
            fib_action5;
            fib_action6;
            fib_action7;
            fib_action8;
             @defaultonly NoAction;
        }
        size = 1024;
        const default_action = NoAction();
    }
```



