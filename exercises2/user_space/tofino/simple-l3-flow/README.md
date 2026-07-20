

# 前缀匹配


```
bfrt.simple_l3.pipe> Ingress.ipv4_lpm
-------------------> Ingress.ipv4_lpm()
BF Runtime CLI Object for pipe.Ingress.ipv4_lpm table

Key fields:
    hdr.ipv4.dst_addr              type=LPM        size=32


Actions, Data fields:
    Ingress.drop
        0 data fields:
    Ingress.send
        1 data fields:
            port                           type=BYTE_STREAM size=9


Available Commands:
add_from_json
add_with_drop
add_with_send
clear
delete
dump
entry_with_drop
entry_with_send
get
get_default
get_handle
get_key
info
mod_with_drop
mod_with_send
reset_default
set_default_with_drop
set_default_with_send
symmetric_mode_get
symmetric_mode_set

bfrt.simple_l3.pipe.Ingress.ipv4_lpm> from ipaddress import ip_address

bfrt.simple_l3.pipe.Ingress.ipv4_lpm> dump
------------------------------------> dump()
----- ipv4_lpm Dump Start -----
Default Entry:
Entry data (action : Ingress.send):
    port                           : 0x40

Table pipe.Ingress.ipv4_lpm has no entries.
----- ipv4_lpm Dump End -----


bfrt.simple_l3.pipe.Ingress.ipv4_lpm> add_with_send(ip_address('192.168.1.0'), 24, 1)

bfrt.simple_l3.pipe.Ingress.ipv4_lpm> add_with_send(ip_address('192.168.2.0'), 24, 1)

bfrt.simple_l3.pipe.Ingress.ipv4_lpm> dump
------------------------------------> dump()
----- ipv4_lpm Dump Start -----
Default Entry:
Entry data (action : Ingress.send):
    port                           : 0x40

Entry 0:
Entry key:
    hdr.ipv4.dst_addr              : ('0xC0A80100', '0x00000018')
Entry data (action : Ingress.send):
    port                           : 0x01

Entry 1:
Entry key:
    hdr.ipv4.dst_addr              : ('0xC0A80200', '0x00000018')
Entry data (action : Ingress.send):
    port                           : 0x01

----- ipv4_lpm Dump End -----


bfrt.simple_l3.pipe.Ingress.ipv4_lpm> 
```

# 精确匹配


```
bfrt.simple_l3.pipe.Ingress.ipv4_lpm> bfrt.simple_l3.pipe.Ingress.ipv4_host
------------------------------------> bfrt.simple_l3.pipe.Ingress.ipv4_host()
BF Runtime CLI Object for pipe.Ingress.ipv4_host table

Key fields:
    hdr.ipv4.dst_addr              type=EXACT      size=32


Actions, Data fields:
    NoAction (DefaultOnly)
        2 data fields:
            $COUNTER_SPEC_BYTES            type=UINT64     size=64
            $COUNTER_SPEC_PKTS             type=UINT64     size=64
    Ingress.ipv4_host_send
        3 data fields:
            port                           type=BYTE_STREAM size=9
            $COUNTER_SPEC_BYTES            type=UINT64     size=64
            $COUNTER_SPEC_PKTS             type=UINT64     size=64
    Ingress.ipv4_host_drop
        2 data fields:
            $COUNTER_SPEC_BYTES            type=UINT64     size=64
            $COUNTER_SPEC_PKTS             type=UINT64     size=64


Available Commands:
add_from_json
add_with_ipv4_host_drop
add_with_ipv4_host_send
clear
delete
dump
entry_with_ipv4_host_drop
entry_with_ipv4_host_send
get
get_default
get_handle
get_key
info
mod_with_ipv4_host_drop
mod_with_ipv4_host_send
operation_counter_sync
symmetric_mode_get
symmetric_mode_set

bfrt.simple_l3.pipe.Ingress.ipv4_host> add_with_ipv4_host_send(ip_address('192.168.1.2'), 2)

bfrt.simple_l3.pipe.Ingress.ipv4_host> dump
-------------------------------------> dump()
----- ipv4_host Dump Start -----
Default Entry:
Entry data (action : NoAction):
    $COUNTER_SPEC_BYTES            : 0
    $COUNTER_SPEC_PKTS             : 0

Entry 0:
Entry key:
    hdr.ipv4.dst_addr              : 0xC0A80102
Entry data (action : Ingress.ipv4_host_send):
    port                           : 0x02
    $COUNTER_SPEC_BYTES            : 0
    $COUNTER_SPEC_PKTS             : 0

----- ipv4_host Dump End -----


bfrt.simple_l3.pipe.Ingress.ipv4_host> 
```