

# tcam ternary match



```
state outer_udp {
        pkt.extract(hdr.outer_udp);

        outer_udp_chksum.subtract(hdr.outer_udp.src);
        outer_udp_chksum.subtract(hdr.outer_udp.dst);
        outer_udp_chksum.subtract(hdr.outer_udp.chksum);

        // Default port range for SCION BRs is [30042, 30052).
        // Port 50000 and up is used by the scion topology generator.
        transition select (hdr.outer_udp.dst) {
            30042 &&& 0xfffe   : scion; // [30042, 30043]
            30044 &&& 0xfffc   : scion; // [30044, 30047]
            30048 &&& 0xfffc   : scion; // [30048, 30051]
            50000 &&& 0xfff0   : scion; // [50000, 50015]
            default            : accept;
        }
    }
```

# table ternary
```
# Choose any one
// #define MATCH_TYPE exact
// #define MATCH_TYPE lpm
#define MATCH_TYPE ternary
```

```
   // This is currently dealing both L3 and L2
    table tbl_l3_routes {
        key = {
            hdr.ipv4.dstAddr : MATCH_TYPE;
        }

        actions = {
            l3_routes_act;
            nop;
        }
        
        const default_action = nop();
        size = 32;
    }
```



# Common rules
```
bfrt.simple_router_l3.pipe.RouterIngress.arp_tbl.add_with_arp_act('192.168.0.1','00:a0:c9:00:00:00')
bfrt.simple_router_l3.pipe.RouterIngress.arp_tbl.add_with_arp_act('192.168.0.2','34:12:78:56:01:00')
```
# Rules for exact matching
```
bfrt.simple_router_l3.pipe.RouterIngress.tbl_l3_routes.add_with_l3_routes_act('192.168.0.1', 64)
bfrt.simple_router_l3.pipe.RouterIngress.tbl_l3_routes.add_with_l3_routes_act("192.168.0.2", 65)
```
# Rules for lpm matching
 Match, prefix, argument)   
```
bfrt.simple_router_l3.pipe.RouterIngress.tbl_l3_routes.add_with_l3_routes_act('192.168.0.1', 32, 64)
bfrt.simple_router_l3.pipe.RouterIngress.tbl_l3_routes.add_with_l3_routes_act("192.168.0.2", 32, 65)
```
# Rules for ternary matching
 (Match, Mask, Priority, argument)    
```
bfrt.simple_router_l3.pipe.RouterIngress.tbl_l3_routes.add_with_l3_routes_act('10.10.0.1', 0x0000FFFF, 0, 64)   
bfrt.simple_router_l3.pipe.RouterIngress.tbl_l3_routes.add_with_l3_routes_act("10.10.0.2", 0x0000FFFF, 0, 65)   
```


```
ipv4_table2.get(0x0A0A0F86,0xFFFFFFFF,65, from_hw=True)
bfrt.tofino_nat64.pipe> SwitchIngress.ipv4_lpm2.get(0x0A0A0F86,0xFFFFFFFF,65, from_hw=True)
Entry 0:
Entry key:
    hdr.ipv4.dst_addr              : ('0x0A0A0F86', '0xFFFFFFFF')
    $MATCH_PRIORITY                : 65
Entry data (action : SwitchIngress.ipv4_translate):
    ipv6dstAddr                    : 0x20080000000000000000000000000004
    port                           : 0xA8

Out[5]: Entry for pipe.SwitchIngress.ipv4_lpm2 table.

bfrt.tofino_nat64.pipe> 
```