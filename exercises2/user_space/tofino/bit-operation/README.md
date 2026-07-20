
```

    action apply_hash1() {
        hdr.ethernet.dst_addr[31:0] = hash1.get({hdr.ethernet.dst_addr[31:0]});
    }

    action apply_hash2() {
        hdr.ethernet.src_addr[31:0] = hash2.get({hdr.ethernet.src_addr[31:0]});
    }
```
# 大小端  

const bit<64> ipv6_prefix = 0x2008000000000000;    
```
hdr.ipv6.src_addr[127:64] = (bit<64>)hdr.ipv4.src_addr;
hdr.ipv6.src_addr[63:0] = ipv6_prefix;
```
tcpdump显示

```
0:0:a0a:f87:2008::.47512 > 2008::4.9999
```


改成

```
hdr.ipv6.src_addr[63:0] = (bit<64>)hdr.ipv4.src_addr;
hdr.ipv6.src_addr[127:64] = ipv6_prefix;
```

```
2008::a0a:f87.34072 > 2008::4.9999
```
