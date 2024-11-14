


```

    action apply_hash1() {
        hdr.ethernet.dst_addr[31:0] = hash1.get({hdr.ethernet.dst_addr[31:0]});
    }

    action apply_hash2() {
        hdr.ethernet.src_addr[31:0] = hash2.get({hdr.ethernet.src_addr[31:0]});
    }
```