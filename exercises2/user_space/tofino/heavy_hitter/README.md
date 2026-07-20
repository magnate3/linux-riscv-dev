
```
    apply {
        if (hdr.ipv4.isValid()){
            if (hdr.tcp.isValid()){
                //Get register position
                output_hash_1 = hash16.get({hdr.ipv4.srcAddr,
                                            hdr.ipv4.dstAddr,
                                            hdr.tcp.srcPort,
                                            hdr.tcp.dstPort,
                                            hdr.ipv4.protocol});

                output_hash_2 = hash32.get({hdr.ipv4.srcAddr,
                                            hdr.ipv4.dstAddr,
                                            hdr.tcp.srcPort,
                                            hdr.tcp.dstPort,
                                            hdr.ipv4.protocol});
                // Update counters
                counter_1 = (count_t)update_bloom_1.execute(output_hash_1);
                counter_1 = (count_t)update_bloom_1.execute(output_hash_2);
                counter_2 = (count_t)update_bloom_2.execute(output_hash_2);
                // Only if IPv4 the rule is applied. Therefore other packets will not be forwarded.
                if (counter_1 > PACKET_THRESHOLD) {
                    if(counter_2 > PACKET_THRESHOLD) {
                        drop();
                    }
                }
            }
            ipv4_lpm.apply();
        }
        else {
            drop();
        }
    }
```

```
   counter_1 = (count_t)update_bloom_1.execute(output_hash_1);
   counter_1 = (count_t)update_bloom_1.execute(output_hash_2);
```
对于update_bloom_1进行两次访问    

```
parser IngressParser(packet_in pkt,
       ^^^^^^^^^^^^^
warning: Parser state min_parse_depth_accept_loop will be unrolled up to 4 times due to @pragma max_loop_depth.
warning: Parser state min_parse_depth_accept_loop will be unrolled up to 4 times due to @pragma max_loop_depth.
error: Table placement cannot make any more progress.  Though some tables have not yet been placed, dependency analysis has found that no more tables are placeable.  This may be due to shared attachments on partly placed tables; may be able to avoid the problem with @stage on those tables

Number of errors exceeded set maximum of 1

CMakeFiles/heavy_hitter-tofino.dir/build.make:60: recipe for target 'heavy_hitter/tofino/bf-rt.json' failed
```