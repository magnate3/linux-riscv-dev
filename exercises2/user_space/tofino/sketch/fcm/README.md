
流量大小估计是每流测量中的一项关键任务。 在实际场景中，在数百万个流中，人们通常对给定流的一小部分特别感兴趣。 这些优先级高的流被称为重要流。 在这张海报中，我们提出了 Cuckoo sketch，它为重要流提供了更高的准确性。 Cuckoo sketch的关键思想是将重要流与不重要流分开，并将它们存储在不同的结构中：重要流存储在 Cuckoo 哈希表中，其中存储流的确切大小，而不重要流存储在count-min sketch。 Cuckoo哈希的重新分配机制是通过驱逐不太重要的流来在Cuckoo哈希表中为重要的流腾出一些空间，这会导致吞吐量的轻微损失。 实验结果表明，Cuckoo sketch 对重要流提供了更好的精度，将平均相对误差降低了 69%，而对不重要流的精度损失几乎可以忽略不计。     

# range match


[线段树 Segment Tree 实战](https://halfrost.com/segment_tree/)   

# mirror   
```
control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    Mirror() mirror;
    Digest <hh_digest_t>() hh_digest;

    apply {
        if(ig_dprsr_md.mirror_type == 1) {
            // session 1, where it points to the recirculation port
            mirror.emit(10w1);
        }

        if(ig_dprsr_md.digest_type == HH_DIGEST) {
            hh_digest.pack({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.ipv4.protocol, meta.src_port, meta.dst_port});
        }

        pkt.emit(hdr);
    }
}
```

# fcm  

[fcm1](P4_SketchLib/p4_16/API/API_O6_flowkey_fcm.p4)   
[fcm_p416](https://github.com/fcm-project/fcm_p4/blob/338dffbaa551d21a456367d5b5007e62eb439986/fcm_p416/fcm.p4)      

```
 ./run_bfshell.sh  -b  config-test/simple_fcm.py 
```


```
./run_bfshell.sh  -b  config-test/digest_fcm.py 
Using SDE /sde/bf-sde-9.7.1
Using SDE_INSTALL /sde/bf-sde-9.7.1/install
Connecting to localhost port 7777 to check status on these devices: [0]
Waiting for device 0 to be ready
/sde/bf-sde-9.7.1/install/bin/bfshell config-test/digest_fcm.py
bfrt_python config-test/digest_fcm.py
exit

        ********************************************
        *      WARNING: Authorised Access Only     *
        ********************************************
    

bfshell> bfrt_python config-test/digest_fcm.py
cwd : /sde/bf-sde-9.7.1

Devices found :  [0]
We've found 1 p4 programs for device 0:
fcm
Creating tree for dev 0 and program fcm

/sde/bf-sde-9.7.1/install/lib/python3.8/site-packages/IPython/core/history.py:226: UserWarning: IPython History requires SQLite, your history will not be saved
  warn("IPython History requires SQLite, your history will not be saved")
Python 3.8.10 (default, Jul 12 2024, 16:34:23) 
Type 'copyright', 'credits' or 'license' for more information
IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
Deregistering old callback function (if any)
Registering callback...
Bound callback to digest
---------------------------------
Received message from data plane!
{'src_addr': 168431495, 'dst_addr': 168431494, 'protocol': 6, 'src_port': 9999, 'dst_port': 59470}
{'src_addr': 168431494, 'dst_addr': 168431495, 'protocol': 6, 'src_port': 59470, 'dst_port': 9999}
Deregistering old callback function (if any)
bfshell> exit
```

## add range match  


```
bfrt.fcm.pipe.SwitchIngress.fcmsketch.tb_fcm_cardinality>  help(add_with_fcm_action_set_cardinality)
Help on method add_with_fcm_action_set_cardinality in module bfrtcli:

add_with_fcm_action_set_cardinality(num_occupied_reg_start=None, num_occupied_reg_end=None, MATCH_PRIORITY=None, card_match=None, pipe=None, gress_dir=None, prsr_id=None) method of bfrtcli.BFLeaf instance
    Add entry to tb_fcm_cardinality table with action: SwitchIngress.fcmsketch.fcm_action_set_cardinality
    
    Parameters:
    ['num_occupied_reg_start', 'num_occupied_reg_end'] type=RANGE      size=19 default=0
    MATCH_PRIORITY                 type=EXACT      size=32 default=0
    card_match                     type=BYTE_STREAM size=32 default=0

```

```

bfrt.fcm.pipe.SwitchIngress.fcmsketch.tb_fcm_cardinality> add_with_fcm_action_set_cardinality('0x1ff','0xfff')

bfrt.fcm.pipe.SwitchIngress.fcmsketch.tb_fcm_cardinality>
```