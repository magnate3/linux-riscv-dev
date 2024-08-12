

#  @stage  


+  test1   
```
@stage(5)
    table reg_match {
        key = {
            hdr.ethernet.dst_addr : exact;
        }
        actions = {
            register_action;
        }
        size = 1024;
    }
 
    action register_action_dir() {
        //test_reg_dir_action.execute();
        test_reg_action.execute(64);
    }

    @stage(4)
    table reg_match_dir {
        key = {
            hdr.ethernet.src_addr : exact;
        }
        actions = {
            register_action_dir;
        }
        size = 1024;
        registers = test_reg_dir;
    }
```

```
Table allocation done 4 time(s), state = FINAL_PLACEMENT
Number of stages in table allocation: 6
  Number of stages for ingress table allocation: 6
  Number of stages for egress table allocation: 0
Critical path length through the table dependency graph: 1
Number of tables allocated: 4
+-------+-----------------------------+
|Stage  |Table Name                   |
+-------+-----------------------------+
|0      |tbl_tna_register156          |
|4      |SwitchIngress.reg_match_dir  |
|5      |SwitchIngress.reg_match_dir  |
|5      |SwitchIngress.reg_match      |
+-------+-----------------------------+
```


+ test2(没共用acton)

更改register_action_dir    
```
    action register_action_dir() {
        test_reg_dir_action.execute();
        //test_reg_action.execute(64);
    }
```

```
Table allocation done 3 time(s), state = REDO_PHV1
Number of stages in table allocation: 6
  Number of stages for ingress table allocation: 6
  Number of stages for egress table allocation: 0
Critical path length through the table dependency graph: 1
Number of tables allocated: 3
+-------+-----------------------------+
|Stage  |Table Name                   |
+-------+-----------------------------+
|0      |tbl_tna_register156          |
|4      |SwitchIngress.reg_match_dir  |
|5      |SwitchIngress.reg_match      |
+-------+-----------------------------+

```

+  test3(使用同一个action)

```
    //@stage(5)
    table reg_match {
        key = {
            hdr.ethernet.dst_addr : exact;
        }
        actions = {
            register_action;
        }
        size = 1024;
    }

    DirectRegister<pair>() test_reg_dir;
    // A simple dual-width 32-bit register action that will increment the two
    // 32-bit sections independently and return the value of one half before the
    // modification.
    DirectRegisterAction<pair, bit<32>>(test_reg_dir) test_reg_dir_action = {
        void apply(inout pair value, out bit<32> read_value){
            read_value = value.second;
            value.first = value.first + 1;
            value.second = value.second + 100;
        }
    };

    action register_action_dir() {
        test_reg_dir_action.execute();
        //test_reg_action.execute(64);
    }

    //@stage(4)
    table reg_match_dir {
        key = {
            hdr.ethernet.src_addr : exact;
        }
        actions = {
            register_action_dir;
        }
        size = 1024;
        registers = test_reg_dir;
    }
```

没有使用@stage,全部在stage0    

```
Table allocation done 1 time(s), state = INITIAL
Number of stages in table allocation: 1
  Number of stages for ingress table allocation: 1
  Number of stages for egress table allocation: 0
Critical path length through the table dependency graph: 1
Number of tables allocated: 3
+-------+-----------------------------+
|Stage  |Table Name                   |
+-------+-----------------------------+
|0      |tbl_tna_register156          |
|0      |SwitchIngress.reg_match_dir  |
|0      |SwitchIngress.reg_match      |
+-------+-----------------------------+

```


# test2


```
    // Hash computation
    Hash<bit<16>>(HashAlgorithm_t.RANDOM) hash_function;

    action compute_hash() {
        ig_md.index = (bit<10>)hash_function.get({
            hdr.ipv4.src_addr,
            hdr.ipv4.dst_addr,
            hdr.tcp.src_port,
            hdr.tcp.dst_port,
            hdr.ipv4.protocol
        });
    }

    table tbl_compute_hash {
        actions = {
            compute_hash;
        }
        const default_action = compute_hash();
        size = 1;
    }

    // Bloom filter instantiation
    Register<register_max_count, register_num_entries>(BLOOM_FILTER_ENTRIES) counting_bloom_filter;

    RegisterAction<register_max_count, register_num_entries,  bit<1>>(counting_bloom_filter)
    read_and_update_bloom_filter = {
        void apply(inout register_max_count data, out bit<1> counter_exceeded) {
            counter_exceeded = 0;
            if (data > PACKET_THRESHOLD) {
                counter_exceeded = 1;
            }
            data = data + 1;
        }
    };

    action update_bloom_filter() {
        ig_md.ctr_exceeded = read_and_update_bloom_filter.execute(ig_md.index);
    }

    table tbl_update_bloom_filter {
        actions = {
            update_bloom_filter;
        }
        const default_action = update_bloom_filter();
        size = 1;
    }
```


# 查看方法    


```
bfrt.tofino_nat64.pipe.SwitchIngress.counting_bloom_filter> info
----------------------------------------------------------> info()
Table Name: counting_bloom_filter
Full Name: pipe.SwitchIngress.counting_bloom_filter
Type: REGISTER
Usage: n/a
Capacity: 1024

Key Fields:
Name             Type      Size  Required    Read Only
---------------  ------  ------  ----------  -----------
$REGISTER_INDEX  EXACT       32  True        False

Data Fields:
Name                                    Type           Size  Required    Read Only
--------------------------------------  -----------  ------  ----------  -----------
SwitchIngress.counting_bloom_filter.f1  BYTE_STREAM      32  True        False

```

## dump json    
```
bfrt.tofino_nat64.pipe.SwitchIngress.counting_bloom_filter> dump(json=True, from_hw=True)
```


```
import json
import sys
import binascii


ID_table = bfrt.cocoSketch.pipe.Ingress.counter_ID
ID_text = ID_table.dump(json=True, from_hw=True)
IDs = json.loads(ID_text)

counter_table = bfrt.cocoSketch.pipe.Ingress.counter_count
counter_text = counter_table.dump(json=True, from_hw=True)
counters = json.loads(counter_text)

with open('/root/cocosketch/input.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        print('Query ID: ' + line.strip())
        flow_id = int(line.strip())
        hash_code = binascii.crc32(flow_id.to_bytes(4, byteorder='big'), 0x00000000) & 0xFFFFFFFF
        pos = hash_code % 65536
        
        if IDs[pos]['data']['Ingress.counter_ID.f1'][0] == flow_id:
            print('pipe 0: ' + str(counters[pos]['data']['Ingress.counter_count.f1'][0]))
        else:
            print('pipe 0: 0')

        if IDs[pos]['data']['Ingress.counter_ID.f1'][1] == flow_id:
            print('pipe 1: ' + str(counters[pos]['data']['Ingress.counter_count.f1'][1]))
        else:
            print('pipe 1: 0')

```

##  get_key   

```

bfrt.tofino_nat64.pipe.SwitchIngress.counting_bloom_filter> get_key(0x21)
$REGISTER_INDEX=33, gress_dir=255, pipe=65535, prsr_id=255
Entry 0:
Entry key:
    $REGISTER_INDEX                : 0x00000021

Out[18]: ({b'$REGISTER_INDEX': 33}, <bfrtcli.CIntfBFRT.BfDevTgt at 0x7f07b50bd740>)

bfrt.tofino_nat64.pipe.SwitchIngress.counting_bloom_filter> get_key(0x21)
$REGISTER_INDEX=33, gress_dir=255, pipe=65535, prsr_id=255
Entry 0:
Entry key:
    $REGISTER_INDEX                : 0x00000021

Out[19]: ({b'$REGISTER_INDEX': 33}, <bfrtcli.CIntfBFRT.BfDevTgt at 0x7f07b509aac0>)
```