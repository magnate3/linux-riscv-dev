

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