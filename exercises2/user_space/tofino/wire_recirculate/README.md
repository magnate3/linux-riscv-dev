# Compile and install
```
cmake $SDE/p4studio/ -DCMAKE_INSTALL_PREFIX=$SDE/install -DCMAKE_MODULE_PATH=$SDE/cmake -DP4_NAME=wire_recirculate -DP4_PATH=<full path of wire_recirculate.p4>
make
sudo make install
```

# Set up the CPU ports
```
sudo ifconfig enp4s0f0 192.168.0.1/24
sudo ip netns add v1
sudo ip link set enp4s0f1 netns v1
sudo ip netns exec ip link set lo up
sudo ip netns exec ifconfig enp4s0f1 192.168.0.2/24
```

# Run SDE and load code
```
$SDE/run_switchd.sh -p wire_recirculate
```

## Port add and set front panel port 4/0 to loopback
```
ucli
pm
port-add 4/0 25G NONE
port-loopback 4/0 mac-near
port-enb 4/0
port-add 57/- 10G NONE
an-set 57/- NONE
port-enb 57/-
show
```

## Install rules
```
rclt_ifce = 134 # Set D_P of the front panel port 4/0 here or simply 68
bfrt.wire_recirculate.pipe.SwitchIngress.tbl_rclt_forward.add_with_act_set_rclt(64, 0xFF, 0, 0xFF, 1, 65, rclt_ifce)
bfrt.wire_recirculate.pipe.SwitchIngress.tbl_rclt_forward.add_with_act_set_rclt(65, 0xFF, 0, 0xFF, 1, 64, rclt_ifce)
bfrt.wire_recirculate.pipe.SwitchIngress.tbl_rclt_forward.add_with_act_continue_rclt(0, 0, 0, 0, 2, rclt_ifce)
bfrt.wire_recirculate.pipe.SwitchIngress.tbl_rclt_forward.add_with_act_clear_rclt(0, 0, 4, 0xFF, 1)

bfrt.wire_recirculate.pipe.SwitchIngress.tbl_wire.add_with_act_wire(64, 65)
bfrt.wire_recirculate.pipe.SwitchIngress.tbl_wire.add_with_act_wire(65, 64)
```

## Ping from another terminal
```
ping 192.168.0.2 -c 5
```

## Query and check recirculation is working
```
bfrt.wire_recirculate.pipe.SwitchIngress.tbl_rclt_forward.dump(from_hw=1)
```

The following result indicates recirculation is working
```
----- tbl_rclt_forward Dump Start -----
Default Entry:
Entry data (action : SwitchIngress.nop):
    $COUNTER_SPEC_BYTES            : 0
    $COUNTER_SPEC_PKTS             : 0

Entry 0:
Entry key:
    ig_intr_md.ingress_port        : ('0x40', '0xFF')
    hdr.rclt.rclt_count            : ('0x0000', '0x00FF')
    $MATCH_PRIORITY                : 1
Entry data (action : SwitchIngress.act_set_rclt):
    dport                          : 0x0041
    iface_rclt                     : 0x86
    $COUNTER_SPEC_BYTES            : 510
    $COUNTER_SPEC_PKTS             : 5

Entry 1:
Entry key:
    ig_intr_md.ingress_port        : ('0x41', '0xFF')
    hdr.rclt.rclt_count            : ('0x0000', '0x00FF')
    $MATCH_PRIORITY                : 1
Entry data (action : SwitchIngress.act_set_rclt):
    dport                          : 0x0040
    iface_rclt                     : 0x86
    $COUNTER_SPEC_BYTES            : 510
    $COUNTER_SPEC_PKTS             : 5

Entry 2:
Entry key:
    ig_intr_md.ingress_port        : ('0x00', '0x00')
    hdr.rclt.rclt_count            : ('0x0000', '0x0000')
    $MATCH_PRIORITY                : 2
Entry data (action : SwitchIngress.act_continue_rclt):
    iface_rclt                     : 0x86
    $COUNTER_SPEC_BYTES            : 3240
    $COUNTER_SPEC_PKTS             : 30

Entry 3:
Entry key:
    ig_intr_md.ingress_port        : ('0x00', '0x00')
    hdr.rclt.rclt_count            : ('0x0004', '0x00FF')
    $MATCH_PRIORITY                : 1
Entry data (action : SwitchIngress.act_clear_rclt):
    $COUNTER_SPEC_BYTES            : 1080
    $COUNTER_SPEC_PKTS             : 10

----- tbl_rclt_forward Dump End -----
```

# Note
Recirculation header just after ethernet works for pipe-loopback mode only. Port 68 in internally configured in that mode. In Tofino 1 pipe-loopback is not supported for any other port. So, we used mac-near mode of loopback With either mac-loopback or physical cable connection recirculate header just after ethernet doesn't work.