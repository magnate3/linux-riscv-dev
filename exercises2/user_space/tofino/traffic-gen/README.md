

# scapy 


```
apt-get -y install python3-scapy
```

# 面板编号 --> dp 编号

```
# For getting the dev_ports from the front panel ports itself
def get_devport(frontpanel, lane):
    port_hdl_info = bfrt_info.table_get("$PORT_HDL_INFO")
    key = port_hdl_info.make_key(
        [gc.KeyTuple("$CONN_ID", frontpanel), gc.KeyTuple("$CHNL_ID", lane)]
    )   
    for data, _ in port_hdl_info.entry_get(target, [key], {"from_hw": False}):
        devport = data.to_dict()["$DEV_PORT"]
        if devport:
            return devport
```

# gen port
```
def port_to_pipe(port):
    local_port = port & 0x7F
    pipe = port >> 7
    return pipe


def make_port(pipe, local_port):
    return (pipe << 7) | local_port
def pgen_port(pipe_id):
    """
    Given a pipe return a port in that pipe which is usable for packet
    generation.  Note that Tofino allows ports 68-71 in each pipe to be used for
    packet generation while Tofino2 allows ports 0-7.  This example will use
    either port 68 or port 6 in a pipe depending on chip type.
    """
    if g_is_tofino:
        pipe_local_port = 68
    if g_is_tofino2:
        pipe_local_port = 6
    return make_port(pipe_id, pipe_local_port)
```




# test
```
$SDE/run_switchd.sh -p pipoTG   --arch tf2
```

```
$SDE/run_bfshell.sh -f ../config-100g.txt
```


#  tableEntries.py    

+ port for pktgen
```
 src_port = 6
```
+ 入口

```
i_port = 8
```

+ 出口    
```
 o_port = 24     # HW port to send the packets
```

```
root@debian:~/PIPO-TG-main/files# python3 tableEntries.py 
Unable to init server: Could not connect: Connection refused
Unable to init server: Could not connect: Connection refused

(tableEntries.py:4906): Gdk-CRITICAL **: 03:07:43.527: gdk_cursor_new_for_display: assertion 'GDK_IS_DISPLAY (display)' failed

(tableEntries.py:4906): Gdk-CRITICAL **: 03:07:43.528: gdk_cursor_new_for_display: assertion 'GDK_IS_DISPLAY (display)' failed
WARNING:root:ROCEv2 support not found in Scapy
WARNING:root:ERSPAN support not found in Scapy
WARNING:root:GENEVE support not found in Scapy
Using packet manipulation module: ptf.packet_scapy
Subscribe attempt #1
INFO:bfruntime_grpc_client:Subscribe attempt #1
Subscribe response received 0
INFO:bfruntime_grpc_client:Subscribe response received 0
Connected to BF Runtime Server
Received pipoTG on GetForwarding on client 0, device 0
INFO:bfruntime_grpc_client:Received pipoTG on GetForwarding on client 0, device 0
The target runs program  pipoTG
Binding with p4_name pipoTG
INFO:bfruntime_grpc_client:Binding with p4_name pipoTG
Binding with p4_name pipoTG successful!!
INFO:bfruntime_grpc_client:Binding with p4_name pipoTG successful!!
clean timer table
configure timer table
Create packet
enable pktgen port
configure pktgen application
configure packet buffer
enable pktgen
```

#  bfrt.tf2.pktgen.port_cfg   


```
bfrt.tf2.pktgen.port_cfg> get(6)
Entry 0:
Entry key:
    dev_port                       : 0x00000006
Entry data:
    recirculation_enable           : True
    pktgen_enable                  : True
    pattern_matching_enable        : True
    clear_port_down_enable         : False

Out[20]: Entry for tf2.pktgen.port_cfg table.

bfrt.tf2.pktgen.port_cfg> 
```


```

bfrt.tf2.pktgen.port_cfg> exit
bf_rt cli exited normally.
Starting UCLI from bf-shell 

Cannot read termcap database;
using dumb terminal settings.
bf-sde> rate-period 1
bf-sde> rate-show
```