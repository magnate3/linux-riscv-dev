

# scapy 


```
apt-get -y install python3-scapy
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