# $language = "python"
# $interface = "1.0"
from ipaddress import ip_address
import re
import time
import random
import math
import linecache
import sys
import os

pre = bfrt.pre
def clear_all():
    global pre

    for table in pre.info(return_info=True, print_info=False):
        if table['type'] in ['PRE_MGID']:
            print("Clearing PRE_MGID {}".format(table['full_name']))
            for entry in table['node'].get(regex=True):
                entry.remove()
                
    for table in pre.info(return_info=True, print_info=False):
        if table['type'] in ['PRE_NODE']:
            print("Clearing PRE_MGID {}".format(table['full_name']))
            for entry in table['node'].get(regex=True):
                entry.remove()  

clear_all()

g_dev_port = [
392,
400,
408,
416,
424,
432,
440,
448,
 64,
 56,
 48,
 40,
 32,
 24,
 16,
  8,
136,
144,
152,
160,
168,
176,
192,
184,
312,
320,
296,
304,
288,
280,
264,
272] 


#g_dev_port[0] 就是port 0 g_dev_port[31] 就是port 31  或者手动填写devport
in_devport = g_dev_port[0]
table_index = 2
out_devport_array=[
g_dev_port[1],
g_dev_port[3],
g_dev_port[5],
g_dev_port[7],
g_dev_port[9],
g_dev_port[11],
g_dev_port[13],
g_dev_port[15],
g_dev_port[17],
g_dev_port[19],
g_dev_port[21],
g_dev_port[23],
g_dev_port[25],
g_dev_port[27],
g_dev_port[29],
g_dev_port[31]]

port_meta_table = bfrt.switch.pipe.SwitchIngressParser.PORT_METADATA
ipv4_acl_table = bfrt.switch.pipe.SwitchIngress.ingress_ipv4_acl.acl
#nexthop_table = bfrt.switch.pipe.nexthop
nexthop_table = bfrt.switch.pipe.SwitchIngress.nexthop.nexthop
#port_map_table = bfrt.switch.pipe.ingress_port_mapping
port_map_table  = bfrt.switch.pipe.SwitchIngress.ingress_port_mapping.port_mapping
#sys_acl_table = bfrt.switch.pipe.ingress_system_acl
sys_acl_table = bfrt.switch.pipe.SwitchIngress.system_acl.system_acl

#清空已添加的acl表    
ipv4_acl_table.clear()    
nexthop_table.clear()

#清空端口映射表
port_map_table.clear()
#清除系统acl
sys_acl_table.clear()

    
print("*******************in_devport:%s out_devport_array:%s*****************" % (in_devport, out_devport_array)) 
   


#根据ingress port 4 设置port_lag_label 为1
port_meta_table.add(ingress_port=in_devport, port_lag_index=table_index, port_lag_label=table_index)

print("*******************add PORT_METADATA OK in_devport:%s port_lag_label:%s*****************" % (in_devport, table_index)) 

#根据port_lag_label 为1 设置nexthop_index为1
ipv4_acl_table.add_with_redirect_nexthop(port_lag_label=table_index, nexthop_index=table_index)
ipv4_acl_table.mod_with_redirect_nexthop(port_lag_label=table_index, nexthop_index=table_index)

print("*******************add ipv4 acl OK port_lag_label:%s nexthop_index:%s*****************" % (table_index, table_index)) 


#根据nexthop_index为1 设置mgid 为5
nexthop_table.add_with_set_nexthop_properties_post_routed_flood(nexthop=table_index, mgid=table_index+5)
nexthop_table.mod_with_set_nexthop_properties_post_routed_flood(nexthop=table_index, mgid=table_index+5)

print("*******************add nexthop OK nexthop:%s mgid:%s*****************" % (table_index, table_index+5)) 

#根据node id为5 设置出端口为100
bfrt.pre.node.add(MULTICAST_NODE_ID=table_index+5, DEV_PORT=out_devport_array)
bfrt.pre.node.mod(MULTICAST_NODE_ID=table_index+5, DEV_PORT=out_devport_array)

print("*******************add node id OK MULTICAST_NODE_ID:%s out_devport_array:%s*****************" % (table_index+5, out_devport_array)) 

#根据mgid为5设置node id为5
bfrt.pre.mgid.add(MGID=table_index+5, MULTICAST_NODE_ID=[table_index+5], MULTICAST_NODE_L1_XID=[0], MULTICAST_NODE_L1_XID_VALID=[0])
bfrt.pre.mgid.mod(MGID=table_index+5, MULTICAST_NODE_ID=[table_index+5], MULTICAST_NODE_L1_XID=[0], MULTICAST_NODE_L1_XID_VALID=[0])
print("*******************add mgid OK MGID:%s MULTICAST_NODE_ID:%s*****************" % (table_index+5, table_index+5))

for portindex in range(0,32):  
    in_port = g_dev_port[portindex]
    out_port = g_dev_port[portindex]
    port_label = portindex+1+table_index
    
    if portindex == 0:
        print("*******************not add in_port:%s out_port:%s port_label:%s*****************" % (in_port, out_port, port_label))
        continue 
        
    if portindex % 2 == 1:
        print("*******************not add in_port:%s out_port:%s port_label:%s*****************" % (in_port, out_port, port_label))
        continue 
        
    print("*******************in_port:%s out_port:%s port_label:%s*****************" % (in_port, out_port, port_label)) 
    port_meta_table.add(ingress_port=in_port, port_lag_index=port_label, port_lag_label=port_label)
    ipv4_acl_table.add_with_redirect_nexthop(port_lag_label=port_label, nexthop_index=port_label)
    nexthop_table.add_with_set_nexthop_properties_post_routed_flood(nexthop=port_label, mgid=port_label+5)
    bfrt.pre.node.add(MULTICAST_NODE_ID=port_label+5, DEV_PORT=[out_port])
    bfrt.pre.node.mod(MULTICAST_NODE_ID=port_label+5, DEV_PORT=[out_port])
    bfrt.pre.mgid.add(MGID=port_label+5, MULTICAST_NODE_ID=[port_label+5], MULTICAST_NODE_L1_XID=[0], MULTICAST_NODE_L1_XID_VALID=[0])
    bfrt.pre.mgid.mod(MGID=port_label+5, MULTICAST_NODE_ID=[port_label+5], MULTICAST_NODE_L1_XID=[0], MULTICAST_NODE_L1_XID_VALID=[0])
    
#print("""*******************DUMP RESULTS*****************""")

#bfrt.switch.pipe.SwitchIngressParser.PORT_METADATA.get(ingress_port=in_devport)
#bfrt.switch.pipe.SwitchIngress.ingress_ipv4_acl.acl.dump()
#bfrt.switch.pipe.nexthop.dump()
#bfrt.pre.mgid.dump()
#bfrt.pre.node.dump()

# Final programming
print("""*******************PROGAMMING RESULTS SUCCESS*****************""")





