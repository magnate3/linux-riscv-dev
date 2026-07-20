import logging
import random
import struct
import sys
import os
from decimal import Decimal

sys.path.append(os.path.expandvars('$SDE/install/lib/python3.8/site-packages/tofino/'))
sys.path.append(os.path.expandvars('$SDE/install/lib/python3.8/site-packages/p4testutils/'))
sys.path.append(os.path.expandvars('$SDE/install/lib/python3.8/site-packages/bf-ptf/'))
from ptf import config
import ptf.testutils as testutils
from bfruntime_client_base_tests import BfRuntimeTest
from bfrt_grpc import client
GRPC_CLIENT=client.ClientInterface(grpc_addr="localhost:50052", client_id=0,device_id=0)
bfrt_info=GRPC_CLIENT.bfrt_info_get(p4_name=None)
GRPC_CLIENT.bind_pipeline_config(p4_name=bfrt_info.p4_name)
tables = bfrt_info.table_dict
target = client.Target(device_id=0, pipe_id=0xFFFF)


class flow:
    def __init__(self,index,srcIP,srcPort,dstIP,dstPort,type,weight) -> None:
        self.index = index
        self.srcIP = srcIP
        self.srcPort = srcPort
        self.dstIP = dstIP
        self.dstPort = dstPort
        self.type = type
        self.weight = weight        #此处的weight指的是向右移位的位数，所以越大代表真实的weight越小
    def compute_limit(self,Q,R):
        self.limit = int(Decimal((int)(Q*1.0/(R*pow(2,self.weight)))).quantize(Decimal("1."), rounding = "ROUND_HALF_UP"))




flowindex = 1
tcp_flowindex = 11
dstport_current = 9000      #udp from 9001
                            #tcp from 5001
srcport_current = 8000
flownum = 4
tcpflownum = 0
udpflownum = 3
tcp_flows = {}
udp_flows = {}

tcp_addr=["10.0.0.1"]
tcp_weights = [3]
udp_weights = [1,2,2]
udp_addr=["10.0.0.1","10.0.0.2","10.0.0.3"]
dst_addr="10.0.0.4"

Q = 24000       #因为10G的port的一个queue只有大约16个满载包大小
R = 1500

for i in range(0,tcpflownum):
    tcp_flows[tcp_flowindex] = flow(tcp_flowindex,tcp_addr[i%tcpflownum],srcport_current,dst_addr,dstport_current,"TCP",tcp_weights[i])
    tcp_flowindex+=1
    dstport_current+=1
    srcport_current+=1
for i in range(0,udpflownum):
    udp_flows[flowindex] = flow(flowindex,udp_addr[i%udpflownum],srcport_current,dst_addr,dstport_current,"UDP",udp_weights[i%udpflownum])
    flowindex+=1
    dstport_current+=1
    srcport_current+=1

#Ingress


#forward table
output_port = 156       #有待修改
tcp_port = 132          #有待修改
match_table_forward = bfrt_info.table_get("SwitchIngress.table_forward")
match_table_forward.info.key_field_annotation_add("hdr.ipv4.dst_addr", "ipv4")
try:
    # match_table_forward.entry_add(
    #     target,
    #     [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr","10.0.0.4"),client.KeyTuple("hdr.worker_t.$valid",0)])],
    #     [match_table_forward.make_data([client.DataTuple("port",output_port)],action_name = "forward")]
    # )
    # match_table_forward.entry_add(
    #     target,
    #     [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr","10.0.0.200"),client.KeyTuple("hdr.worker_t.$valid",1)])],
    #     [match_table_forward.make_data([client.DataTuple("port",164)],action_name = "forward")]
    # )

    # match_table_forward.entry_add(
    #     target,
    #     [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr",tcp_addr[0]),client.KeyTuple("hdr.worker_t.$valid",0)])],
    #     [match_table_forward.make_data([client.DataTuple("port",tcp_port)],action_name = "forward")]
    # )

    match_table_forward.entry_add(
        target,
        [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr","10.0.0.4")])],
        [match_table_forward.make_data([client.DataTuple("port",output_port)],action_name = "forward")]
    )
    match_table_forward.entry_add(
        target,
        [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr",tcp_addr[0])])],
        [match_table_forward.make_data([client.DataTuple("port",tcp_port)],action_name = "forward")]
    )


finally:
    pass

#data packet
#table get TCP index        //tcp index from 11
match_table_TCPIndex = bfrt_info.table_get("SwitchIngress.get_weightindex_TCP_table")
match_table_TCPIndex.info.key_field_annotation_add("hdr.ipv4.src_addr", "ipv4")
try:
    key_src = "hdr.ipv4.src_addr"
    key_dst = "hdr.tcp.dst_port"
    for tcpflow in tcp_flows.values():
        match_table_TCPIndex.entry_add(
            target,
            [match_table_TCPIndex.make_key([client.KeyTuple(key_src,tcpflow.srcIP),client.KeyTuple(key_dst,tcpflow.dstPort)])],
            [match_table_TCPIndex.make_data([client.DataTuple("flow_idx",tcpflow.index)],action_name = "SwitchIngress.get_weightindex_TCP")]
        )
finally:
    pass

#table get UDP index    //udp from 1
match_table_UDPIndex = bfrt_info.table_get("SwitchIngress.get_weightindex_UDP_table")
match_table_UDPIndex.info.key_field_annotation_add("hdr.ipv4.src_addr", "ipv4")
try:
    key_src = "hdr.ipv4.src_addr"
    key_dst = "hdr.udp.dst_port"
    for udpflow in udp_flows.values():
        match_table_UDPIndex.entry_add(
            target,
            [match_table_UDPIndex.make_key([client.KeyTuple(key_src,udpflow.srcIP),client.KeyTuple(key_dst,udpflow.dstPort)])],
            [match_table_UDPIndex.make_data([client.DataTuple("flow_idx",udpflow.index)],action_name = "SwitchIngress.get_weightindex_UDP")]
        )
finally:
    pass


#get weight 1/wf
match_table_getweight = bfrt_info.table_get("SwitchIngress.get_weight_table")
try:
    keyname = "meta.flow_index"
    for tcpflow in tcp_flows.values():
        match_table_getweight.entry_add(
            target,
            [match_table_getweight.make_key([client.KeyTuple(keyname,tcpflow.index)])],
            [match_table_getweight.make_data([client.DataTuple("weight",tcpflow.weight)],action_name = "SwitchIngress.get_weight_action")]
        )
    for udpflow in udp_flows.values():
        match_table_getweight.entry_add(
            target,
            [match_table_getweight.make_key([client.KeyTuple(keyname,udpflow.index)])],
            [match_table_getweight.make_data([client.DataTuple("weight",udpflow.weight)],action_name = "SwitchIngress.get_weight_action")]
        )
finally:
    pass


#get_limit_table

# match_table_limit = bfrt_info.table_get("SwitchIngress.get_limit_table")
# try:
#     keyname = "meta.flow_index"
#     for tcpflow in tcp_flows.values():
#         tcpflow.compute_limit(Q,R)
#         match_table_limit.entry_add(
#             target,
#             [match_table_limit.make_key([client.KeyTuple(keyname,tcpflow.index)])],
#             [match_table_limit.make_data([client.DataTuple("limit",tcpflow.limit)],action_name = "SwitchIngress.get_limit_action")]
#         )
#     for udpflow in udp_flows.values():
#         udpflow.compute_limit(Q,R)
#         match_table_limit.entry_add(
#             target,
#             [match_table_limit.make_key([client.KeyTuple(keyname,udpflow.index)])],
#             [match_table_limit.make_data([client.DataTuple("limit",udpflow.limit)],action_name = "SwitchIngress.get_limit_action")]
#         )
# finally:
#     pass
match_table_limit = bfrt_info.table_get("SwitchIngress.get_limit_table")
try:
    keyname = "meta.weight"
    for i in range(0,5):
        # limit = Q/R>>i
        limit = 16>>i
        match_table_limit.entry_add(
            target,
            [match_table_limit.make_key([client.KeyTuple(keyname,i)])],
            [match_table_limit.make_data([client.DataTuple("limit",limit)],action_name = "SwitchIngress.get_limit_action")]
        )
finally:
    pass

#table get rwf
match_table_rwf = bfrt_info.table_get("SwitchIngress.tbl_get_rwf")
try:
    keyname = "meta.weight"
    for i in range(0,9):
        match_table_rwf.entry_add(
            target,
            [match_table_rwf.make_key([client.KeyTuple(keyname,i)])],
            [match_table_rwf.make_data([], 'SwitchIngress.shift_r_'+str(i))]
        )
finally:
    pass

#worker packet
#nothing

# #Egress
start = 0
add = 1499
starts = []
ends = []
values = []
for i in range(0,16):
    starts.append(i*1500)
    ends.append(starts[i]+1499) 
    values.append(int(Decimal((1500/R)*(16*1.0/(i+1))).quantize(Decimal("1."), rounding = "ROUND_HALF_UP"))) 

match_table_roundadd = bfrt_info.table_get("SwitchEgress.get_round_add_tbl")
try:
    keyname = "eg_intr_md.deq_qdepth"
    for i in range(0,16):
        match_table_roundadd.entry_add(
            target,
            [match_table_roundadd.make_key([client.KeyTuple(keyname,low = starts[i],high = ends[i])])],
            [match_table_roundadd.make_data([client.DataTuple("ra",values[i])],action_name = "SwitchEgress.get_round_add_action")]
        )
finally:
    pass


# # for k,v in bfrt_info.table_dict.values():
# #     print(str(k)+","+str(v))