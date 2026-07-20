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

#forward table
output_port = 156      #有待修改
match_table_forward = bfrt_info.table_get("Ingress.table_forward")
match_table_forward.info.key_field_annotation_add("hdr.ipv4.dst_addr", "ipv4")
try:
    match_table_forward.entry_add(
        target,
        [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr","10.0.0.4")])],
        [match_table_forward.make_data([client.DataTuple("port",156)],action_name = "forward")]
    )
    match_table_forward.entry_add(
        target,
        [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr","10.0.0.1")])],
        [match_table_forward.make_data([client.DataTuple("port",132)],action_name = "forward")]
    )
    match_table_forward.entry_add(
        target,
        [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr","10.0.0.2")])],
        [match_table_forward.make_data([client.DataTuple("port",140)],action_name = "forward")]
    )
    match_table_forward.entry_add(
        target,
        [match_table_forward.make_key([client.KeyTuple("hdr.ipv4.dst_addr","10.0.0.3")])],
        [match_table_forward.make_data([client.DataTuple("port",148)],action_name = "forward")]
    )
finally:
    pass

flowindex = 0
dstport_current = 8010
srcport_current = 9010
tcpflownum = 3
tcp_flows = {}

tcp_addr=["10.0.0.1","10.0.0.2","10.0.0.3"]
tcp_weights = [2,16,32]
dst_addr="10.0.0.4"

# for i in range(0,tcpflownum):
#     tcp_flows[flowindex] = flow(flowindex,tcp_addr[i%tcpflownum],srcport_current,dst_addr,dstport_current,"TCP",tcp_weights[i])
#     flowindex+=1
#     dstport_current+=1
#     srcport_current+=1
    
tcpFlowNumPerHost=15
dst_port_start=[8100,8200,8300]
src_port_start=[9100,9200,9300]
weights_per_host=[2,16,32]
for host_index in range(3):    
    for i in range(tcpFlowNumPerHost):
        weight_index=i*3//tcpFlowNumPerHost
        tcp_flows[flowindex] = flow(flowindex,tcp_addr[host_index],src_port_start[host_index],dst_addr,dst_port_start[host_index],"TCP",weights_per_host[weight_index])
        # tcp_flows[flowindex] = flow(flowindex,tcp_addr[host_index],src_port_start[host_index],dst_addr,dst_port_start[host_index],"TCP",weights_per_host[host_index])
        flowindex+=1
        src_port_start[host_index]+=1
        dst_port_start[host_index]+=1

#Ingress
#data packet
#table get TCP index
match_table_TCPIndex = bfrt_info.table_get("Ingress.get_weightindex_TCP_table")
match_table_TCPIndex.info.key_field_annotation_add("hdr.ipv4.src_addr", "ipv4")
try:
    key_src = "hdr.ipv4.src_addr"
    key_dst = "hdr.tcp.dst_port"
    for tcpflow in tcp_flows.values():
        match_table_TCPIndex.entry_add(
            target,
            [match_table_TCPIndex.make_key([client.KeyTuple(key_src,tcpflow.srcIP),client.KeyTuple(key_dst,tcpflow.dstPort)])],
            [match_table_TCPIndex.make_data([client.DataTuple("flow_idx",tcpflow.index)],action_name = "Ingress.get_weightindex_TCP")]
        )
finally:
    pass


# #get weight 1/wf  TCP
# match_table_getweight = bfrt_info.table_get("Ingress.get_weight_table")
# try:
#     keyname = "meta.flow_index"
#     for tcpflow in tcp_flows.values():
#         match_table_getweight.entry_add(
#             target,
#             [match_table_getweight.make_key([client.KeyTuple(keyname,tcpflow.index)])],
#             [match_table_getweight.make_data([client.DataTuple("weight",tcpflow.weight)],action_name = "Ingress.get_weight_action")]
#         )
# finally:
#     pass

#get finish_time_add
match_table_finishTime = bfrt_info.table_get("Ingress.update_and_get_f_finish_time")
# try:
#     keyname = "meta.flow_index"
#     for i in range(3):
#         match_table_finishTime.entry_add(
#             target,
#             [match_table_finishTime.make_key([client.KeyTuple(keyname,i)])],
#             [match_table_finishTime.make_data([client.DataTuple("flow_index",i)],action_name = "Ingress.update_and_get_f_finish_time"+str(tcp_weights[i]))]
#         )
# finally:
#     pass
try:
    keyname = "meta.flow_index"
    for host_index in range(3):    
        for i in range(tcpFlowNumPerHost):
            weight_index=i*3//tcpFlowNumPerHost
            cur_index=host_index*tcpFlowNumPerHost+i
            match_table_finishTime.entry_add(
                target,
                [match_table_finishTime.make_key([client.KeyTuple(keyname,cur_index)])],
                [match_table_finishTime.make_data([client.DataTuple("flow_index",cur_index)],action_name = "Ingress.update_and_get_f_finish_time"+str(weights_per_host[weight_index]))] 
                # [match_table_finishTime.make_data([client.DataTuple("flow_index",cur_index)],action_name = "Ingress.update_and_get_f_finish_time"+str(weights_per_host[host_index]))] 
            )
finally:
    pass

#get queue_length
match_table_queueLength = bfrt_info.table_get("Ingress.queue_length_lookup")
try:
    keyname = "meta.available_queue"
    for i in range(0,20):
        available_queue_start = 1<<i
        available_queue_end = (1<<(i+1))-1
        # available_queue = 2**(i-1)
        # available_queue_mask = bytearray(((1 << i) - 1).to_bytes(2, byteorder='big'))
        exponent_value = int(i)
        match_table_queueLength.entry_add(
            target,
            [match_table_queueLength.make_key([client.KeyTuple(keyname,low=available_queue_start,high=available_queue_end)])],
            [match_table_queueLength.make_data([client.DataTuple("exponent_value",exponent_value)],action_name = "Ingress.set_exponent_buffer")]
        )
finally:
    pass

#get max_min
match_table_MaxMin = bfrt_info.table_get("Ingress.max_min_lookup")
try:
    keyname = "meta.max_min"
    for i in range(0,16):
        max_min_start = 1<<i
        max_min_end = (1<<(i+1))-1
        # max_min_mask = bytearray(((1 << i) - 1).to_bytes(2, byteorder='big'))
        exponent_value = int(i)
        match_table_MaxMin.entry_add(
            target,
            [match_table_MaxMin.make_key([client.KeyTuple(keyname,low=max_min_start,high=max_min_end)])],
            [match_table_MaxMin.make_data([client.DataTuple("exponent_value",exponent_value)],action_name = "Ingress.set_exponent_max_min")]
        )
finally:
    pass

#get max_min_buffer
match_table_MaxMinBuffer = bfrt_info.table_get("Ingress.max_min_buffer_lookup")
try:
    key_max_min = "meta.max_min_exponent"
    key_buffer = "meta.buffer_exponent"
    for i in range(0,17):
        for j in range(0,22):
            mul=i+j
            match_table_MaxMinBuffer.entry_add(
                target,
                [match_table_MaxMinBuffer.make_key([client.KeyTuple(key_max_min,i),client.KeyTuple(key_buffer,j)])],
                [match_table_MaxMinBuffer.make_data([client.DataTuple("mul",mul)],action_name = "Ingress.calculate_max_min_buffer_mul")]
            )
finally:
    pass

#get dividend
match_table_dividend = bfrt_info.table_get("Ingress.dividend_lookup")
try:
    keyname = "meta.dividend"
    for i in range(0,16):
        dividend_start = 1<<i
        dividend_end = (1<<(i+1))-1
        # dividend_mask = bytearray(((1 << i) - 1).to_bytes(2, byteorder='big'))
        exponent_value = int(i)
        match_table_dividend.entry_add(
            target,
            [match_table_dividend.make_key([client.KeyTuple(keyname,low=dividend_start,high=dividend_end)])],
            [match_table_dividend.make_data([client.DataTuple("exponent_value",exponent_value)],action_name = "Ingress.set_exponent_dividend")]
        )
finally:
    pass




#worker packet
#nothing

