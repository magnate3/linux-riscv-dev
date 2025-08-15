#!/usr/bin/env python
import sys
import os
import time
sys.path.append(os.path.expandvars('$SDE/install/lib/python3.6/site-packages/tofino/'))
sys.path.append(os.path.expandvars('$SDE/install/lib/python3.6/site-packages/'))
sys.path.append(os.path.expandvars('$SDE/install/lib/python3.6/site-packages/bf_ptf/'))
import grpc
import bfrt_grpc.bfruntime_pb2 as bfruntime_pb2
import bfrt_grpc.client as gc

import ptf.testutils as testutils

# Connect to BF Runtime Server
interface = gc.ClientInterface(grpc_addr = "localhost:50052",
                              client_id = 0,
                              device_id = 0)
print('Connected to BF Runtime Server')

# Get the information about the running program on the bfrt server.
bfrt_info = interface.bfrt_info_get()
print('The target runs program ', bfrt_info.p4_name_get())

# Establish that you are working with this program
interface.bind_pipeline_config(bfrt_info.p4_name_get())

####### You can now use BFRT CLIENT #######
target = gc.Target(device_id=0, pipe_id=0xffff)
t_cfg_table = bfrt_info.table_get("$mirror.cfg")
t_fwd_table = bfrt_info.table_get("t")

# ####### t_table ########
print("clean timer table")
resp = t_fwd_table.entry_get(target, [], {"from_hw": True})
for _, key in resp:
  if key:
      t_fwd_table.entry_del(target, [key])

print("configure timer table")
i_port = 68     # Default port for pktgen
pipe_id = 0
g_timer_app_id = 1
batch_id = [0,1,2,3] # 0,1,2,3
packet_id = [0,1] # 0,1
o_port = 0     # HW port to send the packets

# for i in range(4):
#     for j in range(2):
t_fwd_table.entry_add(
 target,
  [t_fwd_table.make_key([ gc.KeyTuple('ig_intr_md.ingress_port', i_port),
                      gc.KeyTuple('hdr.timer.pipe_id', pipe_id),
                      gc.KeyTuple('hdr.timer.app_id', g_timer_app_id),
                      gc.KeyTuple('hdr.timer.batch_id', batch_id[0]),
                      gc.KeyTuple('hdr.timer.packet_id', packet_id[0])])],
  [t_fwd_table.make_data([gc.DataTuple('port', o_port)],
                      'SwitchIngress.match')]
)

pktgen_app_cfg_table = bfrt_info.table_get("app_cfg")
pktgen_pkt_buffer_table = bfrt_info.table_get("pkt_buffer")
pktgen_port_cfg_table = bfrt_info.table_get("port_cfg")

app_id = g_timer_app_id
pktlen = 1024
pgen_pipe_id = 0
src_port = 68
p_count = 1  # packets per batch
b_count = 1  # batch number
buff_offset = 144  # generated packets' payload will be taken from the offset in buffer

# build expected generated packets
print("Create packet")
p = testutils.simple_eth_packet(pktlen=pktlen)

print("enable pktgen port")

pktgen_port_cfg_table.entry_add(
  target,
  [pktgen_port_cfg_table.make_key([gc.KeyTuple('dev_port', src_port)])],
  [pktgen_port_cfg_table.make_data([gc.DataTuple('pktgen_enable', bool_val=True)])])

# Configure the packet generation timer application
print("configure pktgen application")
data = pktgen_app_cfg_table.make_data([gc.DataTuple('timer_nanosec', 10),
                                gc.DataTuple('app_enable', bool_val=False),
                                gc.DataTuple('pkt_len', (pktlen - 6)),
                                gc.DataTuple('pkt_buffer_offset', buff_offset),
                                gc.DataTuple('pipe_local_source_port', src_port),
                                gc.DataTuple('increment_source_port', bool_val=False),
                                gc.DataTuple('batch_count_cfg', b_count - 1),
                                gc.DataTuple('packets_per_batch_cfg', p_count - 1),
                                gc.DataTuple('ibg', 1),
                                gc.DataTuple('ibg_jitter', 0),
                                gc.DataTuple('ipg', 1000),
                                gc.DataTuple('ipg_jitter', 500),
                                gc.DataTuple('batch_counter', 0),
                                gc.DataTuple('pkt_counter', 0),
                                gc.DataTuple('trigger_counter', 0)],
                                'trigger_timer_periodic')
pktgen_app_cfg_table.entry_mod(
  target,
  [pktgen_app_cfg_table.make_key([gc.KeyTuple('app_id', g_timer_app_id)])],
  [data])


print("configure packet buffer")
pktgen_pkt_buffer_table.entry_mod(
  target,
  [pktgen_pkt_buffer_table.make_key([gc.KeyTuple('pkt_buffer_offset', buff_offset),
                                  gc.KeyTuple('pkt_buffer_size', (pktlen - 6))])],
  [pktgen_pkt_buffer_table.make_data([gc.DataTuple('buffer', bytearray(bytes(p)[6:]))])])  # p[6:]))])


print("enable pktgen")
pktgen_app_cfg_table.entry_mod(
  target,
  [pktgen_app_cfg_table.make_key([gc.KeyTuple('app_id', g_timer_app_id)])],
  [pktgen_app_cfg_table.make_data([gc.DataTuple('app_enable', bool_val=True)],
                                  'trigger_timer_periodic')]
)


time.sleep(10)

print("disable pktgen")
pktgen_app_cfg_table.entry_mod(
  target,
  [pktgen_app_cfg_table.make_key([gc.KeyTuple('app_id', g_timer_app_id)])],
  [pktgen_app_cfg_table.make_data([gc.DataTuple('app_enable', bool_val=False)],
                                  'trigger_timer_one_shot')])
