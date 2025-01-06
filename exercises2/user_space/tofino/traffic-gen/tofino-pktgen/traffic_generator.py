import os
import sys
import bfrt_grpc.client as gc
from scapy.all import Ether,IP,UDP
import struct
import argparse

SDE_INSTALL   = os.environ['SDE_INSTALL']
SDE_PYTHON2   = os.path.join(SDE_INSTALL, 'lib', 'python2.7', 'site-packages')
sys.path.append(SDE_PYTHON2)
sys.path.append(os.path.join(SDE_PYTHON2, 'tofino'))
sys.path.append(os.path.join(SDE_PYTHON2, 'tofino', 'bfrt_grpc'))

PYTHON3_VER   = '{}.{}'.format(
    sys.version_info.major,
    sys.version_info.minor)
SDE_PYTHON3   = os.path.join(SDE_INSTALL, 'lib', 'python' + PYTHON3_VER,
                             'site-packages')
sys.path.append(SDE_PYTHON3)
sys.path.append(os.path.join(SDE_PYTHON3, 'tofino'))
sys.path.append(os.path.join(SDE_PYTHON3, 'tofino', 'bfrt_grpc'))

# Connect to the BF Runtime Server
for bfrt_client_id in range(10):
    try:
        interface = gc.ClientInterface(
            grpc_addr = 'localhost:50052',
            client_id = bfrt_client_id,
            device_id = 0,
            num_tries = 1)
        print('Connected to BF Runtime Server as client', bfrt_client_id)
        break
    except:
        print('Could not connect to BF Runtime server')
        quit

# Get the information about the running program
bfrt_info = interface.bfrt_info_get()
print('The target runs the program ', bfrt_info.p4_name_get())

# Establish that you are using this program on the given connection
if bfrt_client_id == 0:
    interface.bind_pipeline_config(bfrt_info.p4_name_get())
dev_tgt = gc.Target(0) 

# Getting the pktgen tables
pktgen_buffer = bfrt_info.table_get("tf1.pktgen.pkt_buffer")
pktgen_port = bfrt_info.table_get("tf1.pktgen.port_cfg")
pktgen_app = bfrt_info.table_get("tf1.pktgen.app_cfg")

# Create the parser
parser = argparse.ArgumentParser(description="Test Configuration")

# Add arguments
parser.add_argument('-r', type=int, help="Line rate (Gbps),           Default :  80", default=80)
parser.add_argument('-b', type=int, help="Burst interval (s),         Default :  4", default=4)
parser.add_argument('-s', type=int, help="Size of the packets in (B)  Default :  1024", default=1024)

# Parse the arguments
args = parser.parse_args()

#  Create packet 
p = Ether()/IP(dst="172.184.1.1")/UDP()/(b'\x01'*(args.s - 42))
pkt_len = len(p) - 6

# Configuring pktgen port
pktgen_port_key = pktgen_port.make_key([gc.KeyTuple('dev_port', 196)])
pktgen_port_action_data = pktgen_port.make_data([gc.DataTuple('pktgen_enable', bool_val = True)])
pktgen_port.entry_mod(dev_tgt,[pktgen_port_key],[pktgen_port_action_data])

# Configuring pktgen buffer
offset = 0
pktgen_pkt_buf_key = pktgen_buffer.make_key([gc.KeyTuple('pkt_buffer_offset', offset),gc.KeyTuple('pkt_buffer_size', pkt_len)])
pktgen_pkt_buf_action_data = pktgen_buffer.make_data([gc.DataTuple('buffer', bytearray(bytes(p)[6:]))])
pktgen_buffer.entry_mod(dev_tgt,[pktgen_pkt_buf_key],[pktgen_pkt_buf_action_data])

## Configuring pktgen app
padding = args.s % 4
overhead = 98 + padding
INTER_PACKET_GAP_NS = round((args.s + overhead)*8/args.r)
INTER_BATCH_GAP_NS = 65536*INTER_PACKET_GAP_NS
BATCH_COUNT = round(args.b*1000000000/INTER_BATCH_GAP_NS) - 1

pktgen_app_key = pktgen_app.make_key([gc.KeyTuple('app_id', 0)])
pktgen_app_action_data = pktgen_app.make_data([
    gc.DataTuple('timer_nanosec', 10),
    gc.DataTuple('app_enable',bool_val = True),
    gc.DataTuple('pkt_len', pkt_len),
    gc.DataTuple('pkt_buffer_offset', 0),
    gc.DataTuple('pipe_local_source_port', 68),
    gc.DataTuple('increment_source_port',bool_val = False),
    gc.DataTuple('batch_count_cfg',BATCH_COUNT),
    gc.DataTuple('packets_per_batch_cfg',65535),
    gc.DataTuple('ibg',INTER_BATCH_GAP_NS),
    gc.DataTuple('ibg_jitter', 0),
    gc.DataTuple('ipg', INTER_PACKET_GAP_NS),
    gc.DataTuple('ipg_jitter', 0),
    gc.DataTuple('batch_counter', 0),
    gc.DataTuple('pkt_counter', 0),
    gc.DataTuple('trigger_counter', 0)], 'trigger_timer_one_shot')
pktgen_app.entry_mod(dev_tgt,[pktgen_app_key],[pktgen_app_action_data])
print("Packet generation is completed")
