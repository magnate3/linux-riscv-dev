from scapy.all import *
import ipaddress
import random
import sys
import binascii
import socket

sys.path.append(os.path.expandvars('$SDE/install/lib/python3.8/site-packages/tofino/bfrt_grpc'))
sys.path.append(os.path.expandvars('$SDE/install/lib/python3.8/site-packages/tofino/'))
sys.path.append(os.path.expandvars('$SDE/install/lib/python3.8/site-packages/'))

import bfrt_grpc.client as gc

print(sys.version)

grpc_addr = 'localhost:50052'
client_id = 0
device_id = 0
is_master = False
notifications = None
perform_bind = True

interface = gc.ClientInterface(grpc_addr, client_id=1, device_id=0)
bfrt_info = interface.bfrt_info_get()
p4_name = bfrt_info.p4_name_get()
if perform_bind:
    interface.bind_pipeline_config(p4_name)
bfrt_info = interface.bfrt_info_get()
target = gc.Target(device_id=0, pipe_id=0xFFFF)


# ---------- Setting -------------

DATA_BLOCKS = 6     # Number of blocks encrypted

recir_ports = [68, 196]     # Recirculation ports
print('n_recir_port:', len(recir_ports))

# --------------------------------


n_cls = len(recir_ports)
width = 32

th = [(1 << width) * i // n_cls for i in range(n_cls + 1)]

key_set = set()
keys = []
data = []

for i in range(0, n_cls):
    lb = th[i]
    ub = th[i + 1]
    
    for j in reversed(range(width + 1)):
        if ub & (1 << j) == 0:
            continue
        mask = 1 << j
        key = (ub & ~(mask - 1)) ^ mask
        if not (key, width - j) in key_set:
            key_set.add((key, width - j))
            keys.append((key, width - j))
            data.append(i)



i7_table = bfrt_info.table_get('MyIngressControl.tb_i7')

key_list = [i7_table.make_key([
    gc.KeyTuple('hdr.chacha_pre.data_pos', DATA_BLOCKS, 255),
    gc.KeyTuple('meta.recir_random', 0, 0),
])]

data_list = [i7_table.make_data([], 'MyIngressControl.i7_app')]

for i in range(len(keys)):
    key_list += [i7_table.make_key([
        gc.KeyTuple('hdr.chacha_pre.data_pos', 0, 0),
        gc.KeyTuple('meta.recir_random', keys[i][0], ((1 << keys[i][1]) - 1) << (32 - keys[i][1])),
    ])]
    
    data_list += [i7_table.make_data([
        gc.DataTuple('eg_port', recir_ports[data[i]])
    ], 'MyIngressControl.i7')]
    
i7_table.entry_add(target, key_list, data_list)

