import os
import sys
import glob

sde_install = os.environ.get('SDE_INSTALL')
tofino_py_path = f'{sde_install}/lib/python3.8/site-packages/tofino'
bfrt_grpc_py_path = f'{sde_install}/lib/python3.8/site-packages/tofino/bfrt_grpc'

sys.path.append(tofino_py_path)
sys.path.append(bfrt_grpc_py_path)


import bfrt_grpc.client as gc

DEVICE_ID = 0
CLIENT_ID = 0
INGRESS_PORT = 0
EGRESS_PORT = 1

interface = gc.ClientInterface(f'127.0.0.1:50052',
        client_id=CLIENT_ID,
        device_id=DEVICE_ID)

target = gc.Target(DEVICE_ID, pipe_id=0xffff)
interface.bind_pipeline_config("patch_panel")
bfrt_info = interface.bfrt_info_get("patch_panel")

table = bfrt_info.table_get('pipe.PatchPanelIngressControl.port_forward')
table.entry_add(target, [
        table.make_key([
            gc.KeyTuple('ig_intr_md.ingress_port', INGRESS_PORT)
        ])
    ], [
        table.make_data([gc.DataTuple('egress_port', EGRESS_PORT)],
                'PatchPanelIngressControl.forward')
    ])