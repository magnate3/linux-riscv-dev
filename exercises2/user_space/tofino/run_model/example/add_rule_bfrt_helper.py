import signal
import struct
from queue import Queue
from queue import Empty
from threading import Thread

import json
import grpc
import os

import bfrt_helper.pb2.bfruntime_pb2 as bfruntime_pb2
import bfrt_helper.pb2.bfruntime_pb2_grpc as bfruntime_pb2_grpc

from bfrt_helper.bfrt import BfRtHelper
from bfrt_helper.bfrt_info import BfRtInfo

HOST = "127.0.0.1:50052"
DEVICE_ID = 0
CLIENT_ID = 0

HOME = os.getenv('HOME', None)
SDE_INSTALL = os.getenv('SDE_INSTALL', None)

PROGRAM_NAME = "patch_panel"
# CTX_PATH = f'{HOME}/example/{PROGRAM_NAME}.tofino/pipe/context.json'
# BIN_PATH = f'{HOME}/example/{PROGRAM_NAME}.tofino/pipe/tofino.bin'
BFRT_PATH = f'{SDE_INSTALL}/share/tofinopd/{PROGRAM_NAME}/bf-rt.json'


channel = grpc.insecure_channel(HOST)
client = bfruntime_pb2_grpc.BfRuntimeStub(channel)

stream_out_queue = Queue()  # Stream request channel (self._stream),
stream_in_queue = Queue()  # Receiving messages from device


def stream_req_iterator():
    while True:
        p = stream_out_queue.get()
        if p is None:
            break
        print("Stream sending: ", p)
        yield p


def stream_recv(stream):
    try:
        for p in stream:
            print("Stream received: ", p)
            stream_in_queue.put(p)
    except Exception as e:
        print(str(e))


stream = client.StreamChannel(stream_req_iterator())
stream_recv_thread = Thread(target=stream_recv, args=(stream,))
stream_recv_thread.start()


def close(sig, frame):
    stream_out_queue.put(None)
    stream_recv_thread.join()


signal.signal(signal.SIGINT, close)


bfrt_data = json.loads(open(BFRT_PATH).read())
bfrt_info = BfRtInfo(bfrt_data)
bfrt_helper = BfRtHelper(DEVICE_ID, CLIENT_ID, bfrt_info)


request = bfrt_helper.create_subscribe_request()
stream_out_queue.put(request)
stream_in_queue.get()


from bfrt_helper.fields import PortId
from bfrt_helper.match import Exact

INGRESS_PORT = 0
EGRESS_PORT = 1

write_request = bfrt_helper.create_table_write( 
    program_name='patch_panel', 
    table_name='pipe.PatchPanelIngressControl.port_forward',
    key={
        'ig_intr_md.ingress_port': Exact(PortId(INGRESS_PORT))
    },
    action_name='PatchPanelIngressControl.forward',
    action_params={
        'egress_port': PortId(EGRESS_PORT),
    })

response = client.Write(write_request)
print(response)


stream_out_queue.put(None)
stream_recv_thread.join()