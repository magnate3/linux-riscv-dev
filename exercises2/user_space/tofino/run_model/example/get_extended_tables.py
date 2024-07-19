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
BFRT_PATH = f'{SDE_INSTALL}/{PROGRAM_NAME}.tofino/bfrt.json'


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

bfrt_data = json.loads(open(BFRT_PATH).read())
bfrt_info = BfRtInfo(bfrt_data)
bfrt_helper = BfRtHelper(DEVICE_ID, CLIENT_ID, bfrt_info)


request = bfrt_helper.create_subscribe_request()
stream_out_queue.put(request)
stream_in_queue.get()


request = bfrt_helper.create_get_pipeline_request()
response = client.GetForwardingPipelineConfig(request)

program_name = response.config[0].p4_name
data = response.non_p4_config.bfruntime_info.decode("utf-8")
non_p4_config = json.loads(data)

p4_config = None

for config in response.config:
    if program_name == config.p4_name:
        p4_config = json.loads(config.bfruntime_info)
        p4_config.get("tables").extend(non_p4_config.get("tables"))

with open('all.json', 'w') as fd:
    fd.write(json.dumps(p4_config, indent=2))


stream_out_queue.put(None)
stream_recv_thread.join()