#!/usr/bin/env python3

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import sys
import socket
import asyncio

sys.path.append('./gen-py')

from ts_pd_rpc import *
from ts_pd_rpc.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.protocol import TMultiplexedProtocol

MAX_MSG_SIZE = 8192
ETH_P_ALL = 3
CPU_HDR_SIZE = 8

def thrift_connect():
    transport = TSocket.TSocket('localhost', 9090)
    transport = TTransport.TBufferedTransport(transport)
    transport.open()

    bProtocol = TBinaryProtocol.TBinaryProtocol(transport)
    protocol = TMultiplexedProtocol.TMultiplexedProtocol(bProtocol, 'ts')
    return ts.Client(protocol) # TODO: This shouldn't work, ts is undefined

class CPU_Header:
    def __init__(self, buffer=b''):
        self.device_port = None
        self.timestamp = None
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        self.device_port = int.from_bytes(buffer[:2], 'big') & 0x01FF
        self.timestamp = int.from_bytes(buffer[2:8], 'big')

    def bytes(self):
        bytes = b''
        bytes += self.device_port.to_bytes(2, 'big')
        # bytes += self.timestamp.to_bytes(6, 'big')
        bytes += b'\xFF' * 6
        return bytes

class Socket:
    def __init__(self, skt_name, port_list):
        self.ports = {1:1}
        self.skt = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
        self.skt.bind((skt_name, ETH_P_ALL))
        self.skt.setblocking(False)
        self.tofino = thrift_connect()
        self.number_of_ports = 1
        self.map_ports(port_list)

    def map_ports(self, filename):
        if filename:
            f = open(filename)
            lines = f.readlines()
            for i in range(len(lines)):
                self.ports[i + 1] = int(lines[i])
            f.close()
            self.number_of_ports = len(lines)

    def send(self, msg, port_number, get_timestamp=False):
        timestamp = None
        cpu_hdr = CPU_Header()
        cpu_hdr.device_port = self.ports[port_number]
        cpu_hdr.timestamp = get_timestamp # Request Egress Timestamp

        self.skt.send(cpu_hdr.bytes() + msg)
        if get_timestamp:
            timestamp = self.tofino.ts_1588_timestamp_tx_get(0, cpu_hdr.device_port).ts
            # timestamp = time.clock_gettime_ns(time.CLOCK_REALTIME) # TODO: get TS7 from tofino

        return timestamp

    async def recv(self):
        loop = asyncio.get_event_loop()
        msg = await loop.sock_recv(self.skt, MAX_MSG_SIZE)
        cpu_hdr = CPU_Header(msg)
        port_list = [port for (port, d_p) in self.ports.items() if d_p == cpu_hdr.device_port]
        # timestamp = time.clock_gettime_ns(time.CLOCK_REALTIME) # TODO: get TS1 from CPU header
        if len(port_list) == 1:
            return (port_list[0], cpu_hdr.timestamp, msg[CPU_HDR_SIZE:])
        else:
            return await self.recv()
