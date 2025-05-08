#!/usr/bin/env python3

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import socket
import time
import asyncio

MAX_MSG_SIZE = 8192
ETH_P_ALL = 3
CPU_HDR_SIZE = 8

class CPU_Header:
    def __init__(self, buffer=b''):
        self.device_port = None
        self.timestamp = None
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        self.device_port = int.from_bytes(buffer[:2], 'big')
        self.timestamp = int.from_bytes(buffer[2:8], 'big')

    def bytes(self):
        bytes = b''
        bytes += self.device_port.to_bytes(2, 'big')
        bytes += self.timestamp.to_bytes(6, 'big')
        return bytes

class Socket:
    def __init__(self, skt_name, port_list):
        self.ports = {1:1}
        self.skt = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
        self.skt.bind((skt_name, ETH_P_ALL))
        self.skt.setblocking(False)
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
            timestamp = time.clock_gettime_ns(time.CLOCK_REALTIME)

        return timestamp

    async def recv(self):
        loop = asyncio.get_event_loop()
        msg = await loop.sock_recv(self.skt, MAX_MSG_SIZE)
        timestamp = time.clock_gettime_ns(time.CLOCK_REALTIME)
        cpu_hdr = CPU_Header(msg)
        port_list = [port for (port, d_p) in self.ports.items() if d_p == cpu_hdr.device_port]
        if len(port_list) == 1:
            return (port_list[0], timestamp, msg[CPU_HDR_SIZE:])
        else:
            return await self.recv()
