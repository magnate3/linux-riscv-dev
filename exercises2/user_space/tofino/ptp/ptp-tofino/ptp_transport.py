#!/usr/bin/env python3

# pylint: disable=invalid-name
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

# TODO: Implement IPv4/6 UDP transports

import struct
from enum import IntEnum
# import tofino
# import system

ETH_P_1588 = 0x88F7
ETH_P_IP = 0x0800
ETH_P_IPV6 = 0x86DD

IANA_PORT_PTP_EVENT = 319
IANA_PORT_PTP_GENERAL = 320

IPV4_PTP_PRIMARY = "224.0.1.129"
IPV4_PTP_PDELAY = "224.0.0.107"

IPV6_PTP_PRIMARY = "FF0X:0:0:0:0:0:0:181" # TODO: configure IPv6 multicast scope (X)
IPV6_PTP_PDELAY = "FF02:0:0:0:0:0:0:6B"

ETH_DST_PTP_PRIMARY = 0x011b19000000.to_bytes(6, 'big')
ETH_DST_PTP_PDELAY = 0x0180c200000e.to_bytes(6, 'big')

class PTP_PROTO(IntEnum):
    UDP_IPV4 = 1
    UDP_IPV6 = 2
    ETHERNET = 3

class Port_Config:
    def __init__(self):
        # TODO: Retrieve correct values from (?)
        self.proto = PTP_PROTO.ETHERNET
        self.src_mac = 0x000000000000.to_bytes(6, 'big')
        self.src_ipv4 = b'\x00' * 4
        self.src_ipv6 = b'\x00' * 16
        self.src_port = 0

class Ethernet:
    parser = struct.Struct('!6s6sH')

    def __init__(self, buffer=b''):
        self.src = None
        self.dst = None
        self.type = None
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        t = self.parser.unpack(buffer[:self.parser.size])
        self.dst = t[0]
        self.src = t[1]
        self.type = t[2]

    def bytes(self):
        t = (self.dst, self.src, self.type)
        return self.parser.pack(*t)

class UDP:
    parser = struct.Struct('!4H')

    def __init__(self):
        self.src = None
        self.dst = None
        self.len = None
        self.chk = 0

    def parse(self, buffer):
        t = self.parser.unpack(buffer)
        self.src = t[0]
        self.dst = t[1]
        self.len = t[2]
        self.chk = t[3]

    def bytes(self):
        t = (self.src, self.dst, self.len, self.chk)
        return self.parser.pack(*t)

class IPv4:
    parser = struct.Struct('!2B3H2BH4s4s')

    def __init__(self):
        self.version = 4
        self.ihl = 5
        self.tos = 0
        self.len = None
        self.id = 0
        self.flags = 0
        self.fragment_offset = 0
        self.ttl = None
        self.proto = None
        self.checksum = None
        self.src = None
        self.dst = None

    def parse(self, buffer):
        pass

    def bytes(self):
        pass

class IPv6:
    parser = struct.Struct('!LHBB16s16s')

    def __init__(self):
        self.version = 6
        self.traffic_class = 0
        self.flow_label = 0
        self.payload_len = None
        self.next_header = None
        self.hop_limit = None
        self.src = None
        self.dst = None

    def parse(self, buffer):
        t = self.parser.unpack(buffer)
        self.version = (t[0] >> 28) & 0x0F
        self.traffic_class = (t[0] >> 20) & 0xFF
        self.flow_label = t[0] & 0x000FFFFF
        self.payload_len = t[1]
        self.next_header = t[2]
        self.hop_limit = t[3]
        self.src = t[4]
        self.dst = t[5]

    def bytes(self):
        t = (
            (self.version << 28) & (self.traffic_class << 20) & self.flow_label,
            self.payload_len,
            self.next_header,
            self.hop_limit,
            self.src,
            self.dst
        )
        return self.parser.pack(*t)

class Transport:
    def __init__(self, skt_name, driver_name, port_list):
        driver = self.load_driver(driver_name)
        self.skt = driver.Socket(skt_name, port_list)
        self.port_config = {}
        self.number_of_ports = self.skt.number_of_ports
        for i in range(1, self.number_of_ports + 1):
            self.port_config[i] = Port_Config()

    def load_driver(self, driver_name):
        if driver_name == 'tofino':
            import tofino as driver
        elif driver_name == 'dummy':
            import dummy as driver
        else:
            print("[ERROR] Unable to locate driver: %s" % (driver_name))
        return driver

    def send_message(self, msg, port_number, get_timestamp=False):
        timestamp = None

        hdr = Ethernet()
        hdr.src = self.port_config[port_number].src_mac

        if self.port_config[port_number].proto == PTP_PROTO.ETHERNET:
            msg.transportSpecific = 0
            hdr.dst = ETH_DST_PTP_PRIMARY # TODO: select destination based on message type
            hdr.type = ETH_P_1588
        else:
            print("[ERROR] Protocol not Implemented")

        if hdr:
            # msg_length = hdr.parser.size + msg.parser.size
            # pad = b'\x00' * (128 - msg_length) if msg_length < 128 else b''
            timestamp = self.skt.send(hdr.bytes() + msg.bytes(), port_number, get_timestamp)

        return timestamp

    def send_buffer(self, buffer, port_number, get_timestamp=False):
        # TODO: merge this with send_message
        # TODO: update UDP checksum if needed
        return self.skt.send(buffer, port_number, get_timestamp)

    async def recv_message(self):
        while True:
            port_number, timestamp, buffer = await self.skt.recv()
            ethernet = Ethernet(buffer)
            if ethernet.type == ETH_P_1588:
                msg_offset = Ethernet.parser.size
            else:
                continue
            return (buffer, msg_offset, port_number, timestamp)
