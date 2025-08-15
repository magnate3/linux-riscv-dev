#!/usr/bin/python3

import struct, sys
from scapy.all import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.protocol import TMultiplexedProtocol

sys.path.append('./gen-py')
from ts_pd_rpc import *
from ts_pd_rpc.ttypes import *

INTERFACE = 'ens1'
XPR_ETHERTYPE = 0x88b5

def thrift_connect():
    transport = TSocket.TSocket('localhost', 9090)
    transport = TTransport.TBufferedTransport(transport)
    transport.open()

    bProtocol = TBinaryProtocol.TBinaryProtocol(transport)
    protocol = TMultiplexedProtocol.TMultiplexedProtocol(bProtocol, 'ts')
    return ts.Client(protocol)

def parse_ethernet(pkt_bytes):
    pkt = {}
    pkt['dst_addr'] = struct.unpack("!Q", b'\x00\x00' + pkt_bytes[0:6])[0]
    pkt['src_addr'] = struct.unpack("!Q", b'\x00\x00' + pkt_bytes[6:12])[0]
    pkt['Ethertype'] = struct.unpack("!H", pkt_bytes[12:14])[0]
    if pkt['Ethertype'] == XPR_ETHERTYPE: parse_pkt(pkt, pkt_bytes)

def parse_pkt(pkt, pkt_bytes):
    pkt['state'] = pkt_bytes[14] >> 6
    pkt['port'] = struct.unpack("!H", pkt_bytes[14:16])[0] & 0x01ff
    pkt['local_egress_ts'] = struct.unpack("!Q", b'\x00\x00' + pkt_bytes[16:22])[0]
    pkt['remote_ingress_ts'] = struct.unpack("!Q", pkt_bytes[24:32])[0]
    pkt['remote_egress_ts'] = struct.unpack("!Q", b'\x00\x00' + pkt_bytes[32:38])[0]
    pkt['local_ingress_ts'] = struct.unpack("!Q", pkt_bytes[40:48])[0]

    if pkt['state'] == 2: calc_offset(pkt)

def get_ts7(port):
    ts7 = 0
    for i in range(5):
        o = c.ts_1588_timestamp_tx_get(0,port)
        if o.ts_valid: ts7 = o.ts
        else: break
    return ts7

def calc_offset(pkt):
    LOCAL_PORT=0x3c
    REMOTE_PORT=0x1c
    local_ts1 = pkt['local_ingress_ts']
    remote_ts1 = pkt['remote_ingress_ts']
    local_ts6 = pkt['local_egress_ts']
    remote_ts6 = pkt['remote_egress_ts']
    #local_ts7 = c.ts_1588_timestamp_tx_get(0,LOCAL_PORT).ts
    #remote_ts7 = c.ts_1588_timestamp_tx_get(0,REMOTE_PORT).ts
    local_ts7 = get_ts7(LOCAL_PORT)
    remote_ts7 = get_ts7(REMOTE_PORT)
    local_ets_delta = local_ts7 - local_ts6
    remote_ets_delta = remote_ts7 - remote_ts6

    ts6_d1 = remote_ts1 - local_ts6
    ts6_d2 = local_ts1 - remote_ts6
    ts7_d1 = remote_ts1 - local_ts7
    ts7_d2 = local_ts1 - remote_ts7

    ts6_owd = (local_ts1 - local_ts6 - remote_ts6 + remote_ts1)/2
    ts7_owd = (local_ts1 - local_ts7 - remote_ts7 + remote_ts1)/2
    ts6_offset = (local_ts6 - remote_ts1 - remote_ts6 + local_ts1)/2
    ts7_offset = (local_ts7 - remote_ts1 - remote_ts7 + local_ts1)/2

    print("%d, %d, %d, %d, %d, %d, " % (local_ts6, local_ts7, remote_ts1, remote_ts6, remote_ts7, local_ts1), end='')
    print("%0.1f, %0.1f, %0.1f, %0.1f, " % (ts6_offset, ts6_owd, ts6_d1, ts6_d2), end='')
    print("%0.1f, %0.1f, %0.1f, %0.1f, " % (ts7_offset, ts7_owd, ts7_d1, ts7_d2), end='')
    print("%d, %d" % (local_ets_delta, remote_ets_delta))

# def calc_ts6_offset(pkt):
#     time_in_switch = pkt['remote_egress_ts'] - pkt['remote_ingress_ts']
#     round_trip_time = pkt['local_ingress_ts'] - pkt['local_egress_ts']
#     time_on_wire = round_trip_time - time_in_switch
#     one_way_delay = time_on_wire / 2
#     offset = pkt['local_egress_ts'] + one_way_delay - pkt['remote_ingress_ts']
#
#     print("Round Trip Time:         %d" % (round_trip_time))
#     print("Time in Remote Switch:   %d" % (time_in_switch))
#     print("Time on Wire:            %d" % (time_on_wire))
#     print("Estimated One Way Delay: %d" % (one_way_delay))
#     print("Calculated Offset:       %d" % (offset))
#     print("Local Egress TS:         %d" % (pkt['local_egress_ts']))
#     print("Remote Ingress TS:       %d" % (pkt['remote_ingress_ts']))
#     print("Remote Egress TS:        %d" % (pkt['remote_egress_ts']))
#     print("Local Ingress TS:        %d" % (pkt['local_ingress_ts']))
#     print("DEBUG D1:                %d" % (pkt['remote_ingress_ts'] - pkt['local_egress_ts']))
#     print("DEBUG D2:                %d" % (pkt['local_ingress_ts'] - pkt['remote_egress_ts']))
#     print('')
#     calc_ts7_offset(pkt)

# def calc_ts7_offset(pkt):
#     LOCAL_PORT=0x3c
#     REMOTE_PORT=0x1c
#     local_ts7 = c.ts_1588_timestamp_tx_get(0,LOCAL_PORT).ts
#     remote_ts7 = c.ts_1588_timestamp_tx_get(0,REMOTE_PORT).ts
#     print("Local Egress TS Delta:  %d" % (local_ts7 - pkt['local_egress_ts']))
#     print("Remote Egress TS Delta: %d" % (remote_ts7 - pkt['remote_egress_ts']))
#
#     time_in_switch = remote_ts7 - pkt['remote_ingress_ts']
#     round_trip_time = pkt['local_ingress_ts'] - local_ts7
#     time_on_wire = round_trip_time - time_in_switch
#     one_way_delay = time_on_wire / 2
#     offset = local_ts7 + one_way_delay - pkt['remote_ingress_ts']
#     print("Round Trip Time:         %d" % (round_trip_time))
#     print("Time in Remote Switch:   %d" % (time_in_switch))
#     print("Time on Wire:            %d" % (time_on_wire))
#     print("Estimated One Way Delay: %d" % (one_way_delay))
#     print("Calculated Offset:       %d" % (offset))
#     print("Local Egress TS:         %d" % (local_ts7))
#     print("Remote Ingress TS:       %d" % (pkt['remote_ingress_ts']))
#     print("Remote Egress TS:        %d" % (remote_ts7))
#     print("Local Ingress TS:        %d" % (pkt['local_ingress_ts']))
#     print("DEBUG D1:                %d" % (pkt['remote_ingress_ts'] - local_ts7))
#     print("DEBUG D2:                %d" % (pkt['local_ingress_ts'] - remote_ts7))
#     print('')

c = thrift_connect()

sniff(iface=INTERFACE, count=1100, prn=lambda x: parse_ethernet(x.build()))
