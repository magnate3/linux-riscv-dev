################################################################################
# BAREFOOT NETWORKS CONFIDENTIAL & PROPRIETARY
#
# Copyright (c) 2019-present Barefoot Networks, Inc.
#
# All Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of
# Barefoot Networks, Inc. and its suppliers, if any. The intellectual and
# technical concepts contained herein are proprietary to Barefoot Networks, Inc.
# and its suppliers and may be covered by U.S. and Foreign Patents, patents in
# process, and are protected by trade secret or copyright law.  Dissemination of
# this information or reproduction of this material is strictly forbidden unless
# prior written permission is obtained from Barefoot Networks, Inc.
#
# No warranty, explicit or implicit is provided, unless granted under a written
# agreement with Barefoot Networks, Inc.
#
################################################################################

import logging

import ptf
from ptf import config
from ptf.thriftutils import *
import ptf.testutils as testutils
from bfruntime_client_base_tests import BfRuntimeTest
import bfrt_grpc.client as gc
import grpc
import subprocess
import time

g_is_tofino = testutils.test_param_get("arch") == "tofino"
g_is_tofino2 = testutils.test_param_get("arch") == "tofino2"
assert g_is_tofino or g_is_tofino2

g_num_pipes = int(testutils.test_param_get("num_pipes"))
g_timer_app_id = 1
g_port_down_app_id = 2

logger = logging.getLogger('Test')
if not len(logger.handlers):
    logger.addHandler(logging.StreamHandler())


def port_to_pipe(port):
    local_port = port & 0x7F
    pipe = port >> 7
    return pipe


def make_port(pipe, local_port):
    return (pipe << 7) | local_port


swports = []
for device, port, ifname in config["interfaces"]:
    pipe = port_to_pipe(port)
    if pipe < g_num_pipes:
        swports.append(port)
        swports.sort()



def CfgPortDownTable(self, target, i_port, pipe_id, port_num, packet_id, o_port):
    logger.info("IgPort %d Pipe %d App %d PortNum %d PktId %d Fwds-to %d", i_port, pipe_id, g_port_down_app_id,
                port_num, packet_id, o_port)
    self.i_p_table.entry_add(
        target,
        [self.i_p_table.make_key([gc.KeyTuple('ig_intr_md.ingress_port', i_port),
                                  gc.KeyTuple(
                                      'hdr.port_down.pipe_id', pipe_id),
                                  gc.KeyTuple('hdr.port_down.app_id',
                                              g_port_down_app_id),
                                  gc.KeyTuple(
                                      'hdr.port_down.port_num', port_num),
                                  gc.KeyTuple('hdr.port_down.packet_id', packet_id)])],
        [self.i_p_table.make_data([gc.DataTuple('port', o_port)],
                                  'SwitchIngress.match')]
    )


def CleanupPortDownTable(self, target=None):
    if not target:
        target = gc.Target(device_id=0, pipe_id=0xFFFF)
    resp = self.i_p_table.entry_get(target, [], {"from_hw": False})
    for _, key in resp:
        if key:
            self.i_p_table.entry_del(target, [key])


def ValueCheck(self, field, data_dict, expect_value):
    value = data_dict[field]
    if (value != expect_value):
        logger.info("Error: data %d, expect %d", value, expect_value)
        # assert(0)


def pgen_timer_hdr_to_dmac(pipe_id, app_id, batch_id, packet_id):
    """
    Given the fields of a 6-byte packet-gen header return an Ethernet MAC address
    which encodes the same values.
    """
    if g_is_tofino:
        pipe_shift = 3
    else:
        pipe_shift = 4
    return '%02x:00:%02x:%02x:%02x:%02x' % ((pipe_id << pipe_shift) | app_id,
                                            batch_id >> 8,
                                            batch_id & 0xFF,
                                            packet_id >> 8,
                                            packet_id & 0xFF)


def pgen_port_down_hdr_to_dmac(pipe_id, app_id, down_port, packet_id):
    """
    Given the fields of a 6-byte packet-gen header return an Ethernet MAC address
    which encodes the same values.
    """
    if g_is_tofino:
        pipe_shift = 3
    else:
        pipe_shift = 4
    return '%02x:00:%02x:%02x:%02x:%02x' % ((pipe_id << pipe_shift) | app_id,
                                            down_port >> 8,
                                            down_port & 0xFF,
                                            packet_id >> 8,
                                            packet_id & 0xFF)


def pgen_port(pipe_id):
    """
    Given a pipe return a port in that pipe which is usable for packet
    generation.  Note that Tofino allows ports 68-71 in each pipe to be used for
    packet generation while Tofino2 allows ports 0-7.  This example will use
    either port 68 or port 6 in a pipe depending on chip type.
    """
    if g_is_tofino:
        pipe_local_port = 68
    if g_is_tofino2:
        pipe_local_port = 6
    return make_port(pipe_id, pipe_local_port)


class PortDownPktgenTest(BfRuntimeTest):
    def setUp(self):
        client_id = 0
        p4_name = "t2na_pktgen_portdown_test"
        BfRuntimeTest.setUp(self, client_id, p4_name)

    def runTest(self):
        logger.info(
            "=============== Testing Packet Generator trigger by port down ===============")
        bfrt_info = self.interface.bfrt_info_get("t2na_pktgen_portdown_test")

        pktgen_app_cfg_table = bfrt_info.table_get("app_cfg")
        pktgen_pkt_buffer_table = bfrt_info.table_get("pkt_buffer")
        pktgen_port_cfg_table = bfrt_info.table_get("port_cfg")
        self.i_p_table = bfrt_info.table_get("SwitchIngress.p")

        target = gc.Target(device_id=0, pipe_id=0xffff)
        pktlen = 100
        p_count = 1  # packets per batch
        b_count = 1
        buff_offset = 0  # generated packets' payload starts from the offset in buffer

        outport = None
        portdown_port = None
        if g_is_tofino:
            outport = 0
            portdown_port = 1
        else:
            outport = 4  # cpu port
            portdown_port = 144  # eth port
        pipe_id = 1
        src_port = make_port(pipe_id, 6)

        ports_to_flap = [portdown_port]
        # Build the expected packets
        p = testutils.simple_ip_packet(
            pktlen=pktlen, eth_dst="99:99:99:99:99:99")
        pkt_lst = []
        pkt_len_list = list()

        try:
            # Add entries to the verify table.
            for port in ports_to_flap:
                for pkt_num in range(p_count):
                    CfgPortDownTable(self, target, src_port,
                                     pipe_id, port, pkt_num, outport)

            # Enable packet generation on the port
            logger.info("enable pktgen port")
            pktgen_port_cfg_table.entry_add(
                target,
                [pktgen_port_cfg_table.make_key(
                    [gc.KeyTuple('dev_port', src_port)])],
                [pktgen_port_cfg_table.make_data([gc.DataTuple('pktgen_enable', bool_val=True)])])

            logger.info("configure pktgen application")
            port_mask_sel = 0
            if g_is_tofino2:
                data = pktgen_app_cfg_table.make_data([gc.DataTuple('port_mask_sel', port_mask_sel),
                                                       gc.DataTuple(
                                                           'app_enable', bool_val=False),
                                                       gc.DataTuple(
                                                           'pkt_len', (pktlen - 0)),
                                                       gc.DataTuple(
                                                           'pkt_buffer_offset', buff_offset),
                                                       gc.DataTuple(
                                                           'pipe_local_source_port', src_port),
                                                       gc.DataTuple(
                                                           'batch_count_cfg', b_count - 1),
                                                       gc.DataTuple(
                                                           'packets_per_batch_cfg', p_count - 1),
                                                       gc.DataTuple('assigned_chnl_id', 6)],
                                                      'trigger_port_down')

            logger.info("configure packet buffer")
            pktgen_pkt_buffer_table.entry_add(
                target,
                [pktgen_pkt_buffer_table.make_key([gc.KeyTuple('pkt_buffer_offset', buff_offset),
                                                   gc.KeyTuple('pkt_buffer_size', (pktlen))])],
                [pktgen_pkt_buffer_table.make_data([gc.DataTuple('buffer', bytearray(bytes(p)[:]))])])

            port_mask = 0
            '''
            if g_is_tofino2:
                pktgen_port_mask_table = bfrt_info.table_get(
                    "tf2.pktgen.port_mask")
                for port in ports_to_flap:
                    port_mask |= 1 << (port & 0x3f)

                print(port & 0x3f)
                print(port_mask)
                # set the port mask
                logger.info("Set port down mask")
                pktgen_port_mask_table.entry_add(
                    target,
                    [pktgen_port_mask_table.make_key(
                        [gc.KeyTuple('port_mask_sel', port_mask_sel)])],
                    [pktgen_port_mask_table.make_data([gc.DataTuple('mask', port_mask)])])
                resp = pktgen_port_mask_table.entry_get(
                    target,
                    [pktgen_port_mask_table.make_key(
                        [gc.KeyTuple('port_mask_sel', port_mask_sel)])],
                    {"from_hw": False})
                data_dict = next(resp)[0].to_dict()
                if data_dict['mask'] != port_mask:
                    logger.error("Port Down Mask not set correctly")
            '''
            logger.info("enable pktgen")
            pktgen_app_cfg_table.entry_mod(
                target,
                [pktgen_app_cfg_table.make_key(
                    [gc.KeyTuple('app_id', g_port_down_app_id)])],
                [data]
            )

            pktgen_port_cfg_table.entry_mod(
                target,
                [pktgen_port_cfg_table.make_key(
                    [gc.KeyTuple('dev_port', portdown_port)])],
                [pktgen_port_cfg_table.make_data([gc.DataTuple('clear_port_down_enable', bool_val=True)])])

            print("You need to port down!")
            print("If you want to close, then hit enter!")
            enter = input()

            # disable tables
            logger.info("disable pktgen")
            pktgen_app_cfg_table.entry_mod(
                target,
                [pktgen_app_cfg_table.make_key(
                    [gc.KeyTuple('app_id', g_port_down_app_id)])],
                [pktgen_app_cfg_table.make_data([gc.DataTuple('app_enable', bool_val=False)],
                                                'trigger_port_down')]
            )

            logger.info("disable port_cfg")
            pktgen_port_cfg_table.entry_mod(
                target,
                [pktgen_port_cfg_table.make_key(
                    [gc.KeyTuple('dev_port', src_port)])],
                [pktgen_port_cfg_table.make_data([gc.DataTuple('pktgen_enable', bool_val=False)])])

            CleanupPortDownTable(self, target)
        finally:
            pass
