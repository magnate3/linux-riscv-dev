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

from ptf import config
from ptf.thriftutils import *
import ptf.testutils as testutils
from bfruntime_client_base_tests import BfRuntimeTest
import bfrt_grpc.client as gc
import time
from timeit import default_timer as timer

g_is_tofino = testutils.test_param_get("arch") == "tofino"
g_is_tofino2 = testutils.test_param_get("arch") == "tofino2"
assert g_is_tofino or g_is_tofino2

g_num_pipes = int(testutils.test_param_get("num_pipes"))
g_timer_app_id = 1
g_port_down_app_id = 2
g_recirc_pattern_app_id = 3

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

swports_0 = []
swports_1 = []
swports_2 = []
swports_3 = []
# the following method categorizes the ports in ports.json file as belonging to either of the pipes (0, 1, 2, 3)
for port in swports:
    pipe = port_to_pipe(port)
    if pipe == 0:
        swports_0.append(port)
    elif pipe == 1:
        swports_1.append(port)
    elif pipe == 2:
        swports_2.append(port)
    elif pipe == 3:
        swports_3.append(port)


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


p4_name = "tna_pktgen_recirc_test"


class TimerPktgenTest(BfRuntimeTest):
    def setUp(self):
        client_id = 0
        BfRuntimeTest.setUp(self, client_id, p4_name)

    def runTest(self):
        logger.info(
            "=============== Testing Packet Generator trigger by Timer ===============")
        bfrt_info = self.interface.bfrt_info_get(p4_name)

        pktgen_app_cfg_table = bfrt_info.table_get("app_cfg")
        pktgen_pkt_buffer_table = bfrt_info.table_get("pkt_buffer")
        pktgen_port_cfg_table = bfrt_info.table_get("port_cfg")
        forward = bfrt_info.table_get("SwitchIngress.forward")
        recirculate = bfrt_info.table_get("SwitchIngress.check_recirculate")

        target = gc.Target(device_id=0, pipe_id=0xffff)
        app_id = g_recirc_pattern_app_id

        pktlen = 100

        p_count = 1  # packets per batch
        b_count = 1  # batch number
        buff_offset = 0

        pattern_value = 0x1111
        pattern_mask = 0xffff
        ing_port = 136
        egr_port = 144
        pgen_port = 6
        cpu_port = 5
        pipe_id = 1
        pipe_pgen_port = make_port(pipe_id, pgen_port)

        # generated packet
        gp = testutils.simple_tcp_packet(
            pktlen=pktlen, eth_dst="CC:AA:DD:CC:AA:DD")

        try:
            # ingress port -> recirculate port -> egress port
            forward_key = [forward.make_key(
                [gc.KeyTuple('ig_intr_md.ingress_port', ing_port)])]
            forward_data = [forward.make_data(
                [gc.DataTuple('port', pipe_pgen_port)], 'SwitchIngress.set_port')]
            forward.entry_add(target, forward_key, forward_data)

            forward_key = [forward.make_key(
                [gc.KeyTuple('ig_intr_md.ingress_port', pipe_pgen_port)])]
            forward_data = [forward.make_data(
                [gc.DataTuple('port', egr_port)], 'SwitchIngress.set_port')]
            forward.entry_add(target, forward_key, forward_data)

            # recirculate port -> if packet is generated by pktgen, then send it to cpu port.
            recirculate_key = [recirculate.make_key(
                [gc.KeyTuple('hdr.recirc_pattern.app_id', app_id)])]
            recirculate_data = [recirculate.make_data(
                [gc.DataTuple('port', cpu_port)], 'SwitchIngress.match_r')]
            recirculate.entry_add(target, recirculate_key, recirculate_data)

            # Enable packet generation on the port
            logger.info("enable pktgen port")
            pktgen_port_cfg_table.entry_add(
                target,
                [pktgen_port_cfg_table.make_key(
                    [gc.KeyTuple('dev_port', pipe_pgen_port)])],
                [pktgen_port_cfg_table.make_data([gc.DataTuple('pktgen_enable', bool_val=True)])])
            pktgen_port_cfg_table.entry_mod(
                target,
                [pktgen_port_cfg_table.make_key(
                    [gc.KeyTuple('dev_port', pipe_pgen_port)])],
                [pktgen_port_cfg_table.make_data([gc.DataTuple('pattern_matching_enable', bool_val=True)])])

            # Configure the packet generation timer application
            logger.info("configure pktgen application")
            if g_is_tofino2:
                data = pktgen_app_cfg_table.make_data([gc.DataTuple('app_enable', bool_val=True),
                                                       gc.DataTuple(
                                                           'pkt_len', pktlen),
                                                       gc.DataTuple(
                                                           'pkt_buffer_offset', buff_offset),
                                                       gc.DataTuple(
                                                           'pipe_local_source_port', pipe_pgen_port),
                                                       gc.DataTuple(
                                                           'increment_source_port', bool_val=False),
                                                       gc.DataTuple(
                                                           'batch_count_cfg', b_count - 1),
                                                       gc.DataTuple(
                                                           'packets_per_batch_cfg', p_count - 1),
                                                       gc.DataTuple(
                                                           'pattern_value', pattern_value),
                                                       gc.DataTuple(
                                                           'pattern_mask', pattern_mask),
                                                       gc.DataTuple('assigned_chnl_id', pgen_port)],
                                                      "trigger_recirc_pattern")

            logger.info("configure packet buffer")
            pktgen_pkt_buffer_table.entry_add(
                target,
                [pktgen_pkt_buffer_table.make_key([gc.KeyTuple('pkt_buffer_offset', buff_offset),
                                                   gc.KeyTuple('pkt_buffer_size', pktlen)])],
                [pktgen_pkt_buffer_table.make_data([gc.DataTuple('buffer', bytearray(bytes(gp)[:]))])])

            pktgen_app_cfg_table.entry_mod(
                target,
                [pktgen_app_cfg_table.make_key(
                    [gc.KeyTuple('app_id', app_id)])],
                [data])

            print("Hit enter to continue..", input())

            # Disable the application.
            logger.info("disable pktgen")
            pktgen_app_cfg_table.entry_mod(
                target,
                [pktgen_app_cfg_table.make_key(
                    [gc.KeyTuple('app_id', app_id)])],
                [pktgen_app_cfg_table.make_data([gc.DataTuple('app_enable', bool_val=False)],
                                                'trigger_recirc_pattern')])

            pktgen_port_cfg_table.entry_mod(
                target,
                [pktgen_port_cfg_table.make_key(
                    [gc.KeyTuple('dev_port', pgen_port)])],
                [pktgen_port_cfg_table.make_data([gc.DataTuple('pktgen_enable', bool_val=False)])])

        except gc.BfruntimeRpcException as e:
            raise e
        finally:
            pass
