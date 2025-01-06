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
import time

g_is_tofino = testutils.test_param_get("arch") == "tofino"
g_is_tofino2 = testutils.test_param_get("arch") == "tofino2"
assert g_is_tofino or g_is_tofino2

g_num_pipes = int(testutils.test_param_get("num_pipes"))
g_trig_dp_app_id = 3
p4_name = "t2na_pktgen_dp_test"


logger = logging.getLogger('Test')
if not len(logger.handlers):
    logger.addHandler(logging.StreamHandler())


def make_port(pipe, local_port):
    return (pipe << 7) | local_port


class DprsrPktgenTest(BfRuntimeTest):
    def setUp(self):
        client_id = 0
        BfRuntimeTest.setUp(self, client_id, p4_name)

    def runTest(self):
        logger.info(
            "=============== Testing Packet Generator by DP ===============")
        trgt = gc.Target(device_id=0, pipe_id=0xffff)
        bfrt_info = self.interface.bfrt_info_get(p4_name)

        # Get pktgen Tables
        pktgen_app_cfg_table = bfrt_info.table_get("app_cfg")
        pktgen_pkt_buffer_table = bfrt_info.table_get("pkt_buffer")
        pktgen_port_cfg_table = bfrt_info.table_get("port_cfg")

        # Get forward Table
        forward = bfrt_info.table_get('SwitchIngress.forward')

        pipe_id = 1
        buff_offset = 0
        pktlen = 64
        pgen_port = 6
        pipe_pgen_port = make_port(pipe_id, 6)
        cpu_port = 5
        ing_port = 136
        egr_port = 144

        print("pgen port", pipe_pgen_port)

        p = testutils.simple_ip_packet(
            pktlen=pktlen, eth_dst="99:99:99:99:99:99", eth_src="11:22:33:44:55:66")

        # Add forward entries
        forward_keys = []
        forward_key = [forward.make_key(
            [gc.KeyTuple('ig_intr_md.ingress_port', ing_port)])]
        forward_data = [forward.make_data(
            [gc.DataTuple('port', egr_port)], 'SwitchIngress.set_port')]
        forward.entry_add(trgt, forward_key, forward_data)
        forward_keys.append(forward_key)

        forward_key = [forward.make_key(
            [gc.KeyTuple('ig_intr_md.ingress_port', egr_port)])]
        forward_data = [forward.make_data(
            [gc.DataTuple('port', ing_port)], 'SwitchIngress.set_port')]
        forward.entry_add(trgt, forward_key, forward_data)
        forward_keys.append(forward_key)

        forward_key = [forward.make_key(
            [gc.KeyTuple('ig_intr_md.ingress_port', pipe_pgen_port)])]
        forward_data = [forward.make_data(
            [gc.DataTuple('port', cpu_port)], 'SwitchIngress.set_port')]
        forward.entry_add(trgt, forward_key, forward_data)
        forward_keys.append(forward_key)

        pktgen_pkt_buffer_table.entry_add(
            trgt,
            [pktgen_pkt_buffer_table.make_key([gc.KeyTuple('pkt_buffer_offset', buff_offset),
                                               gc.KeyTuple('pkt_buffer_size', pktlen)])],
            [pktgen_pkt_buffer_table.make_data([gc.DataTuple('buffer', bytearray(bytes(p)[:]))])])

        # pattern value and pattern mask are not need in dprsr function, but it need to assgin in to app cfg table....
        pattern_value = 0x1234
        pattern_mask = 0x00000000
        # Set pktgen table and enteris
        port = make_port(pipe_id, pgen_port)
        pktgen_port_cfg_table.entry_add(
            trgt,
            [pktgen_port_cfg_table.make_key(
                [gc.KeyTuple('dev_port', pipe_pgen_port)])],
            [pktgen_port_cfg_table.make_data([gc.DataTuple('pktgen_enable', bool_val=True)])])

        pktgen_app_data = pktgen_app_cfg_table.make_data([gc.DataTuple('app_enable', bool_val=True),
                                                          gc.DataTuple(
                                                              'batch_count_cfg', 0),
                                                          gc.DataTuple(
                                                              'packets_per_batch_cfg', 0),
                                                          gc.DataTuple(
                                                              'pkt_len', pktlen),
                                                          gc.DataTuple(
                                                              'pkt_buffer_offset', buff_offset),
                                                          gc.DataTuple(
                                                              'pipe_local_source_port', pipe_pgen_port),
                                                          gc.DataTuple(
                                                              'assigned_chnl_id', pgen_port),
                                                          gc.DataTuple(
                                                              'pattern_value', pattern_value),
                                                          gc.DataTuple(
                                                              'pattern_mask', pattern_mask)],
                                                         'trigger_dprsr')
        pktgen_app_cfg_table.entry_mod(
            trgt,
            [pktgen_app_cfg_table.make_key(
                [gc.KeyTuple('app_id', g_trig_dp_app_id)])],
            [pktgen_app_data])

        print("Waiting Enter", input())

        logger.info("delete forward table")
        for forward_key in forward_keys:
            forward.entry_del(trgt, forward_key)
