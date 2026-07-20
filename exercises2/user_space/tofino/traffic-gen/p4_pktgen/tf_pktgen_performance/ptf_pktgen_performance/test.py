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
from timeit import default_timer as timer

g_is_tofino = testutils.test_param_get("arch") == "tofino"
g_is_tofino2 = testutils.test_param_get("arch") == "tofino2"
assert g_is_tofino or g_is_tofino2

g_num_pipes = int(testutils.test_param_get("num_pipes"))
g_timer_app_id = 1

logger = logging.getLogger('Test')
if not len(logger.handlers):
    logger.addHandler(logging.StreamHandler())


def port_to_pipe(port):
    local_port = port & 0x7F
    pipe = port >> 7
    return pipe


def make_port(pipe, local_port):
    return (pipe << 7) | local_port

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



def make_packet(pkt_len):
        # WARN: If you do not senInvaild hdr.timer at pipeline, then do not use multicast MAC address in eth_dst, 
        # It will cause packet drop.
        p = testutils.simple_eth_packet(
                pktlen=pkt_len, eth_dst="aa:aa:00:99:99:99", eth_src="ca:fe:99:ca:fe:88")
        print(p.show())
        return p

class TimerPktgenTest(BfRuntimeTest):
    def setUp(self):
        client_id = 0
        self.p4_name = "tna_pktgen_performance"
        BfRuntimeTest.setUp(self, client_id, self.p4_name)

    def runTest(self):
        logger.info(
            "=============== Testing Packet Generator trigger by Timer ===============")
        bfrt_info = self.interface.bfrt_info_get(self.p4_name)

        pktgen_app_cfg_table = bfrt_info.table_get("app_cfg")
        pktgen_pkt_buffer_table = bfrt_info.table_get("pkt_buffer")
        pktgen_port_cfg_table = bfrt_info.table_get("port_cfg")

        self.forward = bfrt_info.table_get("SwitchIngress.t")

        # timer pktgen app_id = 1 one shot 0
        target = gc.Target(device_id=0, pipe_id=0xffff)
        app_id = g_timer_app_id
        # pipe 0 : 1, 6
        # pipe 1, 2, 3 : 0, 2, 4, 6
        # pipes = [0, 1, 2, 3]
        # ports = [0, 1, 2, 3, 4, 5, 6, 7]
        
        pktlens = [64, 128, 256, 512, 1024, 1518]

        pgen_pipe_id = 0
        chan_port = 6
        pktlens_id = 1
        src_port = make_port(pgen_pipe_id, chan_port)
        p_count = 1  # packets per batch
        b_count = 1  # batch number
        buff_offset = 0  # generated packets' payload will be taken from the offset in buffer
        time_ns = 0      
        out_port = None
        if g_is_tofino:
            out_port = 0
        if g_is_tofino2:
            out_port =	 144
        print("src_port", src_port)
        try:
            self.forward.entry_add(
                target,
                [self.forward.make_key([gc.KeyTuple('ig_intr_md.ingress_port', src_port)])],
                [self.forward.make_data([gc.DataTuple('port', out_port)],
                                          'SwitchIngress.set_port')]
            )
            p = make_packet(pktlens[pktlens_id])
            pktlen = pktlens[pktlens_id]
            # Enable packet generation on the port
            logger.info("enable pktgen port")
            pktgen_port_cfg_table.entry_add(
                target,
                [pktgen_port_cfg_table.make_key(
                    [gc.KeyTuple('dev_port', src_port)])],
                [pktgen_port_cfg_table.make_data([gc.DataTuple('pktgen_enable', bool_val=True)])])


            # Configure the packet generation timer application
            logger.info("configure pktgen application")
            if g_is_tofino:
                data = pktgen_app_cfg_table.make_data([gc.DataTuple('timer_nanosec', time_ns),
                                                       gc.DataTuple(
                                                           'app_enable', bool_val=False),
                                                       gc.DataTuple(
                                                           'pkt_len', (pktlen - 0)),
                                                       gc.DataTuple(
                                                           'pkt_buffer_offset', buff_offset),
                                                       gc.DataTuple(
                                                           'pipe_local_source_port', src_port),
                                                       gc.DataTuple(
                                                           'increment_source_port', bool_val=False),
                                                       gc.DataTuple(
                                                           'batch_count_cfg', b_count - 1),
                                                       gc.DataTuple(
                                                           'packets_per_batch_cfg', p_count - 1),
                                                       gc.DataTuple('ibg', 1),
                                                       gc.DataTuple(
                                                           'ibg_jitter', 0),
                                                       gc.DataTuple(
                                                           'ipg', 1000),
                                                       gc.DataTuple(
                                                           'ipg_jitter', 500),
                                                       gc.DataTuple(
                                                           'batch_counter', 0),
                                                       gc.DataTuple(
                                                           'pkt_counter', 0),
                                                       gc.DataTuple('trigger_counter', 0)],
                                                      'trigger_timer_periodic')
            if g_is_tofino2:
                data = pktgen_app_cfg_table.make_data([gc.DataTuple('timer_nanosec', time_ns),
                                                       gc.DataTuple(
                                                           'app_enable', bool_val=True),
                                                       gc.DataTuple(
                                                           'pkt_len', (pktlen - 0)),
                                                       gc.DataTuple(
                                                           'pkt_buffer_offset', buff_offset),
                                                       gc.DataTuple(
                                                           'pipe_local_source_port', src_port),
                                                       gc.DataTuple(
                                                           'increment_source_port', bool_val=False),
                                                       gc.DataTuple(
                                                           'batch_count_cfg', b_count - 1),
                                                       gc.DataTuple(
                                                           'packets_per_batch_cfg', p_count - 1),
                                                       gc.DataTuple('ibg', 0),
                                                       gc.DataTuple(
                                                           'ibg_jitter', 0),
                                                       gc.DataTuple(
                                                           'ipg', 0),
                                                       gc.DataTuple(
                                                           'ipg_jitter', 0),
                                                       gc.DataTuple(
                                                           'batch_counter', 0),
                                                       gc.DataTuple(
                                                           'pkt_counter', 0),
                                                       gc.DataTuple(
                                                           'trigger_counter', 0),
                                                       gc.DataTuple('assigned_chnl_id', chan_port)],
                                                      'trigger_timer_periodic')
            logger.info("configure packet buffer")
            pktgen_pkt_buffer_table.entry_mod(
                target,
                [pktgen_pkt_buffer_table.make_key([gc.KeyTuple('pkt_buffer_offset', buff_offset),
                                                   gc.KeyTuple('pkt_buffer_size', (pktlen))])],
                [pktgen_pkt_buffer_table.make_data([gc.DataTuple('buffer', bytearray(bytes(p)[:]))])])
            
            logger.info("enable pktgen")
            # pktgen_app_cfg_table does not have add function
            pktgen_app_cfg_table.entry_mod(
                target,
                [pktgen_app_cfg_table.make_key(
                    [gc.KeyTuple('app_id', g_timer_app_id)])],
                [data])


            time.sleep(1)
            # Disable the application.
            logger.info("disable pktgen")
            pktgen_app_cfg_table.entry_mod(
                target,
                [pktgen_app_cfg_table.make_key(
                    [gc.KeyTuple('app_id', g_timer_app_id)])],
                [pktgen_app_cfg_table.make_data([gc.DataTuple('app_enable', bool_val=False)],
                                                'trigger_timer_periodic')])
            # Disable packet generation on the port
            pktgen_port_cfg_table.entry_mod(
                target,
                [pktgen_port_cfg_table.make_key(
                    [gc.KeyTuple('dev_port', src_port)])],
                [pktgen_port_cfg_table.make_data([gc.DataTuple('pktgen_enable', bool_val=False)])])
           
            time.sleep(1)


            resp = self.forward.entry_get(target,
                                        [self.forward.make_key([gc.KeyTuple('ig_intr_md.ingress_port', src_port)])],
                                        {"from_hw": True},
                                        self.forward.make_data(
                                            [gc.DataTuple("$COUNTER_SPEC_BYTES"),
                                             gc.DataTuple("$COUNTER_SPEC_PKTS")],
                                            'SwitchIngress.set_port', get=True)
                                        )
 
            # parse resp to get the counter
            data_dict = next(resp)[0].to_dict()
            recv_pkts = data_dict["$COUNTER_SPEC_PKTS"]
            recv_bytes = data_dict["$COUNTER_SPEC_BYTES"] 
            print(recv_pkts, recv_bytes)


            self.forward.entry_del(
                target,
                [self.forward.make_key([gc.KeyTuple('ig_intr_md.ingress_port', src_port)])]
            )

        except gc.BfruntimeRpcException as e:
            raise e
        finally:
            pass
