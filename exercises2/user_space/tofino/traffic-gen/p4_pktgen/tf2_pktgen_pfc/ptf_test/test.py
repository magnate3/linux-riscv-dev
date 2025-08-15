# Copyright 2013-present Barefoot Networks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Thrift PD interface basic tests
"""

from collections import OrderedDict

import time
import sys
import logging
import copy
import pdb
import unittest
import random

from ptf import config

import pd_base_tests
from ptf.testutils import *
from ptf.thriftutils import *
from ptf_port import *

from bfruntime_client_base_tests import BfRuntimeTest
import bfrt_grpc.bfruntime_pb2 as bfruntime_pb2
import bfrt_grpc.client as gc

# PD-Fixed
from res_pd_rpc.ttypes import *
from pal_rpc.ttypes import *
from conn_mgr_pd_rpc.ttypes import *
from mc_pd_rpc.ttypes import *
from devport_mgr_pd_rpc.ttypes import *
from pkt_pd_rpc.ttypes import *
from tm_api_rpc.ttypes import *
from mirror_pd_rpc.ttypes import *

# P4-PD
#from t2na_pgr.p4_pd_rpc.ttypes import *

logger = logging.getLogger('Test')
logger.addHandler(logging.StreamHandler())

dev_id = 0
client_id = 0
p4_name = "t2na_pktgen_pfc"

swports = []
pgen_ports = []
recirc_ports = []
num_pipes = int(test_param_get('num_pipes'))
for device, port, ifname in config["interfaces"]:
    pipe = port >> 7
    if pipe in range(num_pipes):
        swports.append(port)
swports.sort()

g_target_model = test_param_get('target').lower() == "asic-model"
g_target_hw = test_param_get('target').lower() == "hw"
g_tofino2 = test_param_get("arch") == "tofino2"

if g_tofino2:
    for pipe in range(num_pipes):
        for local_port in range(0, 8):
            pgen_ports.append((pipe << 7) | local_port)
            recirc_ports.append((pipe << 7) | local_port)

fpports = list(swports)
for p in pgen_ports + recirc_ports:
    if p in fpports:
        fpports.remove(p)

all_app_ids = range(16)


def setup_random(seed_val=None):
    if seed_val is None:
        seed_val = int(time.time())
    logger.info("Seed is: %d", seed_val)
    random.seed(seed_val)


def make_port(pipe, local_port):
    assert pipe >= 0 and pipe < 4
    assert local_port >= 0 and local_port < 72
    return pipe << 7 | local_port


def port_to_pipe(port):
    return port >> 7


def port_to_local_port(port):
    return port & 0x7F


def cleanup_one_table(test, tbl, trgt=None):
    if not trgt:
        trgt = gc.Target(device_id=dev_id, pipe_id=0xFFFF)
    #logger.info("Cleanup: Getting entries from " + tbl)
    t = test.bfrt_info.table_get(tbl)
    for data, key in t.entry_get(trgt, None, {"from_hw": False}):
        t.entry_del(trgt, [key])


def cleanup(test, app_id, pgen_port):
    trgt = gc.Target(device_id=dev_id, pipe_id=0xFFFF)
    dt = DevTarget_t(dev_id, hex_to_i16(0xFFFF))

    # Remove all PVS entries.
    cleanup_one_table(test, "iPrsr.pfc_apps", trgt)

    # Disable all packet gen apps.
    shdl = test.conn_mgr.client_init()
    test.conn_mgr.pktgen_app_disable(shdl, dt, app_id)

    # Reset the app counters.
    test.conn_mgr.pktgen_set_trigger_counter(shdl, dt, app_id, 0)
    test.conn_mgr.pktgen_set_batch_counter(shdl, dt, app_id, 0)
    test.conn_mgr.pktgen_set_pkt_counter(shdl, dt, app_id, 0)

    # Disable packet generation.
    try:
        test.conn_mgr.pktgen_disable(shdl, dev_id, pgen_port)
    except InvalidPktGenOperation:
        pass

    test.conn_mgr.complete_operations(shdl)
    test.conn_mgr.client_cleanup(shdl)


def get_cntr(test, trgt, tbl, idx):
    t = test.bfrt_info.table_get(tbl)
    resp = t.entry_get(trgt,
                       [t.make_key([gc.KeyTuple('$COUNTER_INDEX', idx)])],
                       {"from_hw": True},
                       None)

    data_dict = next(resp)[0].to_dict()
    if '$COUNTER_SPEC_PKTS' in data_dict:
        recv_pkts = data_dict['$COUNTER_SPEC_PKTS']
    else:
        recv_pkts = 0
    if '$COUNTER_SPEC_BYTES' in data_dict:
        recv_bytes = data_dict['$COUNTER_SPEC_BYTES']
    else:
        recv_bytes = 0
    p_name = 'pkt '
    if recv_pkts > 1:
        p_name = 'pkts'
    logger.info("%s[%d] = %d %s %d bytes", tbl,
                idx, recv_pkts, p_name, recv_bytes)
    return recv_pkts, recv_bytes


def clr_cntr(test, trgt, tbl, idx):
    t = test.bfrt_info.table_get(tbl)
    fields = t.info.data_field_name_list_get()
    data_tuples = [gc.DataTuple(n, 0) for n in [
        '$COUNTER_SPEC_PKTS', '$COUNTER_SPEC_BYTES'] if n in fields]
    t.entry_add(trgt,
                [t.make_key([gc.KeyTuple('$COUNTER_INDEX', idx)])],
                [t.make_data(data_tuples)])


class TestPFC(BfRuntimeTest, pd_base_tests.ThriftInterfaceDataPlane):
    def __init__(self):
        BfRuntimeTest.__init__(self)
        pd_base_tests.ThriftInterfaceDataPlane.__init__(self, [p4_name])

    def setUp(self):
        pd_base_tests.ThriftInterfaceDataPlane.setUp(self)
        BfRuntimeTest.setUp(self, client_id, p4_name)
        setup_random()
        self.bfrt_info = self.interface.bfrt_info_get(p4_name)

    def runTest(self):
        dt = DevTarget_t(dev_id, hex_to_i16(0xFFFF))
        trgt = gc.Target(device_id=dev_id)
        shdl = None
        pktlen = 64
        batch_count = 1
        batch_size = 1
        app_id = 3
        logger.info("App %d: %d batches of %d packets",
                    app_id, batch_count, batch_size)
        pgen_port = 6
# pipe1 port 6 is 134
        pgen_port_for_assign_appcfg = 134
        ing_port = 136
        egr_port = 144
        cpu_port = 5

        try:
            shdl = self.conn_mgr.client_init()

            # Set up the parser.
            t = self.bfrt_info.table_get('iPrsr.pfc_apps')
            forward = self.bfrt_info.table_get("i.forward")

            forward_key = [forward.make_key(
                [gc.KeyTuple('ig_intr_md.ingress_port', ing_port)])]
            forward_data = [forward.make_data(
                [gc.DataTuple('port', egr_port)], 'i.set_port')]
            forward.entry_add(trgt, forward_key, forward_data)

            forward_key = [forward.make_key(
                [gc.KeyTuple('ig_intr_md.ingress_port', egr_port)])]
            forward_data = [forward.make_data(
                [gc.DataTuple('port', ing_port)], 'i.set_port')]
            forward.entry_add(trgt, forward_key, forward_data)

            forward_key = [forward.make_key(
                [gc.KeyTuple('ig_intr_md.ingress_port', pgen_port_for_assign_appcfg)])]
            forward_data = [forward.make_data(
                [gc.DataTuple('port', cpu_port)], 'i.set_port')]
            forward.entry_add(trgt, forward_key, forward_data)

            t.entry_add(
                trgt, [t.make_key([gc.KeyTuple('app_id', app_id, 0xF)])], None)
            logger.info("Added pvs entry")

            for pipe in range(num_pipes):
                port = make_port(pipe, pgen_port)
                print("pipe : %s, port %s" % (pipe, port))
            for pipe in range(num_pipes):
                port = make_port(pipe, pgen_port)
                print("pipe : %s, port %s" % (pipe, port))
                self.conn_mgr.pktgen_enable(shdl, dev_id, port)

            logger.info("Added ports entry")
            print("port %s, pgen_port %s" % (port, pgen_port))

            # Configure the PFC app and enable it.
            pfc_hdr = [hex_to_byte((i << 4) | i) for i in range(16)]
            pfc_hdr_test = [(i << 4) | i for i in range(16)]
            print(pfc_hdr_test)
            print(pfc_hdr)

            app = PktGenAppCfg_tof2_t(trigger_type=PktGenTriggerType_t.PFC,
                                      # Fix me, length > 56 is required by driver.  Remove that restriction.
                                      length=100,
                                      batch_count=batch_count-1,
                                      pkt_count=batch_size-1,
                                      assigned_chnl_id=pgen_port,
                                      src_port=pgen_port_for_assign_appcfg,
                                      pfc_max_msgs=16,
                                      pfc_timer_en=False,
                                      pfc_timer=1234,
                                      pfc_hdr=pfc_hdr)
            self.conn_mgr.pktgen_cfg_app_tof2(shdl, dt, app_id, app)
            logger.info("Configured PFC app")
            self.conn_mgr.pktgen_app_enable(shdl, dt, app_id)
            logger.info("Enabled PFC app")

            # Create a PPG on the ingress port
            ppg = self.tm.tm_allocate_ppg(dev_id, ing_port)
            print("ppg:", ppg)
            self.tm.tm_set_ppg_guaranteed_min_limit(
                dev_id, ppg, 100)  # 200, 70
            self.tm.tm_set_ppg_skid_limit(dev_id, ppg, 60)  # 120, 30
            self.tm.tm_set_ppg_icos_mapping(dev_id, ppg, 0xFF)
            self.tm.tm_enable_lossless_treatment(dev_id, ppg)
            # Enable PFC (not pause)
            self.tm.tm_set_port_flowcontrol_mode(dev_id, ing_port, 1)
            icos_map = tm_pfc_cos_map_t(CoS0_to_iCos=0,
                                        CoS1_to_iCos=1,
                                        CoS2_to_iCos=2,
                                        CoS3_to_iCos=3,
                                        CoS4_to_iCos=4,
                                        CoS5_to_iCos=5,
                                        CoS6_to_iCos=6,
                                        CoS7_to_iCos=7)
            self.tm.tm_set_port_pfc_cos_mapping(dev_id, ing_port, icos_map)
            # Queue Config
            self.tm.tm_set_q_guaranteed_min_limit(dev_id, egr_port, 0, 100)
            #self.tm.tm_set_q_guaranteed_min_limit(dev_id, egr_port, 0, 1000)
            self.tm.tm_set_q_shaping_rate(
                dev_id, egr_port, 0, True, 100, 250)  # 100, 250
            self.tm.tm_enable_q_max_shaping_rate(dev_id, egr_port, 0)

            input("Hit enter to clean up")

        finally:
            if shdl:
                self.conn_mgr.client_cleanup(shdl)
