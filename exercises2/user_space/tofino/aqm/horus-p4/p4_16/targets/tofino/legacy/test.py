# Test
import logging
import time
from ptf import config
import ptf.testutils as testutils
from bfruntime_client_base_tests import BfRuntimeTest
import bfrt_grpc.bfruntime_pb2 as bfruntime_pb2
import bfrt_grpc.client as client
import random
from pkts import *
import os
import collections
import math

logger = logging.getLogger('Test')
if not len(logger.handlers):
    logger.addHandler(logging.StreamHandler())

num_pipes = int(testutils.test_param_get('num_pipes'))
pipes = list(range(num_pipes))

swports = []
for device, port, ifname in config["interfaces"]:
    pipe = port >> 7
    if pipe in pipes:
        swports.append(port)
swports.sort()

SAMPLE_ETH_SRC = "AA:BB:CC:DD:EE:00"
SAMPLE_ETH_DST = "EE:DD:CC:BB:AA:00"
SAMPLE_IP_SRC = "192.168.0.10"
SAMPLE_IP_DST = "192.168.0.11"
eth_kwargs = {
        'src_mac': SAMPLE_ETH_SRC,
        'dst_mac': SAMPLE_ETH_DST
        }
TEST_VCLUSTER_ID = 0
MAX_VCLUSTER_WORKERS = 32
INVALID_VALUE_8bit = 0x7F
INVALID_VALUE_16bit = 0x7FFF

def VerifyReadRegisters(register_name, pipe_id, expected_register_value, data_dict):
    # since the table is symmetric and exists only in stage 0, we know that the response data is going to have
    # 8 data fields (2 (hi and lo) * 4 (num pipes) * 1 (num_stages)). bfrt_server returns all (4) register values
    # corresponding to one field id followed by all (4) register values corresponding to the other field id
    logger.info("Verifying read back register values...")
    value = data_dict[register_name][pipe_id]
    assert value == expected_register_value, "Register field didn't match with the read value." \
                                             " Expected Value (%s) : Read value (%s)" % (
                                             str(expected_register_value), str(value))

def register_write(target, register_object, register_name, index, register_value):
        logger.info("Inserting entry in %s register with value = %s ", str(register_name), str(register_value))

        register_object.entry_add(
            target,
            [register_object.make_key([client.KeyTuple('$REGISTER_INDEX', index)])],
            [register_object.make_data([client.DataTuple(register_name, register_value)])])

# Reads a given register at given index for and validates the expected value (if given) for a specific pipe
# If expected_register_value is not given, does not verify and only logs the register value
def test_register_read(target, register_object, register_name, pipe_id, index, expected_register_value=None):
        
        resp = register_object.entry_get(
                target,
                [register_object.make_key([client.KeyTuple('$REGISTER_INDEX', index)])],
                {"from_hw": True})
        # TODO: Modify 0 for other pipe IDs currenlty showing only pipe 0 value
        
        data_dict = next(resp)[0].to_dict()
        res = data_dict[register_name][0]
        logger.info("Reading Register: %s [%d] = %d", str(register_name), index, res)
        #print(data_dict) 
        if(expected_register_value):
            VerifyReadRegisters(register_name, pipe_id, expected_register_value, data_dict)
        return res

class TestFalconLeaf(BfRuntimeTest):
    def setUp(self):
        client_id = 0
        p4_name = "falcon"
        self.target = client.Target(device_id=0, pipe_id=0xffff) ## Here pipe_id  0xffff it mean All Pipes ? TODO: how to set spcific pipe?
        self.tables = []
        BfRuntimeTest.setUp(self, client_id, p4_name)
        
    def cleanup(self):
        """ Delete all the stored entries. """
        print("\n")
        print("Table Cleanup:")
        print("==============")
        try:
            for t in reversed(self.tables):
                print(("  Clearing Table {}".format(t.info.name_get())))
                t.entry_del(self.target)
        except Exception as e:
            print(("Error cleaning up: {}".format(e)))
    
    def init_tables(self):
        bfrt_info = self.interface.bfrt_info_get("falcon")
        # Registers
        self.register_idle_count = bfrt_info.table_get("LeafIngress.idle_count")
        self.register_idle_list = bfrt_info.table_get("LeafIngress.idle_list")
        self.register_aggregate_queue_len = bfrt_info.table_get("LeafIngress.aggregate_queue_len_list")
        self.register_linked_sq_sched = bfrt_info.table_get("LeafIngress.linked_sq_sched")
        self.register_linked_iq_sched = bfrt_info.table_get("LeafIngress.linked_iq_sched")
        self.register_queue_len_list_1 = bfrt_info.table_get("LeafIngress.queue_len_list_1")
        self.register_queue_len_list_2 = bfrt_info.table_get("LeafIngress.queue_len_list_2")
        self.register_deferred_list_1 = bfrt_info.table_get("LeafIngress.deferred_queue_len_list_1")
        self.register_deferred_list_2 = bfrt_info.table_get("LeafIngress.deferred_queue_len_list_2")
        self.register_spine_probed_id = bfrt_info.table_get("LeafIngress.spine_probed_id")
        self.register_spine_iq_len_1 = bfrt_info.table_get("LeafIngress.spine_iq_len_1")

        # MA Tables
        self.forward_falcon_switch_dst = bfrt_info.table_get("LeafIngress.forward_falcon_switch_dst")
        self.forward_falcon_switch_dst.info.key_field_annotation_add("hdr.falcon.dst_id", "wid")
        self.set_queue_len_unit = bfrt_info.table_get("LeafIngress.set_queue_len_unit")
        self.set_queue_len_unit.info.key_field_annotation_add("hdr.falcon.cluster_id", "vcid")
        self.get_cluster_num_valid = bfrt_info.table_get("LeafIngress.get_cluster_num_valid")
        self.get_cluster_num_valid.info.key_field_annotation_add("hdr.falcon.cluster_id", "vcid")
        self.adjust_random_range_ds = bfrt_info.table_get("LeafIngress.adjust_random_range_ds")
        self.adjust_random_range_ds.info.key_field_annotation_add("falcon_md.cluster_num_valid_ds", "num_valid_ds")
        self.adjust_random_range_us = bfrt_info.table_get("LeafIngress.adjust_random_range_us")
        self.adjust_random_range_us.info.key_field_annotation_add("falcon_md.cluster_num_valid_us", "num_valid_us")
        self.get_spine_dst_id = bfrt_info.table_get("LeafIngress.get_spine_dst_id")
        self.get_spine_dst_id.info.key_field_annotation_add("falcon_md.random_id_1", "random_id")

        # HW config tables (Mirror and multicast)
        self.mirror_cfg_table = bfrt_info.table_get("$mirror.cfg")

        # Add tables to list for easier cleanup()
        self.tables.append(self.register_idle_count)
        self.tables.append(self.register_idle_list)
        self.tables.append(self.register_aggregate_queue_len)
        self.tables.append(self.register_linked_sq_sched)
        self.tables.append(self.register_linked_iq_sched)
        self.tables.append(self.register_queue_len_list_1)
        self.tables.append(self.register_queue_len_list_2)
        self.tables.append(self.register_deferred_list_1)
        self.tables.append(self.register_deferred_list_2)
        self.tables.append(self.register_spine_probed_id)
        self.tables.append(self.register_spine_iq_len_1)

        self.tables.append(self.adjust_random_range_ds)
        self.tables.append(self.adjust_random_range_us)
        self.tables.append(self.forward_falcon_switch_dst)
        self.tables.append(self.set_queue_len_unit)
        self.tables.append(self.get_cluster_num_valid)
        self.tables.append(self.get_spine_dst_id)

        #self.tables.append(self.mirror_cfg_table)

    def runTest(self):
        # TODO: fix this, test script does not recognize the exported env variables
        #test_case = os.environ.get('FALCON_TEST_CASE')
        test_case = 'sq_link_function'
        logger.info("\n***** Starting Test Case: %s \n", str(test_case))
        
        self.init_tables()
        self.cleanup()

        if test_case == 'schedule_idle_state':# Testing scheduling when idle worker are available
            self.schedule_task_idle_state()
        elif test_case == 'schedule_sq_state':# Testing scheduling when no idle worker is available
            self.schedule_task_sq_state()
        elif test_case == 'idle_link_function':
            self.idle_link_function()
        elif test_case == 'sq_link_function':
            self.sq_link_function()

    # Test scenario:
    # Leaf starts with no SQ link
    # 1. One spine sends QUEUE_SCAN packet to get the queue length info:
    # Expected: 1. Leaf should report its aggregate queue length as QUEUE_SIGNAL_INIT AND 2. store the spine id as linked SQ
    # 2. A task done packet is received from a worker
    # Expected: 1. Leaf should report the updated aggreget queue length as QUEUE_SIGNAL to the linked spine AND 2. Forwards original response packet to the correct dst
    # 3. Spine sends a QUEUE_REMOVE packet to the leaf
    # Expected: Leaf removes the linkage
    def sq_link_function(self):
        pipe_id = 0

        initial_idle_list = []
        initial_idle_count = len(initial_idle_list)
        wid_port_mapping = {11:1, 12: 2, 13: 2, 14:1, 100: 10}
        spine_port_mapping = {1001: 12, 1002: 13}
        port_mac_mapping = {1: 'F8:F2:1E:3A:13:EC', 2: 'F8:F2:1E:3A:13:C4', 10:'AA:BB:CC:DD:EE:FF', 12:SAMPLE_ETH_DST, 13:SAMPLE_ETH_DST}
        spine_port_mapping = collections.OrderedDict(sorted(spine_port_mapping.items()))
        initial_agg_qlen = 10
        qlen_unit = 0b00000010 # assuming 3bit showing fraction and 5bit decimal: 0.25 (4 workers in this rack)

        initial_linked_iq_spine = INVALID_VALUE_16bit # ID of linked spine for Idle link (Invalid = 0x00F0)
        initial_linked_sq_spine = INVALID_VALUE_16bit # ID of linked spine for SQ link (Invalid = 0x00F0)
        
        # Scenario when <num_task_done> workers finish their task 
        num_task_done = 10
        num_runs = 2 # Number of times to link and unlink to spine

        num_valid_ds_elements = 4 # num bits for available workers for this vcluster in this rack (the worker number will be 2^W)
        num_valid_us_elements = 1

        register_write(self.target,
                self.register_linked_iq_sched,
                register_name='LeafIngress.linked_iq_sched.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_linked_iq_spine)

        register_write(self.target,
                self.register_linked_sq_sched,
                register_name='LeafIngress.linked_sq_sched.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_linked_sq_spine)

        register_write(self.target,
            self.register_idle_count,
            register_name='LeafIngress.idle_count.f1',
            index=TEST_VCLUSTER_ID,
            register_value=initial_idle_count)
        
        register_write(self.target,
                self.register_aggregate_queue_len,
                register_name='LeafIngress.aggregate_queue_len_list.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_agg_qlen)
        
        register_write(self.target,
            self.register_spine_iq_len_1,
            register_name='LeafIngress.spine_iq_len_1.f1',
            index=TEST_VCLUSTER_ID,
            register_value=INVALID_VALUE_8bit)

        register_write(self.target,
            self.register_spine_probed_id,
            register_name='LeafIngress.spine_probed_id.f1',
            index=TEST_VCLUSTER_ID,
            register_value=INVALID_VALUE_16bit)

        # Insert idle_list values (wid of idle workers)
        for i in range(initial_idle_count):
            register_write(self.target,
                self.register_idle_list,
                register_name='LeafIngress.idle_list.f1',
                index=i,
                register_value=initial_idle_list[i])

            test_register_read(self.target,
                self.register_idle_list,
                register_name="LeafIngress.idle_list.f1",
                pipe_id=pipe_id,
                index=i,
                expected_register_value=initial_idle_list[i])

        logger.info("********* Populating Table Entires *********")
        # Insert port mapping for workers
        for wid in wid_port_mapping.keys():
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', wid)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', wid_port_mapping[wid]), client.DataTuple('dst_mac', client.mac_to_bytes(port_mac_mapping[wid_port_mapping[wid]]))],
                                             'LeafIngress.act_forward_falcon')]
            )
        logger.info("Inserted entries in forward_falcon_switch_dst table with key-values = %s ", str(wid_port_mapping))
        
        for i, spine_id in enumerate(spine_port_mapping.keys()):
            self.get_spine_dst_id.entry_add(
                self.target,
                    [self.get_spine_dst_id.make_key([client.KeyTuple('falcon_md.random_id_1', i)])],
                    [self.get_spine_dst_id.make_data([client.DataTuple('spine_dst_id', spine_id)],
                                                 'LeafIngress.act_get_spine_dst_id')]
                )
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', spine_id)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', spine_port_mapping[spine_id]), client.DataTuple('dst_mac', client.mac_to_bytes(port_mac_mapping[spine_port_mapping[spine_id]]))],
                                             'LeafIngress.act_forward_falcon')]
            )
        # Insert qlen unit entries
        self.set_queue_len_unit.entry_add(
                self.target,
                [self.set_queue_len_unit.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                [self.set_queue_len_unit.make_data([client.DataTuple('cluster_unit', qlen_unit)],
                                             'LeafIngress.act_set_queue_len_unit')]
            )
        self.get_cluster_num_valid.entry_add(
                self.target,
                [self.get_cluster_num_valid.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_cluster_num_valid.make_data([client.DataTuple('num_ds_elements', num_valid_ds_elements), client.DataTuple('num_us_elements', num_valid_us_elements)],
                                             'LeafIngress.act_get_cluster_num_valid')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('falcon_md.cluster_num_valid_us', 2)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_1')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('falcon_md.cluster_num_valid_us', 4)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_2')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('falcon_md.cluster_num_valid_us', 16)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_4')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('falcon_md.cluster_num_valid_us', 256)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_8')]
            )

        logger.info("\n********* Configure mirror session *********\n")
        
        # The mirror_cfg table controls what a specific mirror session id does to a packet.
        # This is programming the mirror block in hardware.
        # mirror_cfg_bfrt_key is equivalent to old "mirror_id" in PD term
        for wid in wid_port_mapping.keys():
            mirror_cfg_bfrt_key  = self.mirror_cfg_table.make_key([client.KeyTuple('$sid', wid)])
            mirror_cfg_bfrt_data = self.mirror_cfg_table.make_data([
                client.DataTuple('$direction', str_val="INGRESS"),
                client.DataTuple('$ucast_egress_port', swports[wid_port_mapping[wid]]),
                client.DataTuple('$ucast_egress_port_valid', bool_val=True),
                client.DataTuple('$session_enable', bool_val=True),
            ], "$normal")
            self.mirror_cfg_table.entry_add(self.target, [ mirror_cfg_bfrt_key ], [ mirror_cfg_bfrt_data ])

        for run_id in range(num_runs):
            logger.info("\n********* Sending PKT_TYPE_SCAN_QUEUE_SIGNAL packet from spine *********\n")
            dst_id = 100 # this will be identifier for client (receiving reply pkt)
            src_id = spine_port_mapping.keys()[0]
            ig_port = spine_port_mapping[src_id]
            scan_queue_pkt = make_falcon_scan_queue_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, seq_num=0x10+i, **eth_kwargs)
            expected_pkt = make_falcon_queue_signal_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=src_id, seq_num=0x10+i, is_init=True, **eth_kwargs)
            logger.info("Sending packet on port %d", ig_port)
            testutils.send_packet(self, ig_port, scan_queue_pkt)
            logger.info(" Verifying expected QUEUE_SIGNAL_INIT on port %d", ig_port)
            testutils.verify_packets(self, expected_pkt, [ig_port])
            # poll_res = testutils.dp_poll(self)
            # (rcv_device, rcv_port, rcv_pkt, pkt_time) = poll_res
            # print(poll_res)
            test_register_read(self.target,
                self.register_linked_sq_sched,
                'LeafIngress.linked_sq_sched.f1',
                pipe_id,
                TEST_VCLUSTER_ID, 
                src_id)
            for i in range(num_task_done):
                task_done_packet = make_falcon_task_done_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=4, dst_id=dst_id, is_idle=False, q_len=i, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
                
                logger.info("\n********* Sending TASK_DONE packet from a worker *********")
                testutils.send_packet(self, ig_port, task_done_packet)
                
                logger.info("Manually verify: TASK_DONE should be forwarded to its destination")
                if (i%num_valid_ds_elements == 0): # This is were an additional update message should be sent to spine
                    logger.info("***** a QUEUE_SIGNAL should be sent to linked spine *****") 
                for i in range(2):
                    poll_res = testutils.dp_poll(self)
                    (rcv_device, rcv_port, rcv_pkt, pkt_time) = poll_res
                    print(poll_res)

            logger.info("\n********* Sending QUEUE_REMOVE packet from the linked spine *********")
            queue_remove_pkt = make_falcon_queue_remove_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, seq_num=0x10+i, **eth_kwargs)
            testutils.send_packet(self, ig_port, queue_remove_pkt)
            poll_res = testutils.dp_poll(self)

            logger.info("\n********* Verifying switch internal state: linked spine should be INVALID value *********")
            test_register_read(self.target,
                self.register_linked_sq_sched,
                'LeafIngress.linked_sq_sched.f1',
                pipe_id,
                TEST_VCLUSTER_ID, 
                INVALID_VALUE_16bit)
            for i in range(4):
                task_done_packet = make_falcon_task_done_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=4, dst_id=dst_id, is_idle=False, q_len=i, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
                
                logger.info("\n********* After unlink: Sending TASK_DONE packet from worker  *********")
                testutils.send_packet(self, ig_port, task_done_packet)
                
                logger.info("****** Manually verify: Only the TASK_DONE should be forwarded to its destination and no more QUEUE_SIGNAL messages ******")
                for i in range(2):
                    poll_res = testutils.dp_poll(self)
                    (rcv_device, rcv_port, rcv_pkt, pkt_time) = poll_res
                    print(poll_res)
        for wid in wid_port_mapping.keys():
            mirror_cfg_bfrt_key  = self.mirror_cfg_table.make_key([client.KeyTuple('$sid', wid)])
            self.mirror_cfg_table.entry_del(self.target, [mirror_cfg_bfrt_key]) 

    # Test scenario: 
    # Starts with no idle state. 
    # 1. One worker sends IDLE signal, 
    # Expected: 1.incremented idle count in leaf AND 2. leaf should forward the workers original resd AND 3. send another control pkt to a random spine to check the len of idle in that spine
    # 2. Spine responds with its idle len, 
    # Expected: 1.leaf should store the first idle len AND 2. send another probe to a random spine
    # 3. Second spine responds with its idle len
    # Expected: 1.leaf should compare these and store the shortest one as linked_IQ AND 2. send a PKT_TYPE_IDLE_SIGNAL to the selected spine
    def idle_link_function(self):
        pipe_id = 0

        initial_idle_list = []
        initial_idle_count = len(initial_idle_list)
        wid_port_mapping = {11:1, 12: 2, 13:3, 14: 4, 15:5, 16: 6, 17:7, 18: 8, 100: 10}
        spine_port_mapping = {1001: 12, 1002: 13}
        spine_port_mapping = collections.OrderedDict(sorted(spine_port_mapping.items()))
        spine_idle_qlen = [2, 6]
        initial_agg_qlen = 0
        qlen_unit = 0b00000010 # assuming 3bit showing fraction and 5bit decimal: 0.25 (4 workers in this rack)

        initial_linked_iq_spine = INVALID_VALUE_16bit # ID of linked spine for Idle link (Invalid = 0x00F0)
        initial_linked_sq_spine = INVALID_VALUE_16bit # ID of linked spine for SQ link (Invalid = 0x00F0)
        
        # Scenario when <num_task_done> workers finish their task 
        num_task_done = 1

        num_valid_ds_elements = 2 # num bits for available workers for this vcluster in this rack (the worker number will be 2^W)
        num_valid_us_elements = 1

        register_write(self.target,
                self.register_linked_iq_sched,
                register_name='LeafIngress.linked_iq_sched.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_linked_iq_spine)

        register_write(self.target,
                self.register_linked_sq_sched,
                register_name='LeafIngress.linked_sq_sched.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_linked_sq_spine)

        register_write(self.target,
            self.register_idle_count,
            register_name='LeafIngress.idle_count.f1',
            index=TEST_VCLUSTER_ID,
            register_value=initial_idle_count)
        
        register_write(self.target,
                self.register_aggregate_queue_len,
                register_name='LeafIngress.aggregate_queue_len_list.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_agg_qlen)
        
        register_write(self.target,
            self.register_spine_iq_len_1,
            register_name='LeafIngress.spine_iq_len_1.f1',
            index=TEST_VCLUSTER_ID,
            register_value=INVALID_VALUE_8bit)

        register_write(self.target,
            self.register_spine_probed_id,
            register_name='LeafIngress.spine_probed_id.f1',
            index=TEST_VCLUSTER_ID,
            register_value=INVALID_VALUE_16bit)

        # Simply verify the read from hw to check correct values were inserted
        test_register_read(self.target,
            self.register_idle_count,
            'LeafIngress.idle_count.f1',
            pipe_id,
            TEST_VCLUSTER_ID,
            initial_idle_count)

        test_register_read(self.target,
            self.register_aggregate_queue_len,
            'LeafIngress.aggregate_queue_len_list.f1',
            pipe_id,
            TEST_VCLUSTER_ID,
            initial_agg_qlen)

        
        # Insert idle_list values (wid of idle workers)
        for i in range(initial_idle_count):
            register_write(self.target,
                self.register_idle_list,
                register_name='LeafIngress.idle_list.f1',
                index=i,
                register_value=initial_idle_list[i])

            test_register_read(self.target,
                self.register_idle_list,
                register_name="LeafIngress.idle_list.f1",
                pipe_id=pipe_id,
                index=i,
                expected_register_value=initial_idle_list[i])

        logger.info("********* Populating Table Entires *********")
        # Insert port mapping for workers
        for wid in wid_port_mapping.keys():
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', wid)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', wid_port_mapping[wid])],
                                             'LeafIngress.act_forward_falcon')]
            )
        logger.info("Inserted entries in forward_falcon_switch_dst table with key-values = %s ", str(wid_port_mapping))
        
        for i, spine_id in enumerate(spine_port_mapping.keys()):
            self.get_spine_dst_id.entry_add(
                self.target,
                    [self.get_spine_dst_id.make_key([client.KeyTuple('falcon_md.random_id_1', i)])],
                    [self.get_spine_dst_id.make_data([client.DataTuple('spine_dst_id', spine_id)],
                                                 'LeafIngress.act_get_spine_dst_id')]
                )
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', spine_id)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', spine_port_mapping[spine_id])],
                                             'LeafIngress.act_forward_falcon')]
            )
        # Insert qlen unit entries
        self.set_queue_len_unit.entry_add(
                self.target,
                [self.set_queue_len_unit.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                [self.set_queue_len_unit.make_data([client.DataTuple('cluster_unit', qlen_unit)],
                                             'LeafIngress.act_set_queue_len_unit')]
            )
        self.get_cluster_num_valid.entry_add(
                self.target,
                [self.get_cluster_num_valid.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_cluster_num_valid.make_data([client.DataTuple('num_ds_elements', num_valid_ds_elements), client.DataTuple('num_us_elements', num_valid_us_elements)],
                                             'LeafIngress.act_get_cluster_num_valid')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('falcon_md.cluster_num_valid_us', 1)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_1')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('falcon_md.cluster_num_valid_us', 2)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_2')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('falcon_md.cluster_num_valid_us', 4)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_4')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('falcon_md.cluster_num_valid_us', 8)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_8')]
            )

        logger.info("\n********* Configure mirror session *********\n")
        # The mirror_cfg table controls what a specific mirror session id does to a packet.
        # This is programming the mirror block in hardware.
        # mirror_cfg_bfrt_key is equivalent to old "mirror_id" in PD term
        for wid in wid_port_mapping.keys():
            mirror_cfg_bfrt_key  = self.mirror_cfg_table.make_key([client.KeyTuple('$sid', wid)])
            mirror_cfg_bfrt_data = self.mirror_cfg_table.make_data([
                client.DataTuple('$direction', str_val="INGRESS"),
                client.DataTuple('$ucast_egress_port', swports[wid_port_mapping[wid]]),
                client.DataTuple('$ucast_egress_port_valid', bool_val=True),
                client.DataTuple('$session_enable', bool_val=True),
            ], "$normal")
            self.mirror_cfg_table.entry_add(self.target, [ mirror_cfg_bfrt_key ], [ mirror_cfg_bfrt_data ])

        logger.info("\n********* Sending PKT_TYPE_TASK_DONE_IDLE packet from worker *********\n")
        dst_id = 100 # this will be identifier for client (receiving reply pkt)
        src_id = wid_port_mapping.keys()[i]
        ig_port = wid_port_mapping[src_id]

        task_done_idle_packet = make_falcon_task_done_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, is_idle=True, q_len=i, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
        expected_packet = make_falcon_task_done_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, is_idle=True, q_len=i, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
        expected_packet_probe_idle = make_falcon_probe_idle_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=1001, pkt_len=0, seq_num=0x10+i, **eth_kwargs)            
        logger.info("Sending done_idle packet on port %d", ig_port)
        testutils.send_packet(self, ig_port, task_done_idle_packet)
        #testutils.verify_packets(self, expected_packet_probe_idle, [12, 13])
        #testutils.verify_any_packet_any_port(self, [expected_packet_probe_idle], [swports[12] , swports[13]])
        logger.info("\n[Note] When first idle pkt is received, swith should send another pkt to spines to probe their IQ len, manually tested in Tofino modle logs\n")
        # TODO: Check multiple packets on multiple ports? testutils.verify_any_packet_any_port fails
        for i in range(2):
            poll_res = testutils.dp_poll(self)
            (rcv_device, rcv_port, rcv_pkt, pkt_time) = poll_res
            print(poll_res)
            
        logger.info("*** Checking the idle_count increase ***")
        test_register_read(self.target,
            self.register_idle_count,
            'LeafIngress.idle_count.f1',
            pipe_id,
            TEST_VCLUSTER_ID)

        logger.info("\n*** Spine 1 sends idle response ***")
        spine_src_1 = spine_port_mapping.keys()[0]
        probe_resp_1 = make_falcon_probe_idle_response_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=spine_src_1, dst_id=dst_id, seq_num=0x10+i, q_len=spine_idle_qlen[0], **eth_kwargs)
        testutils.send_packet(self, ig_port, probe_resp_1)
        logger.info("\n*** Leaf should send another probe to random spine ***")
        poll_res = testutils.dp_poll(self)
        (rcv_device, rcv_port, rcv_pkt, pkt_time) = poll_res
        print(poll_res)
        logger.info("\n*** Checking interinal state for the time between first spine response and second spine response ***")
        test_register_read(self.target,
            self.register_spine_probed_id,
            'LeafIngress.spine_probed_id.f1',
            pipe_id,
            TEST_VCLUSTER_ID,
            spine_src_1)
        test_register_read(self.target,
            self.register_spine_iq_len_1,
            'LeafIngress.spine_iq_len_1.f1',
            pipe_id,
            TEST_VCLUSTER_ID,
            spine_idle_qlen[0])

        logger.info("\n*** Spine 2 sends idle response ***")
        spine_src_2 = spine_port_mapping.keys()[1]
        probe_resp_2 = make_falcon_probe_idle_response_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=spine_src_2, dst_id=dst_id, seq_num=0x10+i, q_len=spine_idle_qlen[1], **eth_kwargs)
        testutils.send_packet(self, ig_port, probe_resp_2)
        logger.info("\n*** Leaf should send an idle signal to selected spine (one with min idle queue length)***")
        poll_res = testutils.dp_poll(self)
        print(poll_res)

        logger.info("\n*** Checking internal switch state ***")
        logger.info("*** spine_probed_id and spine_iq_len_1 should reset to INVALID (0x7F)  ***")
        test_register_read(self.target,
            self.register_spine_probed_id,
            'LeafIngress.spine_probed_id.f1',
            pipe_id,
            TEST_VCLUSTER_ID, 
            INVALID_VALUE_16bit)
        test_register_read(self.target,
            self.register_spine_iq_len_1,
            'LeafIngress.spine_iq_len_1.f1',
            pipe_id,
            TEST_VCLUSTER_ID,
            INVALID_VALUE_8bit)

        logger.info("*** linked_iq_sched  Should be the spine with min idle queue length ***")
        test_register_read(self.target,
            self.register_linked_iq_sched,
            'LeafIngress.linked_iq_sched.f1',
            pipe_id,
            TEST_VCLUSTER_ID, 
            spine_src_1)

        for wid in wid_port_mapping.keys():
            mirror_cfg_bfrt_key  = self.mirror_cfg_table.make_key([client.KeyTuple('$sid', wid)])
            self.mirror_cfg_table.entry_del(self.target, [mirror_cfg_bfrt_key]) 

    def schedule_task_idle_state(self):
        pipe_id = 0

        initial_idle_count = 4
        initial_idle_list = [15, 11, 12, 17]
        wid_port_mapping = {11:1, 12: 2, 13:3, 14: 4, 15:5, 16: 6, 17:7, 18: 8, 100: 10}

        initial_agg_qlen = 0
        qlen_unit = 0b00000010 # assuming 3bit showing fraction and 5bit decimal: 0.25 (4 workers in this rack)

        initial_linked_iq_spine = 105 # ID of linked spine for Idle link
        initial_linked_sq_spine = 0xFFFF # ID of linked spine for SQ link (Invalid = 0xFFFF, since we are in idle state)
        
        # Scenario when 2 workers finish their task 
        num_task_done = 2

        register_write(self.target,
                self.register_linked_iq_sched,
                register_name='LeafIngress.linked_iq_sched.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_linked_iq_spine)

        register_write(self.target,
                self.register_linked_sq_sched,
                register_name='LeafIngress.linked_sq_sched.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_linked_sq_spine)

        register_write(self.target,
            self.register_idle_count,
            register_name='LeafIngress.idle_count.f1',
            index=TEST_VCLUSTER_ID,
            register_value=initial_idle_count)
        
        register_write(self.target,
                self.register_aggregate_queue_len,
                register_name='LeafIngress.aggregate_queue_len_list.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_agg_qlen)

        # Simply verify the read from hw to check correct values were inserted
        test_register_read(self.target,
            self.register_idle_count,
            'LeafIngress.idle_count.f1',
            pipe_id,
            TEST_VCLUSTER_ID,
            initial_idle_count)

        test_register_read(self.target,
            self.register_aggregate_queue_len,
            'LeafIngress.aggregate_queue_len_list.f1',
            pipe_id,
            TEST_VCLUSTER_ID,
            initial_agg_qlen)

        # Insert idle_list values (wid of idle workers)
        for i in range(initial_idle_count):
            register_write(self.target,
                self.register_idle_list,
                register_name='LeafIngress.idle_list.f1',
                index=i,
                register_value=initial_idle_list[i])

            test_register_read(self.target,
                self.register_idle_list,
                register_name="LeafIngress.idle_list.f1",
                pipe_id=pipe_id,
                index=i,
                expected_register_value=initial_idle_list[i])

        logger.info("********* Populating Table Entires *********")
        # Insert port mapping for workers
        for wid in wid_port_mapping.keys():
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', wid)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', wid_port_mapping[wid])],
                                             'LeafIngress.act_forward_falcon')]
            )
        logger.info("Inserted entries in forward_falcon_switch_dst table with key-values = %s ", str(wid_port_mapping))
        
        # Insert qlen unit entries
        self.set_queue_len_unit.entry_add(
                self.target,
                [self.set_queue_len_unit.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                [self.set_queue_len_unit.make_data([client.DataTuple('cluster_unit', qlen_unit)],
                                             'LeafIngress.act_set_queue_len_unit')]
            )

        logger.info("********* Sending NEW_TASK packets, Testing scheduling on idle workers *********")
        
        expected_agg_qlen = initial_agg_qlen
        for i in range(initial_idle_count):
            test_register_read(self.target,
                self.register_idle_count,
                'LeafIngress.idle_count.f1',
                pipe_id,
                TEST_VCLUSTER_ID,
                initial_idle_count - i)

            idle_pointer_stack = initial_idle_count - i - 1 # This is the expected stack pointer in the switch (iterates the list from the last element)
            ig_port = swports[random.randint(9, 15)] # port connected to spine
            eg_port = swports[wid_port_mapping[initial_idle_list[idle_pointer_stack]]]
            src_id = random.randint(1, 255)
            dst_id = random.randint(1, 255)

            new_task_packet = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=random.randint(1, 255), pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            expected_packet = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=initial_idle_list[idle_pointer_stack], pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            
            logger.info("Sending task packet on port %d", ig_port)
            testutils.send_packet(self, ig_port, new_task_packet)
            logger.info("Expecting task packet on port %d (IFACE-OUT)", eg_port)
            testutils.verify_packets(self, expected_packet, [eg_port])
            expected_agg_qlen += qlen_unit
            test_register_read(self.target,
                self.register_aggregate_queue_len,
                'LeafIngress.aggregate_queue_len_list.f1',
                pipe_id,
                TEST_VCLUSTER_ID,
                expected_agg_qlen)
        
        for i in range(num_task_done):
            dst_id = 100 # this will be identifier for client (receiving reply pkt)
            src_id = initial_idle_list[i]
            ig_port = wid_port_mapping[initial_idle_list[i]]
            task_done_idle_packet = make_falcon_task_done_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, is_idle=True, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            expected_packet = make_falcon_task_done_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, is_idle=True, pkt_len=0, seq_num=0x10+i, **eth_kwargs)            
            expected_packet_probe_idle = make_falcon_probe_idle_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, pkt_len=0, seq_num=0x10+i, **eth_kwargs)            
            logger.info("Sending done_idle packet on port %d", ig_port)
            testutils.send_packet(self, ig_port, task_done_idle_packet)
            
            poll_res = testutils.dp_poll(self)
            logger.info("Original response pkt should be forwarded by switch")
            print(poll_res)
            # TODO: Check multiple packets on multiple ports? testutils.verify_any_packet_any_port fails
            logger.info("\n[Note] When first idle pkt is received, swith should send another pkt to spines to probe their IQ len, manually tested in Tofino modle logs\n")
            logger.info("*** Checking the idle_count increase ***")
            test_register_read(self.target,
                self.register_idle_count,
                'LeafIngress.idle_count.f1',
                pipe_id,
                TEST_VCLUSTER_ID,
                i + 1)

    def schedule_task_sq_state(self):
        pipe_id = 0

        initial_idle_count = 0
        initial_idle_list = []
        wid_port_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:4, 6:4, 7:4, 100: 10}

        qlen_unit = 0b00000010
        
        initial_linked_iq_spine = 0xFFFF # ID of linked spine for Idle link (Invalid = 0xFFFF, since we hav no idle workers)
        initial_linked_sq_spine = 0xFFFF # ID of linked spine for SQ link (Invalid = 0xFFFF, test this later) TODO: test sq report to spine

        intitial_qlen_state = [6, 1, 5, 2]
        initial_num_waiting_tasks = sum(intitial_qlen_state)
        initial_agg_qlen = initial_num_waiting_tasks * qlen_unit 

        intitial_deferred_state = [0, 0, 0, 0, 0]
        num_valid_ds_elements = 2 # num bits for available workers for this vcluster in this rack (the worker number will be 2^W)
        num_valid_us_elements = 1
        num_tasks_to_send = 10

        workers_start_idx = TEST_VCLUSTER_ID * MAX_VCLUSTER_WORKERS
        logger.info("***** Writing registers for initial state  *******")
        register_write(self.target,
                self.register_linked_iq_sched,
                register_name='LeafIngress.linked_iq_sched.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_linked_iq_spine)

        register_write(self.target,
                self.register_linked_sq_sched,
                register_name='LeafIngress.linked_sq_sched.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_linked_sq_spine)

        register_write(self.target,
                self.register_aggregate_queue_len,
                register_name='LeafIngress.aggregate_queue_len_list.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_agg_qlen)

        for i, qlen in enumerate(intitial_qlen_state):
            register_write(self.target,
                    self.register_queue_len_list_1,
                    register_name='LeafIngress.queue_len_list_1.f1',
                    index=workers_start_idx + i,
                    register_value=qlen)
            register_write(self.target,
                    self.register_queue_len_list_2,
                    register_name='LeafIngress.queue_len_list_2.f1',
                    index=workers_start_idx + i,
                    register_value=qlen)

            # Initial state for deferred lists is 0
            register_write(self.target,
                    self.register_deferred_list_1,
                    register_name='LeafIngress.deferred_queue_len_list_1.f1',
                    index=workers_start_idx + i,
                    register_value=intitial_deferred_state[i])
            register_write(self.target,
                    self.register_deferred_list_2,
                    register_name='LeafIngress.deferred_queue_len_list_2.f1',
                    index=workers_start_idx + i,
                    register_value=intitial_deferred_state[i])

        logger.info("********* Populating Table Entires *********")
        # TODO: can we add constant entrires for this table in .p4 file?
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('falcon_md.cluster_num_valid_ds', 1)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_1')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('falcon_md.cluster_num_valid_ds', 2)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_2')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('falcon_md.cluster_num_valid_ds', 4)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_4')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('falcon_md.cluster_num_valid_ds', 8)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_8')]
            )

        # Insert port mapping for workers
        for wid in wid_port_mapping.keys():
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', wid)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', wid_port_mapping[wid])],
                                             'LeafIngress.act_forward_falcon')]
            )
        logger.info("Inserted entries in forward_falcon_switch_dst table with key-values = %s ", str(wid_port_mapping))

        # Insert qlen unit entries
        self.set_queue_len_unit.entry_add(
                self.target,
                [self.set_queue_len_unit.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                [self.set_queue_len_unit.make_data([client.DataTuple('cluster_unit', qlen_unit)],
                                             'LeafIngress.act_set_queue_len_unit')]
            )

        # Insert num_valid to set actual num workers for this vcluster (in this rack)
        self.get_cluster_num_valid.entry_add(
                self.target,
                [self.get_cluster_num_valid.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_cluster_num_valid.make_data([client.DataTuple('num_ds_elements', num_valid_ds_elements), client.DataTuple('num_us_elements', num_valid_us_elements)],
                                             'LeafIngress.act_get_cluster_num_valid')]
            )

        logger.info("\n********* Sending NEW_TASK packets, Testing scheduling on busy workers *********\n")
        
        expected_agg_qlen = initial_agg_qlen
        for i in range(num_tasks_to_send):
            test_register_read(self.target, #Idle count should stay at 0
                self.register_idle_count,
                'LeafIngress.idle_count.f1',
                pipe_id,
                TEST_VCLUSTER_ID,
                0)

            ig_port = swports[random.randint(9, 15)] # port connected to spine
            src_id = random.randint(1, 255)
            dst_id = random.randint(1, 255)

            new_task_packet = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=random.randint(1, 255), pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            expected_packet_0 = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=0, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            expected_packet_1 = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=1, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            logger.info("Sending task packet on port %d", ig_port)
            testutils.send_packet(self, ig_port, new_task_packet)
            # Note: Can't expect the packet since random sampling happens inside the switch 

            #logger.info("Expecting task packet to be forwarded on port %d (IFACE-OUT)", eg_port)

            #testutils.verify_packets(self, expected_packet, [eg_port])
            #testutils.verify_any_packet_any_port(self, [expected_packet_0, expected_packet_1], [0,1])
            
            poll_res = testutils.dp_poll(self)
            print(poll_res)
            out_port = int(poll_res[1]) # For simplicity we same dst_id as outport
            
            # Check aggregate queue len is incremented
            expected_agg_qlen += qlen_unit
            test_register_read(self.target,
                self.register_aggregate_queue_len,
                'LeafIngress.aggregate_queue_len_list.f1',
                pipe_id,
                TEST_VCLUSTER_ID,
                expected_agg_qlen)
            #time.sleep(0.5)
            
            qlen_1 = 0
            qlen_2 = 0
            deferred_1 = 0
            deferred_2 = 0
            # manually (log-based) check switch internal state about selected worker is updated 
            for wid in range(len(intitial_qlen_state)):
                qlen_1 += test_register_read(self.target,
                    self.register_queue_len_list_1,
                    'LeafIngress.queue_len_list_1.f1',
                    pipe_id,
                    wid)

                qlen_2 += test_register_read(self.target,
                    self.register_queue_len_list_2,
                    'LeafIngress.queue_len_list_2.f1',
                    pipe_id,
                    wid)

                deferred_1+= test_register_read(self.target,
                    self.register_deferred_list_1,
                    'LeafIngress.deferred_queue_len_list_1.f1',
                    pipe_id,
                    wid)

                deferred_2 += test_register_read(self.target,
                    self.register_deferred_list_2,
                    'LeafIngress.deferred_queue_len_list_2.f1',
                    pipe_id,
                    wid)
        # It is not straightforward to verify intermediate states automatically, we check them manually in the loop above, at the end (loose) overall constraints are checked
        assert (deferred_1 + qlen_1 - initial_num_waiting_tasks) == num_tasks_to_send, "Overall maintained state is not correct"
        assert (deferred_1==deferred_2), "Deferred list copies inconsistent"
        assert (qlen_1==qlen_2), "Queue len list copies inconsistent"

        logger.info("\n********* Sending DONE packets, Testing state updates *********\n")
        for i in range(len(intitial_qlen_state)):
            dst_id = 100 # this will be identifier for client (receiving reply pkt)
            src_id = i # each worker sends one task done to check if states are updated correctly in the switch
            ig_port = 5
            task_done_idle_packet = make_falcon_task_done_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, is_idle=False, q_len=i, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            logger.info("Sending done packet (reply) on port %d", ig_port)
            testutils.send_packet(self, ig_port, task_done_idle_packet)
            
            poll_res = testutils.dp_poll(self)
            print(poll_res)
        logger.info("\n********* Checking state registers *********\n") 
        for wid in range(len(intitial_qlen_state)):
            test_register_read(self.target,
                self.register_queue_len_list_1,
                'LeafIngress.queue_len_list_1.f1',
                pipe_id,
                wid,
                wid)

            qlen_2 += test_register_read(self.target,
                self.register_queue_len_list_2,
                'LeafIngress.queue_len_list_2.f1',
                pipe_id,
                wid,
                wid)

            deferred_1+= test_register_read(self.target,
                self.register_deferred_list_1,
                'LeafIngress.deferred_queue_len_list_1.f1',
                pipe_id,
                wid,
                0)

            deferred_2 += test_register_read(self.target,
                self.register_deferred_list_2,
                'LeafIngress.deferred_queue_len_list_2.f1',
                pipe_id,
                wid,
                0)

class TestFalconSpine(BfRuntimeTest):
    def setUp(self):
        client_id = 0
        p4_name = "falcon"
        self.target = client.Target(device_id=0, pipe_id=0xffff) ## Here pipe_id  0xffff it mean All Pipes ? TODO: how to set spcific pipe?
        self.tables = []
        BfRuntimeTest.setUp(self, client_id, p4_name)
        
    def cleanup(self):
        """ Delete all the stored entries. """
        print("\n")
        print("Table Cleanup:")
        print("==============")
        try:
            for t in reversed(self.tables):
                print(("  Clearing Table {}".format(t.info.name_get())))
                t.entry_del(self.target)
        except Exception as e:
            print(("Error cleaning up: {}".format(e)))
    
    def init_tables(self):
        bfrt_info = self.interface.bfrt_info_get("falcon")
        # Registers
        self.register_idle_count = bfrt_info.table_get("SpineIngress.idle_count")
        self.register_idle_list = bfrt_info.table_get("SpineIngress.idle_list")
        self.register_idle_list_idx_mapping = bfrt_info.table_get("SpineIngress.idle_list_idx_mapping")
        self.register_queue_len_list_1 = bfrt_info.table_get("SpineIngress.queue_len_list_1")
        self.register_queue_len_list_2 = bfrt_info.table_get("SpineIngress.queue_len_list_2")
        self.register_deferred_queue_len_list_1 = bfrt_info.table_get("SpineIngress.deferred_queue_len_list_1")
        self.register_deferred_queue_len_list_2 = bfrt_info.table_get("SpineIngress.deferred_queue_len_list_2")
        self.register_lid_list_1 = bfrt_info.table_get("SpineIngress.lid_list_1")
        self.register_lid_list_2 = bfrt_info.table_get("SpineIngress.lid_list_2")
        self.register_queue_signal_count = bfrt_info.table_get("SpineIngress.queue_signal_count")
        
        # # MA Tables
        self.forward_falcon_switch_dst = bfrt_info.table_get("SpineIngress.forward_falcon_switch_dst")
        self.forward_falcon_switch_dst.info.key_field_annotation_add("hdr.falcon.dst_id", "lid")
        self.set_queue_len_unit_1 = bfrt_info.table_get("SpineIngress.set_queue_len_unit_1")
        self.set_queue_len_unit_1.info.key_field_annotation_add("hdr.falcon.cluster_id", "vcid")
        self.set_queue_len_unit_1.info.key_field_annotation_add("falcon_md.random_id_1", "lid")
        self.set_queue_len_unit_2 = bfrt_info.table_get("SpineIngress.set_queue_len_unit_2")
        self.set_queue_len_unit_2.info.key_field_annotation_add("hdr.falcon.cluster_id", "vcid")
        self.set_queue_len_unit_2.info.key_field_annotation_add("falcon_md.random_id_2", "lid")

        self.get_cluster_num_valid_leafs = bfrt_info.table_get("SpineIngress.get_cluster_num_valid_leafs")
        self.get_cluster_num_valid_leafs.info.key_field_annotation_add("hdr.falcon.cluster_id", "vcid")
        self.adjust_random_range_all_leafs = bfrt_info.table_get("SpineIngress.adjust_random_range_all_leafs")
        self.adjust_random_range_all_leafs.info.key_field_annotation_add("falcon_md.cluster_num_valid_ds", "num_valid_ds")
        self.adjust_random_range_sq_leafs = bfrt_info.table_get("SpineIngress.adjust_random_range_sq_leafs")
        self.adjust_random_range_sq_leafs.info.key_field_annotation_add("falcon_md.cluster_num_valid_queue_signals", "num_valid_queue_signal")
        self.get_leaf_dst_id = bfrt_info.table_get("SpineIngress.get_leaf_dst_id")
        self.get_leaf_dst_id.info.key_field_annotation_add("falcon_md.random_ds_index_1", "random_id")

        self.mgid_table = bfrt_info.table_get("$pre.mgid")
        self.node_table = bfrt_info.table_get("$pre.node")

        # Add tables to list for easier cleanup()
        self.tables.append(self.register_idle_count)
        self.tables.append(self.register_idle_list)
        self.tables.append(self.register_idle_list_idx_mapping)
        self.tables.append(self.register_queue_len_list_1)
        self.tables.append(self.register_queue_len_list_2)
        self.tables.append(self.register_deferred_queue_len_list_1)
        self.tables.append(self.register_deferred_queue_len_list_2)
        self.tables.append(self.register_lid_list_1)
        self.tables.append(self.register_lid_list_2)
        self.tables.append(self.register_queue_signal_count)

        self.tables.append(self.adjust_random_range_all_leafs)
        self.tables.append(self.adjust_random_range_sq_leafs)
        self.tables.append(self.forward_falcon_switch_dst)
        self.tables.append(self.set_queue_len_unit_1)
        self.tables.append(self.set_queue_len_unit_2)
        self.tables.append(self.get_cluster_num_valid_leafs)
        self.tables.append(self.get_leaf_dst_id)

        #self.tables.append(self.mgid_table)
        #self.tables.append(self.node_table)

    def runTest(self):
        # TODO: fix this, test script does not recognize the exported env variables
        #test_case = os.environ.get('FALCON_TEST_CASE')
        test_case = 'transition_to_no_idle'
        logger.info("\n***** Starting Test Case: %s \n", str(test_case))
        
        self.init_tables()
        self.cleanup()

        if test_case == 'schedule_idle_state':# Testing scheduling when idle worker are available
            self.schedule_task_idle_state()
        elif test_case == 'remove_idle_leaf':
            self.remove_idle_leaf()
        elif test_case == 'schedule_task_sq_state':# Testing scheduling when no idle worker is available
            self.schedule_task_sq_state()
        elif test_case == 'transition_to_no_idle':
            self.transition_to_no_idle()
        # elif test_case == 'sq_link_function':
        #     self.sq_link_function()
    
    # Test scenario:
    # Spine starts in state that some idle leafs are available
    # 1. Spine receives idle remove from leafs that don't have any idle workers until there is no idle leafs available
    # Expected: 1. Spine should broadcast PKT_TYPE_SCAN_QUEUE_SIGNAL message to all of the leafs to get their queue lens
    #   The broadcasted pacekt should be forwarded according to mcast port configurations
    def transition_to_no_idle(self):
        pipe_id = 0
        initial_idle_count = 1
        initial_idle_list = [11]
        leaf_port_mapping = {11:1, 12: 2, 13:3, 14: 4, 15:5, 16: 6, 17:7, 18: 8, 100: 10}
        spine_under_test_id = 100
        intitial_qlen_state = []
        initial_lid_list = [] # leaf IDs for the queue lengths
        intitial_deferred_state = [0, 0, 0, 0, 0]
        num_valid_ds_elements = 2 # num bits for available workers for this vcluster in this rack (the worker number will be 2^W)
        max_linked_leafs = 2
        workers_start_idx = TEST_VCLUSTER_ID * MAX_VCLUSTER_WORKERS
        
        l1_id = 1
        mcast_group = 1 # Broadcasting according to the number of leafs in vcluster 
        broadcast_port_ids = [2, 3, 4]


        broadcast_ports = []
        for port_id in broadcast_port_ids:
            broadcast_ports.append(swports[port_id])
        

        try:
            register_write(self.target,
                self.register_idle_count,
                register_name='SpineIngress.idle_count.f1',
                index=TEST_VCLUSTER_ID,
                register_value=initial_idle_count)

            register_write(self.target,
                self.register_queue_signal_count,
                register_name='SpineIngress.queue_signal_count.f1',
                index=TEST_VCLUSTER_ID,
                register_value=0)

            for i, qlen in enumerate(intitial_qlen_state):
                register_write(self.target,
                        self.register_queue_len_list_1,
                        register_name='SpineIngress.queue_len_list_1.f1',
                        index=workers_start_idx + initial_lid_list[i],
                        register_value=qlen)
                register_write(self.target,
                        self.register_queue_len_list_2,
                        register_name='SpineIngress.queue_len_list_2.f1',
                        index=workers_start_idx + initial_lid_list[i],
                        register_value=qlen)

                # Initial state for deferred lists is 0
                register_write(self.target,
                        self.register_deferred_queue_len_list_1,
                        register_name='SpineIngress.deferred_queue_len_list_1.f1',
                        index=workers_start_idx + initial_lid_list[i],
                        register_value=intitial_deferred_state[i])
                register_write(self.target,
                        self.register_deferred_queue_len_list_2,
                        register_name='SpineIngress.deferred_queue_len_list_2.f1',
                        index=workers_start_idx + initial_lid_list[i],
                        register_value=intitial_deferred_state[i])
                
                # Initial leaf ID list for available queue signals
                register_write(self.target,
                    self.register_lid_list_1,
                    register_name='SpineIngress.lid_list_1.f1',
                    index=workers_start_idx + i,
                    register_value=initial_lid_list[i])
                register_write(self.target,
                    self.register_lid_list_2,
                    register_name='SpineIngress.lid_list_2.f1',
                    index=workers_start_idx + i,
                    register_value=initial_lid_list[i])
                
            logger.info("********* Populating Table Entires *********")
            # Insert qlen unit entires for each leaf in cluster
            for i, lid in enumerate(initial_lid_list):
                self.set_queue_len_unit_1.entry_add(
                    self.target,
                    [self.set_queue_len_unit_1.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID), client.KeyTuple('falcon_md.random_id_1', lid)])],
                    [self.set_queue_len_unit_1.make_data([client.DataTuple('cluster_unit', qlen_units[i])],
                                                 'SpineIngress.act_set_queue_len_unit_1')]
                    )
                self.set_queue_len_unit_2.entry_add(
                    self.target,
                    [self.set_queue_len_unit_2.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID), client.KeyTuple('falcon_md.random_id_2', lid)])],
                    [self.set_queue_len_unit_2.make_data([client.DataTuple('cluster_unit', qlen_units[i])],
                                                 'SpineIngress.act_set_queue_len_unit_2')]
                    )

            # Insert port mapping for workers
            for lid in leaf_port_mapping.keys():
                self.forward_falcon_switch_dst.entry_add(
                    self.target,
                    [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', lid)])],
                    [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', leaf_port_mapping[lid])],
                                                 'SpineIngress.act_forward_falcon')]
                )

            # Insert num_valid to set total num leafs for this vcluster
            self.get_cluster_num_valid_leafs.entry_add(
                    self.target,
                    [self.get_cluster_num_valid_leafs.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                    [self.get_cluster_num_valid_leafs.make_data([client.DataTuple('num_leafs', num_valid_ds_elements), client.DataTuple('max_linked_leafs', max_linked_leafs)],
                                                 'SpineIngress.act_get_cluster_num_valid_leafs')]
                )

            logger.info("********* Configuring multicast groups (broadcast to leafs) *********")
            print("Adding MGID table entries")
            
            self.mgid_table.entry_add(
                self.target,
                [self.mgid_table.make_key(
                    [client.KeyTuple('$MGID', mcast_group)])])
            # Allocate a single L1 node
            # We make a single l1 node with rid =1 
            # TODO: What is RID?
            rid = l1_id
            self.node_table.entry_add(
                    self.target,
                    [self.node_table.make_key([
                        client.KeyTuple('$MULTICAST_NODE_ID', l1_id)])],
                    [self.node_table.make_data([
                        client.DataTuple('$MULTICAST_RID', rid),
                        client.DataTuple('$MULTICAST_LAG_ID', int_arr_val=[]),
                        client.DataTuple('$DEV_PORT', int_arr_val=[])])]
                )

            # Add L2 nodes to the L1s
            self.node_table.entry_mod(
                    self.target,
                    [self.node_table.make_key([
                        client.KeyTuple('$MULTICAST_NODE_ID', l1_id)])],
                    [self.node_table.make_data([
                        client.DataTuple('$MULTICAST_RID', rid),
                        client.DataTuple('$MULTICAST_LAG_ID', int_arr_val=[]),
                        client.DataTuple('$DEV_PORT', int_arr_val=broadcast_ports)])]
                )

            self.mgid_table.entry_mod(
                self.target,
                [self.mgid_table.make_key([
                    client.KeyTuple('$MGID', mcast_group)])],
                [self.mgid_table.make_data([
                    client.DataTuple('$MULTICAST_NODE_ID', int_arr_val=[l1_id]),
                    client.DataTuple('$MULTICAST_NODE_L1_XID_VALID', bool_arr_val=[0]),
                    client.DataTuple('$MULTICAST_NODE_L1_XID', int_arr_val=[0])])])
            
            ig_port = swports[random.randint(9, 15)] # port connected to spine
            idle_remove_pkt = make_falcon_idle_remove_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=initial_idle_list[0], dst_id=spine_under_test_id, seq_num=0x10, **eth_kwargs)
            testutils.send_packet(self, ig_port, idle_remove_pkt)
            for i in range(len(broadcast_port_ids)):
                poll_res = testutils.dp_poll(self)
                (rcv_device, rcv_port, rcv_pkt, pkt_time) = poll_res
                print(poll_res)
                assert int(rcv_port) in broadcast_port_ids, "Received packet on an unexpected port"

            # Make sure no other packets are forwarded
            poll_res = testutils.dp_poll(self)
            (rcv_device, rcv_port, rcv_pkt, pkt_time) = poll_res
            assert rcv_pkt is None, "Received additional packet while expecting no more packets"
        
        finally:
            self.mgid_table.entry_del(
                    self.target,
                    [self.mgid_table.make_key([client.KeyTuple('$MGID', mcast_group)])])
            self.node_table.entry_del(
                    self.target,
                    [self.node_table.make_key([client.KeyTuple('$MULTICAST_NODE_ID', l1_id)])])

    def remove_idle_leaf(self):
        pipe_id = 0

        initial_idle_count = 3
        initial_idle_list = [15, 11, 14]

        leaf_port_mapping = {11:1, 12: 2, 13:3, 14: 4, 15:5, 16: 6, 17:7, 18: 8, 100: 10}
        num_idle_remove = 2
        spine_under_test_id = 100
        register_write(self.target,
            self.register_idle_count,
            register_name='SpineIngress.idle_count.f1',
            index=TEST_VCLUSTER_ID,
            register_value=initial_idle_count)
        
        #Insert idle_list values (lid of idle leafs)
        for i in range(initial_idle_count):
            register_write(self.target,
                self.register_idle_list,
                register_name='SpineIngress.idle_list.f1',
                index=i,
                register_value=initial_idle_list[i])

            test_register_read(self.target,
                self.register_idle_list,
                register_name="SpineIngress.idle_list.f1",
                pipe_id=pipe_id,
                index=i,
                expected_register_value=initial_idle_list[i])

            register_write(self.target,
                self.register_idle_list_idx_mapping,
                register_name="SpineIngress.idle_list_idx_mapping.f1",
                index=initial_idle_list[i],
                register_value=i)

        logger.info("********* Populating Table Entires *********")
        # Insert port mapping for workers
        for lid in leaf_port_mapping.keys():
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', lid)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', leaf_port_mapping[lid])],
                                             'SpineIngress.act_forward_falcon')]
            )
        logger.info("Inserted entries in forward_falcon_switch_dst table with key-values = %s ", str(leaf_port_mapping))
        logger.info("********* Sending NEW_TASK packets, Testing scheduling on idle workers *********")
        expected_idle_count = initial_idle_count
        idle_pointer_stack = initial_idle_count - 1 # This is the expected stack pointer in the switch (iterates the list from the last element)
        
        for idles_in_rack in range(num_idle_remove):
            test_register_read(self.target,
                self.register_idle_count,
                'SpineIngress.idle_count.f1',
                pipe_id,
                TEST_VCLUSTER_ID,
                expected_idle_count)
            
            for idle_leaf_id in initial_idle_list:
                test_register_read(self.target,
                    self.register_idle_list_idx_mapping,
                    'SpineIngress.idle_list_idx_mapping.f1',
                    pipe_id,
                    idle_leaf_id)
            ig_port = swports[random.randint(9, 15)] # port connected to spine
            idle_remove_pkt = make_falcon_idle_remove_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=initial_idle_list[idle_pointer_stack], dst_id=spine_under_test_id, seq_num=0x10+i, **eth_kwargs)
            testutils.send_packet(self, ig_port, idle_remove_pkt)
            expected_idle_count -= 1
            idle_pointer_stack -= 1
            poll_res = testutils.dp_poll(self)
            print(poll_res)
            test_register_read(self.target,
                    self.register_idle_count,
                    'SpineIngress.idle_count.f1',
                    pipe_id,
                    TEST_VCLUSTER_ID,
                    expected_idle_count)
            for idle_list_idx in range(initial_idle_count):
                test_register_read(self.target,
                    self.register_idle_list,
                    'SpineIngress.idle_list.f1',
                    pipe_id,
                    idle_list_idx)


    def schedule_task_idle_state(self):
        pipe_id = 0

        initial_idle_count = 3
        initial_idle_list = [15, 11, 14]
        num_idles_in_racks = [3, 2, 2] # Number of idle workers each rack has. After this number of tasks, leaf sends idle remove

        leaf_port_mapping = {11:1, 12: 2, 13:3, 14: 4, 15:5, 16: 6, 17:7, 18: 8, 100: 10}

        spine_under_test_id = 100
        register_write(self.target,
            self.register_idle_count,
            register_name='SpineIngress.idle_count.f1',
            index=TEST_VCLUSTER_ID,
            register_value=initial_idle_count)
        
        #Insert idle_list values (lid of idle leafs)
        for i in range(initial_idle_count):
            register_write(self.target,
                self.register_idle_list,
                register_name='SpineIngress.idle_list.f1',
                index=i,
                register_value=initial_idle_list[i])

            test_register_read(self.target,
                self.register_idle_list,
                register_name="SpineIngress.idle_list.f1",
                pipe_id=pipe_id,
                index=i,
                expected_register_value=initial_idle_list[i])

            register_write(self.target,
                self.register_idle_list_idx_mapping,
                register_name="SpineIngress.idle_list_idx_mapping.f1",
                index=initial_idle_list[i],
                register_value=i)

        logger.info("********* Populating Table Entires *********")
        # Insert port mapping for workers
        for lid in leaf_port_mapping.keys():
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', lid)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', leaf_port_mapping[lid])],
                                             'SpineIngress.act_forward_falcon')]
            )
        logger.info("Inserted entries in forward_falcon_switch_dst table with key-values = %s ", str(leaf_port_mapping))
        logger.info("********* Sending NEW_TASK packets, Testing scheduling on idle workers *********")

        expected_idle_count = initial_idle_count
        idle_pointer_stack = initial_idle_count - 1 # This is the expected stack pointer in the switch (iterates the list from the last element)
        
        for idles_in_rack in num_idles_in_racks:
            test_register_read(self.target,
                self.register_idle_count,
                'SpineIngress.idle_count.f1',
                pipe_id,
                TEST_VCLUSTER_ID,
                expected_idle_count)
            
            for idle_list_idx in range(initial_idle_count):
                test_register_read(self.target,
                    self.register_idle_list,
                    'SpineIngress.idle_list.f1',
                    pipe_id,
                    idle_list_idx)
            for idle_leaf_id in initial_idle_list:
                test_register_read(self.target,
                    self.register_idle_list_idx_mapping,
                    'SpineIngress.idle_list_idx_mapping.f1',
                    pipe_id,
                    idle_leaf_id)
            for i in range(idles_in_rack):
                ig_port = swports[random.randint(9, 15)] # port connected to spine
                eg_port = swports[leaf_port_mapping[initial_idle_list[idle_pointer_stack]]]
                src_id = random.randint(1, 255)
                dst_id = random.randint(1, 255)

                new_task_packet = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=spine_under_test_id, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
                expected_packet = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=spine_under_test_id, dst_id=initial_idle_list[idle_pointer_stack], pkt_len=0, seq_num=0x10+i, **eth_kwargs)
                
                logger.info("Sending task packet on port %d", ig_port)
                testutils.send_packet(self, ig_port, new_task_packet)
                logger.info("Expecting task packet on port %d (IFACE-OUT)", eg_port)
                testutils.verify_packets(self, expected_packet, [eg_port])

            idle_remove_pkt = make_falcon_idle_remove_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=initial_idle_list[idle_pointer_stack], dst_id=spine_under_test_id, seq_num=0x10+i, **eth_kwargs)
            testutils.send_packet(self, ig_port, idle_remove_pkt)
            expected_idle_count -= 1
            idle_pointer_stack -= 1
            poll_res = testutils.dp_poll(self)
            print(poll_res)
            test_register_read(self.target,
                    self.register_idle_count,
                    'SpineIngress.idle_count.f1',
                    pipe_id,
                    TEST_VCLUSTER_ID,
                    expected_idle_count)


    def schedule_task_sq_state(self):
        pipe_id = 0

        initial_idle_count = 0
        initial_idle_list = []
        qlen_units = [0b00000001, 0b000000100]

        leaf_port_mapping = {11:1, 12: 2, 13:3, 14: 4, 15:5, 16: 6, 17:7, 18: 8, 100: 10}

        spine_under_test_id = 100
        intitial_qlen_state = [6, 1]
        initial_lid_list = [11, 12] # leaf IDs for the queue lengths

        initial_num_waiting_tasks = sum(intitial_qlen_state)

        intitial_deferred_state = [0, 0, 0, 0, 0]
        num_valid_ds_elements = 2 # num bits for available workers for this vcluster in this rack (the worker number will be 2^W)
        max_linked_leafs = 2

        workers_start_idx = TEST_VCLUSTER_ID * MAX_VCLUSTER_WORKERS

        num_tasks_to_send = 3

        register_write(self.target,
            self.register_idle_count,
            register_name='SpineIngress.idle_count.f1',
            index=TEST_VCLUSTER_ID,
            register_value=initial_idle_count)

        register_write(self.target,
            self.register_queue_signal_count,
            register_name='SpineIngress.queue_signal_count.f1',
            index=TEST_VCLUSTER_ID,
            register_value=int(math.log(len(intitial_qlen_state))/math.log(2)))

        for i, qlen in enumerate(intitial_qlen_state):
            register_write(self.target,
                    self.register_queue_len_list_1,
                    register_name='SpineIngress.queue_len_list_1.f1',
                    index=workers_start_idx + initial_lid_list[i],
                    register_value=qlen)
            register_write(self.target,
                    self.register_queue_len_list_2,
                    register_name='SpineIngress.queue_len_list_2.f1',
                    index=workers_start_idx + initial_lid_list[i],
                    register_value=qlen)

            # Initial state for deferred lists is 0
            register_write(self.target,
                    self.register_deferred_queue_len_list_1,
                    register_name='SpineIngress.deferred_queue_len_list_1.f1',
                    index=workers_start_idx + initial_lid_list[i],
                    register_value=intitial_deferred_state[i])
            register_write(self.target,
                    self.register_deferred_queue_len_list_2,
                    register_name='SpineIngress.deferred_queue_len_list_2.f1',
                    index=workers_start_idx + initial_lid_list[i],
                    register_value=intitial_deferred_state[i])
            
            # Initial leaf ID list for available queue signals
            register_write(self.target,
                self.register_lid_list_1,
                register_name='SpineIngress.lid_list_1.f1',
                index=workers_start_idx + i,
                register_value=initial_lid_list[i])
            register_write(self.target,
                self.register_lid_list_2,
                register_name='SpineIngress.lid_list_2.f1',
                index=workers_start_idx + i,
                register_value=initial_lid_list[i])
        
        logger.info("********* Populating Table Entires *********")
        # Insert qlen unit entires for each leaf in cluster
        for i, lid in enumerate(initial_lid_list):
            self.set_queue_len_unit_1.entry_add(
                self.target,
                [self.set_queue_len_unit_1.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID), client.KeyTuple('falcon_md.random_id_1', lid)])],
                [self.set_queue_len_unit_1.make_data([client.DataTuple('cluster_unit', qlen_units[i])],
                                             'SpineIngress.act_set_queue_len_unit_1')]
                )
            self.set_queue_len_unit_2.entry_add(
                self.target,
                [self.set_queue_len_unit_2.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID), client.KeyTuple('falcon_md.random_id_2', lid)])],
                [self.set_queue_len_unit_2.make_data([client.DataTuple('cluster_unit', qlen_units[i])],
                                             'SpineIngress.act_set_queue_len_unit_2')]
                )

        # Insert port mapping for workers
        for lid in leaf_port_mapping.keys():
            self.forward_falcon_switch_dst.entry_add(
                self.target,
                [self.forward_falcon_switch_dst.make_key([client.KeyTuple('hdr.falcon.dst_id', lid)])],
                [self.forward_falcon_switch_dst.make_data([client.DataTuple('port', leaf_port_mapping[lid])],
                                             'SpineIngress.act_forward_falcon')]
            )

        # Insert num_valid to set total num leafs for this vcluster
        self.get_cluster_num_valid_leafs.entry_add(
                self.target,
                [self.get_cluster_num_valid_leafs.make_key([client.KeyTuple('hdr.falcon.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_cluster_num_valid_leafs.make_data([client.DataTuple('num_leafs', num_valid_ds_elements), client.DataTuple('max_linked_leafs', max_linked_leafs)],
                                             'SpineIngress.act_get_cluster_num_valid_leafs')]
            )
        # Insert adjust random range tables
        self.adjust_random_range_all_leafs.entry_add(
                self.target,
                [self.adjust_random_range_all_leafs.make_key([client.KeyTuple('falcon_md.cluster_num_valid_ds', 1)])],
                [self.adjust_random_range_all_leafs.make_data([], 'SpineIngress.adjust_random_leaf_index_1')]
            )
        self.adjust_random_range_all_leafs.entry_add(
                self.target,
                [self.adjust_random_range_all_leafs.make_key([client.KeyTuple('falcon_md.cluster_num_valid_ds', 2)])],
                [self.adjust_random_range_all_leafs.make_data([], 'SpineIngress.adjust_random_leaf_index_2')]
            )
        self.adjust_random_range_all_leafs.entry_add(
                self.target,
                [self.adjust_random_range_all_leafs.make_key([client.KeyTuple('falcon_md.cluster_num_valid_ds', 4)])],
                [self.adjust_random_range_all_leafs.make_data([], 'SpineIngress.adjust_random_leaf_index_4')]
            )
        self.adjust_random_range_all_leafs.entry_add(
                self.target,
                [self.adjust_random_range_all_leafs.make_key([client.KeyTuple('falcon_md.cluster_num_valid_ds', 8)])],
                [self.adjust_random_range_all_leafs.make_data([], 'SpineIngress.adjust_random_leaf_index_8')]
            )
        
        self.adjust_random_range_sq_leafs.entry_add(
                self.target,
                [self.adjust_random_range_sq_leafs.make_key([client.KeyTuple('falcon_md.cluster_num_valid_queue_signals', 1)])],
                [self.adjust_random_range_sq_leafs.make_data([], 'SpineIngress.adjust_random_leaf_index_1')]
            )
        self.adjust_random_range_sq_leafs.entry_add(
                self.target,
                [self.adjust_random_range_sq_leafs.make_key([client.KeyTuple('falcon_md.cluster_num_valid_queue_signals', 2)])],
                [self.adjust_random_range_sq_leafs.make_data([], 'SpineIngress.adjust_random_leaf_index_2')]
            )
        self.adjust_random_range_sq_leafs.entry_add(
                self.target,
                [self.adjust_random_range_sq_leafs.make_key([client.KeyTuple('falcon_md.cluster_num_valid_queue_signals', 4)])],
                [self.adjust_random_range_sq_leafs.make_data([], 'SpineIngress.adjust_random_leaf_index_4')]
            )
        self.adjust_random_range_sq_leafs.entry_add(
                self.target,
                [self.adjust_random_range_sq_leafs.make_key([client.KeyTuple('falcon_md.cluster_num_valid_queue_signals', 8)])],
                [self.adjust_random_range_sq_leafs.make_data([], 'SpineIngress.adjust_random_leaf_index_8')]
            )

        logger.info("Inserted entries in tables")
        
        logger.info("\n********* Sending NEW_TASK packets, Testing scheduling on busy leafs *********\n")
        
        for i in range(num_tasks_to_send):
            test_register_read(self.target, #Idle count should stay at 0
                self.register_idle_count,
                'SpineIngress.idle_count.f1',
                pipe_id,
                TEST_VCLUSTER_ID,
                0)

            ig_port = swports[random.randint(9, 15)] # port connected to spine
            src_id = random.randint(1, 255)
            dst_id = spine_under_test_id

            new_task_packet = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=dst_id, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            expected_packet_0 = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=0, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            expected_packet_1 = make_falcon_task_pkt(dst_ip=SAMPLE_IP_DST, cluster_id=TEST_VCLUSTER_ID, local_cluster_id=TEST_VCLUSTER_ID, src_id=src_id, dst_id=1, pkt_len=0, seq_num=0x10+i, **eth_kwargs)
            logger.info("Sending task packet on port %d", ig_port)
            testutils.send_packet(self, ig_port, new_task_packet)
            poll_res = testutils.dp_poll(self)
            print(poll_res)
            for lid in initial_lid_list:
                test_register_read(self.target,
                    self.register_queue_len_list_1,
                    'SpineIngress.queue_len_list_1.f1',
                    pipe_id,
                    workers_start_idx + lid)
                test_register_read(self.target,
                    self.register_queue_len_list_2,
                    'SpineIngress.queue_len_list_2.f1',
                    pipe_id,
                    workers_start_idx + lid)
                test_register_read(self.target,
                    self.register_deferred_queue_len_list_1,
                    'SpineIngress.deferred_queue_len_list_1.f1',
                    pipe_id,
                    workers_start_idx + lid)
                test_register_read(self.target,
                    self.register_deferred_queue_len_list_2,
                    'SpineIngress.deferred_queue_len_list_2.f1',
                    pipe_id,
                    workers_start_idx + lid)
