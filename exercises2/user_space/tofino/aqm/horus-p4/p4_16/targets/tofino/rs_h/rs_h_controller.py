#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.expandvars('$SDE/install/lib/python2.7/site-packages/tofino/'))
import logging
import time
import grpc
import bfrt_grpc.bfruntime_pb2 as bfruntime_pb2
import bfrt_grpc.client as client
import random
import collections
import math

DEBUG_DUMP_REGS = True

TEST_VCLUSTER_ID = 0
MAX_VCLUSTER_WORKERS = 32
INVALID_VALUE_8bit = 0x7F
INVALID_VALUE_16bit = 0x7FFF

def register_write(target, register_object, register_name, index, register_value):
        print("Inserting entry in %s[%d] register with value = %s " %(str(register_name), index, str(register_value)))

        register_object.entry_add(
            target,
            [register_object.make_key([client.KeyTuple('$REGISTER_INDEX', index)])],
            [register_object.make_data([client.DataTuple(register_name, register_value)])])

def test_register_read(target, register_object, register_name, pipe_id, index):    
    resp = register_object.entry_get(
            target,
            [register_object.make_key([client.KeyTuple('$REGISTER_INDEX', index)])],
            {"from_hw": True})
    
    data_dict = next(resp)[0].to_dict()
    res = data_dict[register_name][0]
    print("Reading Register: %s [%d] = %d" %(str(register_name), index, res))
    return res

class LeafController():
    def __init__(self, target, bfrt_info, setup):
        self.target = target
        self.bfrt_info = bfrt_info
        self.tables = []
        self.setup = setup
        self.init_tables()
        self.init_data()
        
    def init_tables(self):
        bfrt_info = self.bfrt_info
        self.register_queue_len_list_1 = bfrt_info.table_get("LeafIngress.queue_len_list_1")
        self.register_queue_len_list_2 = bfrt_info.table_get("LeafIngress.queue_len_list_2")
        self.register_aggregate_queue_len = bfrt_info.table_get("LeafIngress.aggregate_queue_len_list")
        self.register_linked_sq_sched = bfrt_info.table_get("LeafIngress.linked_sq_sched")
        # MA Tables
        self.forward_horus_switch_dst = bfrt_info.table_get("LeafIngress.forward_horus_switch_dst")
        self.forward_horus_switch_dst.info.key_field_annotation_add("hdr.horus.dst_id", "wid")
        self.get_cluster_num_valid = bfrt_info.table_get("LeafIngress.get_cluster_num_valid")
        self.get_cluster_num_valid.info.key_field_annotation_add("hdr.horus.cluster_id", "vcid")
        self.adjust_random_range_ds = bfrt_info.table_get("LeafIngress.adjust_random_range_ds")
        self.adjust_random_range_ds.info.key_field_annotation_add("horus_md.cluster_num_valid_ds", "num_valid_ds")
        self.set_queue_len_unit = bfrt_info.table_get("LeafIngress.set_queue_len_unit")
        self.set_queue_len_unit.info.key_field_annotation_add("hdr.horus.cluster_id", "vcid")

        # HW config tables (Mirror and multicast)
        self.mirror_cfg_table = bfrt_info.table_get("$mirror.cfg")

    def init_data(self):
        self.pipe_id = 0
        self.NUM_LEAVES = 4 # Number of virtual leafs 
        
        self.MAX_VCLUSTER_WORKERS = MAX_VCLUSTER_WORKERS # This number should be the same in p4 code (fixed at compile time)
        if (self.setup == 's'):
            self.initial_node_list = [[0, 1, 2, 3],
                    [0, 1, 2, 3],  
                    [0, 1, 2, 3, 4, 5, 6, 7], 
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32],
                    ]
            self.wid_port_mapping = [
                        {0:132, 1:132, 2:132, 3:132},
                        {0:132, 1:132, 2:132, 3:132},
                        {0:134, 1:134, 2:134, 3:134, 4:134, 5:134, 6:134, 7:134},
                        {0:150, 1:150, 2:150, 3:150, 4:150, 5:150, 6:150, 7:150, 8:150, 9:150, 10:150, 11:150, 12:150, 13:150, 14:150, 15:150, 16:140, 17:140, 18:140, 19:140, 20:140, 21:140, 22:140, 23:140, 24:142, 25:142, 26:142, 27:142, 28:142, 29:142, 30:142, 31:142},
                        ]
        elif (self.setup== 'b'):
            self.initial_node_list = [[0, 1, 2, 3, 4, 5, 6, 7],[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
            self.wid_port_mapping = [
                        {0:132, 1:132, 2:132, 3:132, 4:132, 5:132, 6:132, 7:132},
                        {0:134, 1:134, 2:134, 3:134, 4:134, 5:134, 6:134, 7:134},
                        {0:140, 1:140, 2:140, 3:140, 4:140, 5:140, 6:140, 7:140},
                        {0:142, 1:142, 2:142, 3:142, 4:142, 5:142, 6:142, 7:142},
                        ]
        elif (self.setup == 'r'):
            self.NUM_LEAVES = 1
            self.initial_node_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
            self.wid_port_mapping = [{0:132, 1:132, 2:132, 3:132, 4:132, 5:132, 6:132, 7:132,
                                     8:134, 9:134, 10:134, 11:134, 12:134, 13:134, 14:134, 15:134,
                                     16:140, 17:140, 18:140, 19:140, 20:140, 21:140, 22:140, 23:140,
                                     24:142, 25:142, 26:142, 27:142, 28:142, 29:142, 30:142, 31:142,
                                     }]

        self.intitial_qlen_state = [0] * (self.MAX_VCLUSTER_WORKERS) # All zeros in the begining
        
        self.initial_agg_qlen = 0
        self.qlen_unit = [] # assuming 3bit showing fraction and 5bit decimal
        self.num_valid_ds_elements = [] # num available workers for this vcluster in this rack (the worker number will be 2^W)
    
        for rack_node_list in self.initial_node_list:
            if len(rack_node_list) == 4:
                self.qlen_unit.append(8)
            elif len(rack_node_list) == 8:
                self.qlen_unit.append(4)
            elif len(rack_node_list) == 32:
                self.qlen_unit.append(1)
            self.num_valid_ds_elements.append(len(rack_node_list))

        print("Num workers per rack:")
        print(self.num_valid_ds_elements)
        print("Unit for aggregate qlen report: ")
        print(self.qlen_unit)

        self.initial_linked_sq_spine = 100 # ID of linked spine for SQ link
        
        if (self.setup != 'r'):
            self.spine_port_mapping = {100: 152, 110:144, 111:160, 150:144}
        else: 
            self.spine_port_mapping = {110:56, 150:56} 

        self.port_mac_mapping = {132: 'F8:F2:1E:3A:13:EC', 134: 'F8:F2:1E:3A:13:0C',  140: 'F8:F2:1E:3A:13:C4', 142:'F8:F2:1E:3A:07:24', 150:' F8:F2:1E:13:CA:FC',
                    152: '40:A6:B7:3C:45:64', 160:'40:A6:B7:3C:24:C8', 144: 'F8:F2:1E:3A:13:C4', 56: 'F8:F2:1E:3A:13:C4'}
        
        self.num_valid_us_elements = 2
        self.workers_start_idx = []
        for leaf_id in range(self.NUM_LEAVES):
            self.workers_start_idx.append(leaf_id * self.MAX_VCLUSTER_WORKERS)

    def set_tables(self):
        for leaf_id in range(self.NUM_LEAVES):
            print(" *********  Virtual Leaf: " + str(leaf_id) + "  *********  ")
            print("********* Inserting initial Reg Entires *********")
            for i, qlen in enumerate(self.intitial_qlen_state):
                register_write(self.target,
                        self.register_queue_len_list_1,
                        register_name='LeafIngress.queue_len_list_1.f1',
                        index=self.workers_start_idx[leaf_id] + i,
                        register_value=qlen)
                register_write(self.target,
                        self.register_queue_len_list_2,
                        register_name='LeafIngress.queue_len_list_2.f1',
                        index=self.workers_start_idx[leaf_id] + i,
                        register_value=qlen)
            register_write(self.target,
                        self.register_aggregate_queue_len,
                        register_name='LeafIngress.aggregate_queue_len_list.f1',
                        index=leaf_id,
                        register_value=self.initial_agg_qlen)
            register_write(self.target,
                    self.register_linked_sq_sched,
                    register_name='LeafIngress.linked_sq_sched.f1',
                    index=leaf_id,
                    register_value=self.initial_linked_sq_spine)

            print("********* Populating Table Entires *********")
            self.set_queue_len_unit.entry_add(
                    self.target,
                    [self.set_queue_len_unit.make_key([client.KeyTuple('hdr.horus.cluster_id', leaf_id)])],
                    [self.set_queue_len_unit.make_data([client.DataTuple('cluster_unit', self.qlen_unit[leaf_id])],
                                                 'LeafIngress.act_set_queue_len_unit')]
                )
            for wid in self.wid_port_mapping[leaf_id].keys():
                self.forward_horus_switch_dst.entry_add(
                    self.target,
                    [self.forward_horus_switch_dst.make_key([client.KeyTuple('hdr.horus.dst_id', self.workers_start_idx[leaf_id] + wid)])],
                    [self.forward_horus_switch_dst.make_data([client.DataTuple('port', self.wid_port_mapping[leaf_id][wid]), client.DataTuple('dst_mac', client.mac_to_bytes(self.port_mac_mapping[self.wid_port_mapping[leaf_id][wid]]))],
                                                 'LeafIngress.act_forward_horus')]
                )
            #print("Inserted entries in forward_horus_switch_dst table with key-values = ", str(self.wid_port_mapping[leaf_id]))
            
            self.get_cluster_num_valid.entry_add(
                self.target,
                [self.get_cluster_num_valid.make_key([client.KeyTuple('hdr.horus.cluster_id', leaf_id)])],
                [self.get_cluster_num_valid.make_data([client.DataTuple('num_ds_elements', self.num_valid_ds_elements[leaf_id])],
                                             'LeafIngress.act_get_cluster_num_valid')]
            )

        for sid in self.spine_port_mapping.keys():
            self.forward_horus_switch_dst.entry_add(
                self.target,
                [self.forward_horus_switch_dst.make_key([client.KeyTuple('hdr.horus.dst_id', sid)])],
                [self.forward_horus_switch_dst.make_data([client.DataTuple('port', self.spine_port_mapping[sid]), client.DataTuple('dst_mac', client.mac_to_bytes(self.port_mac_mapping[self.spine_port_mapping[sid]]))],
                                             'LeafIngress.act_forward_horus')]
            )
            mirror_cfg_bfrt_key  = self.mirror_cfg_table.make_key([client.KeyTuple('$sid', sid)])
            mirror_cfg_bfrt_data = self.mirror_cfg_table.make_data([
                client.DataTuple('$direction', str_val="INGRESS"),
                client.DataTuple('$ucast_egress_port', self.spine_port_mapping[sid]),
                client.DataTuple('$ucast_egress_port_valid', bool_val=True),
                client.DataTuple('$session_enable', bool_val=True),
            ], "$normal")
            self.mirror_cfg_table.entry_add(self.target, [ mirror_cfg_bfrt_key ], [ mirror_cfg_bfrt_data ])

        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_ds', 2)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_1')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_ds', 4)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_2')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_ds', 8)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_3')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_ds', 16)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_4')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_ds', 32)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_5')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_ds', 256)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_8')]
            )

    def read_reg_stats(self):
        for k in range(self.NUM_LEAVES):
            for i in range(len(self.initial_node_list[k])):
                test_register_read(self.target,
                    self.register_queue_len_list_1,
                    'LeafIngress.queue_len_list_1.f1',
                    self.pipe_id,
                    self.workers_start_idx[k] + i)
                test_register_read(self.target,
                    self.register_queue_len_list_2,
                    'LeafIngress.queue_len_list_2.f1',
                    self.pipe_id,
                    self.workers_start_idx[k] + i)

class SpineController():
    def __init__(self, target, bfrt_info, setup):
        self.target = target
        self.bfrt_info = bfrt_info
        self.tables = []
        self.setup = setup
        self.init_tables()
        self.init_data()

    def init_tables(self):
        bfrt_info = self.bfrt_info
        self.register_queue_len_list_1 = bfrt_info.table_get("SpineIngress.queue_len_list_1")
        self.register_queue_len_list_2 = bfrt_info.table_get("SpineIngress.queue_len_list_2")
        # MA Tables
        self.forward_horus_switch_dst = bfrt_info.table_get("SpineIngress.forward_horus_switch_dst")
        self.forward_horus_switch_dst.info.key_field_annotation_add("hdr.horus.dst_id", "id")
        self.get_cluster_num_valid = bfrt_info.table_get("SpineIngress.get_cluster_num_valid_leafs")
        self.get_cluster_num_valid.info.key_field_annotation_add("hdr.horus.cluster_id", "vcid")
        self.adjust_random_range_ds = bfrt_info.table_get("SpineIngress.adjust_random_range_sq_leafs")
        self.adjust_random_range_ds.info.key_field_annotation_add("horus_md.cluster_num_valid_queue_signals", "num_valid_ds")
        self.get_rand_leaf_id_1 = bfrt_info.table_get("SpineIngress.get_rand_leaf_id_1")
        self.get_rand_leaf_id_1.info.key_field_annotation_add("horus_md.random_ds_index_1", "rand_idx_1")
        self.get_rand_leaf_id_2 = bfrt_info.table_get("SpineIngress.get_rand_leaf_id_2")
        self.get_rand_leaf_id_2.info.key_field_annotation_add("horus_md.random_ds_index_2", "rand_idx_2")

    def init_data(self):
        self.pipe_id = 0
        self.TEST_VCLUSTER_ID = 0
        self.MAX_VCLUSTER_LEAVES = 16 # This number is per cluster. *Important: should be the same in p4 code (fixed at compile time)
        self.initial_node_list = [0, 1, 2, 3]
        self.intitial_qlen_state = [0] * len(self.initial_node_list)
        self.wid_port_mapping = {0:36, 1:44, 2:20, 3:52, 4:28, 5:20, 110:56, 111: 58, 100:28}
        
        if self.setup == 'b':
            self.qlen_unit = [4] * len(self.initial_node_list) # assuming 3bit showing fraction and 5bit decimal: 0.125 (8 workers in each rack)
        elif self.setup == 's':
            self.qlen_unit = [8, 8, 4, 1] # proportional to 1/workers in the rack
        
        self.num_valid_ds_elements = len(self.initial_node_list) # num available leaves this vcluster (the number in hardware will be 2^W)
        self.leaf_start_idx = self.TEST_VCLUSTER_ID * self.MAX_VCLUSTER_LEAVES
    
    def set_tables(self):
        # Table entries
        print("********* Populating Table Entires *********")
        for wid in self.wid_port_mapping.keys():
            self.forward_horus_switch_dst.entry_add(
                self.target,
                [self.forward_horus_switch_dst.make_key([client.KeyTuple('hdr.horus.dst_id', wid)])],
                [self.forward_horus_switch_dst.make_data([client.DataTuple('port', self.wid_port_mapping[wid])],
                                             'SpineIngress.act_forward_horus')]
            )
        
        for idx, leaf_id in enumerate(self.initial_node_list):
            self.get_rand_leaf_id_1.entry_add(
                self.target,
                [self.get_rand_leaf_id_1.make_key([client.KeyTuple('horus_md.random_ds_index_1', idx), client.KeyTuple('hdr.horus.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_rand_leaf_id_1.make_data([client.DataTuple('leaf_id', leaf_id)],
                                             'SpineIngress.act_get_rand_leaf_id_1')]
            )
            self.get_rand_leaf_id_2.entry_add(
                self.target,
                [self.get_rand_leaf_id_2.make_key([client.KeyTuple('horus_md.random_ds_index_2', idx), client.KeyTuple('hdr.horus.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_rand_leaf_id_2.make_data([client.DataTuple('leaf_id', leaf_id)],
                                             'SpineIngress.act_get_rand_leaf_id_2')]
            )
            
        self.get_cluster_num_valid.entry_add(
                self.target,
                [self.get_cluster_num_valid.make_key([client.KeyTuple('hdr.horus.cluster_id', self.TEST_VCLUSTER_ID)])],
                [self.get_cluster_num_valid.make_data([client.DataTuple('num_leafs', self.num_valid_ds_elements)],
                                             'SpineIngress.act_get_cluster_num_valid_leafs')]
            )

        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 2)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_1')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 4)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_2')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 5)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_2')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 16)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_4')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 256)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_8')]
        )

    def read_reg_stats(self):
        for i in range(self.NUM_LEAVES):
            test_register_read(self.target,
                self.register_queue_len_list_1,
                'SpineIngress.queue_len_list_1.f1',
                self.pipe_id,
                i)
            test_register_read(self.target,
                self.register_queue_len_list_2,
                'SpineIngress.queue_len_list_2.f1',
                self.pipe_id,
                i)

class RandomSpineController():
    def __init__(self, target, bfrt_info):
        self.target = target
        self.bfrt_info = bfrt_info
        self.tables = []
        self.init_tables()
        self.init_data()

    def init_tables(self):
        bfrt_info = self.bfrt_info
        # MA Tables
        self.forward_horus_switch_dst = bfrt_info.table_get("SpineIngress.forward_horus_switch_dst")
        self.forward_horus_switch_dst.info.key_field_annotation_add("hdr.horus.dst_id", "id")
        self.get_cluster_num_valid = bfrt_info.table_get("SpineIngress.get_cluster_num_valid_leafs")
        self.get_cluster_num_valid.info.key_field_annotation_add("hdr.horus.cluster_id", "vcid")
        self.adjust_random_range_ds = bfrt_info.table_get("SpineIngress.adjust_random_range_sq_leafs")
        self.adjust_random_range_ds.info.key_field_annotation_add("horus_md.cluster_num_valid_queue_signals", "num_valid_ds")
        self.get_rand_leaf_id_1 = bfrt_info.table_get("SpineIngress.get_rand_leaf_id_1")
        self.get_rand_leaf_id_1.info.key_field_annotation_add("horus_md.random_ds_index_1", "rand_idx_1")
        
    def init_data(self):
        self.pipe_id = 0
        self.TEST_VCLUSTER_ID = 0
        self.MAX_VCLUSTER_LEAVES = 16 # This number is per cluster. *Important: should be the same in p4 code (fixed at compile time)
        self.initial_idle_list = [0, 1, 2, 3]
        self.wid_port_mapping = {0:36, 1:44, 2:20, 3:52, 4:28, 5:20, 110:56, 111: 58, 100:28}

        self.num_valid_ds_elements = len(self.initial_idle_list) # num available leaves this vcluster (the number in hardware will be 2^W)

        self.leaf_start_idx = self.TEST_VCLUSTER_ID * self.MAX_VCLUSTER_LEAVES
    
    def set_tables(self):
        # Table entries
        print("********* Populating Table Entires *********")
        for wid in self.wid_port_mapping.keys():
            self.forward_horus_switch_dst.entry_add(
                self.target,
                [self.forward_horus_switch_dst.make_key([client.KeyTuple('hdr.horus.dst_id', wid)])],
                [self.forward_horus_switch_dst.make_data([client.DataTuple('port', self.wid_port_mapping[wid])],
                                             'SpineIngress.act_forward_horus')]
            )
        
        for idx, leaf_id in enumerate(self.initial_idle_list):
            self.get_rand_leaf_id_1.entry_add(
                self.target,
                [self.get_rand_leaf_id_1.make_key([client.KeyTuple('horus_md.random_ds_index_1', idx), client.KeyTuple('hdr.horus.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_rand_leaf_id_1.make_data([client.DataTuple('leaf_id', leaf_id)],
                                             'SpineIngress.act_get_rand_leaf_id_1')]
            )
            
        self.get_cluster_num_valid.entry_add(
                self.target,
                [self.get_cluster_num_valid.make_key([client.KeyTuple('hdr.horus.cluster_id', self.TEST_VCLUSTER_ID)])],
                [self.get_cluster_num_valid.make_data([client.DataTuple('num_leafs', self.num_valid_ds_elements)],
                                             'SpineIngress.act_get_cluster_num_valid_leafs')]
            )

        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 2)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_1')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 4)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_2')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 5)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_2')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 16)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_4')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('horus_md.cluster_num_valid_queue_signals', 256)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_8')]
        )

if __name__ == "__main__":
    setup = str(sys.argv[1])
    if setup == 'b':
        print("*****Using the balanced worker placement*****")
        print("**Four racks with eight workers each**")
    elif setup == 's':
        print("*****Using the skewed worker placement*****")
        print("**Two racks with four workers, one rack with eight workers, one rack with 32 workers**")
    elif setup == 'r':
        print("*****Using one rack worker placement*****")
        print("**One racks with 32 workers**")
    else:
        print("Argument required for placement setup: use \"s\"(skewed) or \"b\"(balanced)")
        exit(1)

    # Connect to BF Runtime Server
    interface = client.ClientInterface(grpc_addr = "localhost:50052",
                                    client_id = 0,
                                    device_id = 0)
    print("Connected to BF Runtime Server")
    
    # Get the information about the running program on the bfrt server.
    bfrt_info = interface.bfrt_info_get()
    print('The target runs program ', bfrt_info.p4_name_get())

    # Establish that you are working with this program
    interface.bind_pipeline_config(bfrt_info.p4_name_get())

    ####### You can now use BFRT CLIENT #######
    target = client.Target(device_id=0, pipe_id=0xffff)
    if setup == 'r':
        spine_controller = RandomSpineController(target, bfrt_info)
        spine_controller.set_tables()
    else:
        spine_controller = SpineController(target, bfrt_info, setup=setup)
        spine_controller.set_tables()

    leaf_controller = LeafController(target, bfrt_info, setup=setup)
    leaf_controller.set_tables()
    while (True):
        print ("\n")
        #spine_controller.read_reg_stats()
        print ("\n")
        leaf_controller.read_reg_stats()
        print("\n")
        print("#################")
        print("\n")
        time.sleep(1)


