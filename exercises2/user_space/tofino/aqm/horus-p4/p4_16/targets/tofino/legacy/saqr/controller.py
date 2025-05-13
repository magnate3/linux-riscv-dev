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
import numpy as np

DEBUG_DUMP_REGS = False

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

def test_register_read(target, register_object, register_name, pipe_id, index, print_reg=True):    
    resp = register_object.entry_get(
            target,
            [register_object.make_key([client.KeyTuple('$REGISTER_INDEX', index)])],
            {"from_hw": True})
    
    data_dict = next(resp)[0].to_dict()
    res = data_dict[register_name][0]
    if print_reg:
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
        self.register_idle_count = bfrt_info.table_get("LeafIngress.idle_count")
        self.register_idle_list = bfrt_info.table_get("LeafIngress.idle_list")
        self.register_aggregate_queue_len = bfrt_info.table_get("LeafIngress.aggregate_queue_len_list")
        self.register_linked_sq_sched = bfrt_info.table_get("LeafIngress.linked_sq_sched")
        self.register_linked_iq_sched = bfrt_info.table_get("LeafIngress.linked_iq_sched")
        self.register_queue_len_list_1 = bfrt_info.table_get("LeafIngress.queue_len_list_1")
        self.register_queue_len_list_2 = bfrt_info.table_get("LeafIngress.queue_len_list_2")
        self.register_deferred_list_1 = bfrt_info.table_get("LeafIngress.deferred_queue_len_list_1")
        self.register_deferred_list_2 = bfrt_info.table_get("LeafIngress.deferred_queue_len_list_2")
        self.register_stat_count_resub = bfrt_info.table_get("LeafIngress.stat_count_resub")
        self.register_stat_count_idle_signal = bfrt_info.table_get("LeafIngress.stat_count_idle_signal")
        self.register_stat_count_load_signal = bfrt_info.table_get("LeafIngress.stat_count_load_signal")
        self.register_stat_count_task = bfrt_info.table_get("LeafIngress.stat_count_task")
        self.register_ingress_tstamp = bfrt_info.table_get("LeafIngress.ingress_tstamp")
        self.register_egress_tstamp = bfrt_info.table_get("SaqrEgress.egress_tstamp")

        # MA Tables
        self.forward_saqr_switch_dst = bfrt_info.table_get("LeafIngress.forward_saqr_switch_dst")
        self.forward_saqr_switch_dst.info.key_field_annotation_add("hdr.saqr.dst_id", "wid")
        self.set_queue_len_unit = bfrt_info.table_get("LeafIngress.set_queue_len_unit")
        self.set_queue_len_unit.info.key_field_annotation_add("hdr.saqr.cluster_id", "vcid")
        self.get_cluster_num_valid = bfrt_info.table_get("LeafIngress.get_cluster_num_valid")
        self.get_cluster_num_valid.info.key_field_annotation_add("hdr.saqr.cluster_id", "vcid")
        self.adjust_random_range_ds = bfrt_info.table_get("LeafIngress.adjust_random_range_ds")
        self.adjust_random_range_ds.info.key_field_annotation_add("saqr_md.cluster_num_valid_ds", "num_valid_ds")
        self.adjust_random_range_us = bfrt_info.table_get("LeafIngress.adjust_random_range_us")
        self.adjust_random_range_us.info.key_field_annotation_add("saqr_md.cluster_num_valid_us", "num_valid_us")
        self.get_spine_dst_id = bfrt_info.table_get("LeafIngress.get_spine_dst_id")
        self.get_spine_dst_id.info.key_field_annotation_add("saqr_md.random_id_1", "random_id")

        # HW config tables (Mirror and multicast)
        self.mirror_cfg_table = bfrt_info.table_get("$mirror.cfg")

    def init_data(self):
        self.pipe_id = 0
        self.NUM_LEAVES = 4 # Number of virtual leafs 
        
        self.MAX_VCLUSTER_WORKERS = 16 # This number should be the same in p4 code (fixed at compile time)

        if (self.setup == 's'):
            self.initial_idle_list = [[0, 1, 2, 3],
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
            self.initial_idle_list = [[0, 1, 2, 3, 4, 5, 6, 7],[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
            self.wid_port_mapping = [
                        {0:132, 1:132, 2:132, 3:132, 4:132, 5:132, 6:132, 7:132},
                        {0:150, 1:150, 2:150, 3:150, 4:150, 5:150, 6:150, 7:150},
                        {0:140, 1:140, 2:140, 3:140, 4:140, 5:140, 6:140, 7:140},
                        {0:142, 1:142, 2:142, 3:142, 4:142, 5:142, 6:142, 7:142},
                        ]

        self.CPU_PORT_ID = 192
        self.initial_idle_count = []
        for rack_idle_list in self.initial_idle_list:
            self.initial_idle_count.append(len(rack_idle_list))

        self.intitial_qlen_state = [0] * (self.MAX_VCLUSTER_WORKERS) # All zeros in the begining
        self.intitial_deferred_state = [0] * (self.MAX_VCLUSTER_WORKERS) # All zeros in the begining

        self.initial_agg_qlen = 0
        self.qlen_unit = [] # assuming 3bit showing fraction and 5bit decimal
        self.num_valid_ds_elements = [] # num available workers for this vcluster in this rack (the worker number will be 2^W)
        for rack_idle_list in self.initial_idle_list:
            if len(rack_idle_list) == 4:
                self.qlen_unit.append(8)
            elif len(rack_idle_list) == 8:
                self.qlen_unit.append(4)
            elif len(rack_idle_list) == 32:
                self.qlen_unit.append(1)
            self.num_valid_ds_elements.append(len(rack_idle_list))

        print("Num workers per rack:")
        print(self.num_valid_ds_elements)
        print("Unit for aggregate qlen report: ")
        print(self.qlen_unit)

        self.initial_linked_iq_spine = 100 # ID of linked spine for Idle link
        self.initial_linked_sq_spine = 100 # ID of linked spine for SQ link (Invalid = 0xFFFF, since we are in idle state)
        
        
        

        self.spine_port_mapping = {100: 152, 110:144, 111:160} 
        self.port_mac_mapping = {132: 'F8:F2:1E:3A:13:EC', 134: 'F8:F2:1E:3A:13:0C',  140: 'F8:F2:1E:3A:13:C4', 142:'F8:F2:1E:3A:07:24', 150:' F8:F2:1E:13:CA:FC',
                    152: '40:A6:B7:3C:45:64', 160:'40:A6:B7:3C:24:C8', 144: 'F8:F2:1E:3A:13:C4', self.CPU_PORT_ID: '00:02:00:00:03:00'}
        
        self.num_valid_us_elements = 2
        self.workers_start_idx = []
        for leaf_id in range(self.NUM_LEAVES):
            self.workers_start_idx.append(leaf_id * self.MAX_VCLUSTER_WORKERS)

    def set_tables(self):
        for leaf_id in range(self.NUM_LEAVES):
            print(" *********  Virtual Leaf: " + str(leaf_id) + "  *********  ")
            print("********* Inserting initial Reg Entires *********")
            
            register_write(self.target,
                    self.register_linked_iq_sched,
                    register_name='LeafIngress.linked_iq_sched.f1',
                    index=leaf_id,
                    register_value=self.initial_linked_iq_spine)

            register_write(self.target,
                    self.register_linked_sq_sched,
                    register_name='LeafIngress.linked_sq_sched.f1',
                    index=leaf_id,
                    register_value=self.initial_linked_sq_spine)

            register_write(self.target,
                self.register_idle_count,
                register_name='LeafIngress.idle_count.f1',
                index=leaf_id,
                register_value=self.initial_idle_count[leaf_id])
        
            register_write(self.target,
                    self.register_aggregate_queue_len,
                    register_name='LeafIngress.aggregate_queue_len_list.f1',
                    index=leaf_id,
                    register_value=self.initial_agg_qlen)

            test_register_read(self.target,
                self.register_idle_count,
                'LeafIngress.idle_count.f1',
                self.pipe_id,
                leaf_id)

            # Insert idle_list values (wid of idle workers)
            for i in range(self.initial_idle_count[leaf_id]):
                register_write(self.target,
                    self.register_idle_list,
                    register_name='LeafIngress.idle_list.f1',
                    index= self.workers_start_idx[leaf_id] + i,
                    register_value=self.workers_start_idx[leaf_id] + self.initial_idle_list[leaf_id][i])

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
                        self.register_deferred_list_1,
                        register_name='LeafIngress.deferred_queue_len_list_1.f1',
                        index=self.workers_start_idx[leaf_id] + i,
                        register_value=self.intitial_deferred_state[i])

            for i in range(self.num_valid_us_elements): # TODO: Make number of spines dynamic
                self.get_spine_dst_id.entry_add(
                    self.target,
                    [self.get_spine_dst_id.make_key([client.KeyTuple('saqr_md.random_id_1', i), client.KeyTuple('hdr.saqr.cluster_id', leaf_id)])],
                    [self.get_spine_dst_id.make_data([client.DataTuple('spine_dst_id', 100)],
                                                 'LeafIngress.act_get_spine_dst_id')]
                )

            print("********* Populating Table Entires *********")
            for wid in self.wid_port_mapping[leaf_id].keys():
                self.forward_saqr_switch_dst.entry_add(
                    self.target,
                    [self.forward_saqr_switch_dst.make_key([client.KeyTuple('hdr.saqr.dst_id', self.workers_start_idx[leaf_id] + wid)])],
                    [self.forward_saqr_switch_dst.make_data([client.DataTuple('port', self.wid_port_mapping[leaf_id][wid]), client.DataTuple('dst_mac', client.mac_to_bytes(self.port_mac_mapping[self.wid_port_mapping[leaf_id][wid]]))],
                                                 'LeafIngress.act_forward_saqr')]
                )
                #print("Inserted entries in forward_saqr_switch_dst table with key-values = " + str(self.wid_port_mapping[leaf_id]))
            
            
            self.get_cluster_num_valid.entry_add(
                self.target,
                [self.get_cluster_num_valid.make_key([client.KeyTuple('hdr.saqr.cluster_id', leaf_id)])],
                [self.get_cluster_num_valid.make_data([client.DataTuple('num_ds_elements', self.num_valid_ds_elements[leaf_id]), client.DataTuple('num_us_elements', self.num_valid_us_elements)],
                                             'LeafIngress.act_get_cluster_num_valid')]
            )
            # Insert qlen unit entries
            self.set_queue_len_unit.entry_add(
                    self.target,
                    [self.set_queue_len_unit.make_key([client.KeyTuple('hdr.saqr.cluster_id', leaf_id)])],
                    [self.set_queue_len_unit.make_data([client.DataTuple('cluster_unit', self.qlen_unit[leaf_id])],
                                                 'LeafIngress.act_set_queue_len_unit')]
                )

        self.forward_saqr_switch_dst.entry_add(
                self.target,
                [self.forward_saqr_switch_dst.make_key([client.KeyTuple('hdr.saqr.dst_id', self.CPU_PORT_ID)])],
                [self.forward_saqr_switch_dst.make_data([client.DataTuple('port', self.CPU_PORT_ID), client.DataTuple('dst_mac', client.mac_to_bytes(self.port_mac_mapping[self.CPU_PORT_ID]))],
                                             'LeafIngress.act_forward_saqr')]
            )

        for sid in self.spine_port_mapping.keys():
            self.forward_saqr_switch_dst.entry_add(
                self.target,
                [self.forward_saqr_switch_dst.make_key([client.KeyTuple('hdr.saqr.dst_id', sid)])],
                [self.forward_saqr_switch_dst.make_data([client.DataTuple('port', self.spine_port_mapping[sid]), client.DataTuple('dst_mac', client.mac_to_bytes(self.port_mac_mapping[self.spine_port_mapping[sid]]))],
                                             'LeafIngress.act_forward_saqr')]
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
                [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_ds', 2)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_1')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_ds', 4)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_2')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_ds', 8)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_3')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_ds', 16)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_4')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_ds', 32)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_5')]
            )
        self.adjust_random_range_ds.entry_add(
                self.target,
                [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_ds', 256)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_8')]
            )

        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('saqr_md.cluster_num_valid_us', 2)])],
                [self.adjust_random_range_ds.make_data([], 'LeafIngress.adjust_random_worker_range_1')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('saqr_md.cluster_num_valid_us', 4)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_2')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('saqr_md.cluster_num_valid_us', 8)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_3')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('saqr_md.cluster_num_valid_us', 16)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_4')]
            )
        self.adjust_random_range_us.entry_add(
                self.target,
                [self.adjust_random_range_us.make_key([client.KeyTuple('saqr_md.cluster_num_valid_us', 256)])],
                [self.adjust_random_range_us.make_data([], 'LeafIngress.adjust_random_worker_range_8')]
            )

    def read_reg_stats(self):
        if DEBUG_DUMP_REGS:
            for k in range(4):
                for i in range(len(self.initial_idle_list[k])):
                    test_register_read(self.target,
                        self.register_idle_list,
                        'LeafIngress.idle_list.f1',
                        self.pipe_id,
                        self.workers_start_idx[k] + i)
            for k in range(4):
                for i in range(len(self.initial_idle_list[k])):
                    test_register_read(self.target,
                        self.register_queue_len_list_1,
                        'LeafIngress.queue_len_list_1.f1',
                        self.pipe_id,
                        self.workers_start_idx[k] + i)
                    test_register_read(self.target,
                        self.register_deferred_list_1,
                        'LeafIngress.deferred_queue_len_list_1.f1',
                        self.pipe_id,
                        self.workers_start_idx[k] + i)
        total_resub_leaf = 0
        total_msg_load = 0
        total_msg_idle = 0
        for i in range(len(self.initial_idle_list)):
            if DEBUG_DUMP_REGS:
                test_register_read(self.target,
                    self.register_idle_count,
                    'LeafIngress.idle_count.f1',
                    self.pipe_id,
                    i)
            total_resub_leaf += test_register_read(self.target,
                self.register_stat_count_resub,
                'LeafIngress.stat_count_resub.f1',
                self.pipe_id,
                i)
            total_msg_load += test_register_read(self.target,
                self.register_stat_count_load_signal,
                'LeafIngress.stat_count_load_signal.f1',
                self.pipe_id,
                i)
            total_msg_idle += test_register_read(self.target,
                self.register_stat_count_idle_signal,
                'LeafIngress.stat_count_idle_signal.f1',
                self.pipe_id,
                i)
        
        task_tot = test_register_read(self.target,
            self.register_stat_count_task,
            'LeafIngress.stat_count_task.f1',
            self.pipe_id,
            0)
        print ("Total tasks arrived at Leaf: %d" %(task_tot))
        delay_list = []
        if task_tot >= 10000:
            for i in range(10000):
                ingress_tstamp = test_register_read(self.target,
                self.register_ingress_tstamp,
                'LeafIngress.ingress_tstamp.f1',
                self.pipe_id,
                i, 
                print_reg=False)
                egress_tstamp = test_register_read(self.target,
                self.register_egress_tstamp,
                'SaqrEgress.egress_tstamp.f1',
                self.pipe_id,
                i,
                print_reg=False)
                delay_list.append(egress_tstamp - ingress_tstamp)
            print (delay_list)
            np.savetxt('leaf_latency.csv', [np.array(delay_list)], delimiter=', ', fmt='%d')

        print ("Leaf Total Resubmission: %d" %(total_resub_leaf))
        print ("Total Msgs for Load Signals: %d" %(total_msg_load))
        print ("Total Msgs for Idle Signals: %d" %(total_msg_idle))
        print ("Sum Total State Update Msgs: %d" %(total_msg_load+total_msg_idle))
        if task_tot >= 10000:
            exit(0)
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
        self.register_idle_count = bfrt_info.table_get("SpineIngress.idle_count")
        self.register_idle_list = bfrt_info.table_get("SpineIngress.idle_list")
        self.register_queue_len_list_1 = bfrt_info.table_get("SpineIngress.queue_len_list_1")
        self.register_queue_len_list_2 = bfrt_info.table_get("SpineIngress.queue_len_list_2")
        self.register_deferred_list_1 = bfrt_info.table_get("SpineIngress.deferred_queue_len_list_1")
        self.register_deferred_list_2 = bfrt_info.table_get("SpineIngress.deferred_queue_len_list_2")
        self.register_idle_list_idx_mapping = bfrt_info.table_get("SpineIngress.idle_list_idx_mapping")
        self.register_stat_count_resub = bfrt_info.table_get("SpineIngress.stat_count_resub")
        self.register_stat_count_task = bfrt_info.table_get("SpineIngress.stat_count_task")
        self.register_ingress_tstamp = bfrt_info.table_get("SpineIngress.ingress_tstamp")
        self.register_egress_tstamp = bfrt_info.table_get("SpineEgress.egress_tstamp")
        
        # # MA Tables
        self.forward_saqr_switch_dst = bfrt_info.table_get("SpineIngress.forward_saqr_switch_dst")
        self.forward_saqr_switch_dst.info.key_field_annotation_add("hdr.saqr.dst_id", "id")
        self.set_queue_len_unit_1 = bfrt_info.table_get("SpineIngress.set_queue_len_unit_1")
        self.set_queue_len_unit_1.info.key_field_annotation_add("hdr.saqr.cluster_id", "vcid")
        self.set_queue_len_unit_2 = bfrt_info.table_get("SpineIngress.set_queue_len_unit_2")
        self.set_queue_len_unit_2.info.key_field_annotation_add("hdr.saqr.cluster_id", "vcid")
        self.get_cluster_num_valid = bfrt_info.table_get("SpineIngress.get_cluster_num_valid_leafs")
        self.get_cluster_num_valid.info.key_field_annotation_add("hdr.saqr.cluster_id", "vcid")
        self.adjust_random_range_ds = bfrt_info.table_get("SpineIngress.adjust_random_range_sq_leafs")
        self.adjust_random_range_ds.info.key_field_annotation_add("saqr_md.cluster_num_valid_queue_signals", "num_valid_ds")
        self.get_rand_leaf_id_1 = bfrt_info.table_get("SpineIngress.get_rand_leaf_id_1")
        self.get_rand_leaf_id_1.info.key_field_annotation_add("saqr_md.random_ds_index_1", "rand_idx_1")
        self.get_rand_leaf_id_2 = bfrt_info.table_get("SpineIngress.get_rand_leaf_id_2")
        self.get_rand_leaf_id_2.info.key_field_annotation_add("saqr_md.random_ds_index_2", "rand_idx_2")

    def init_data(self):
        self.pipe_id = 0
        self.TEST_VCLUSTER_ID = 0
        self.MAX_VCLUSTER_LEAVES = 16 # This number is per cluster. *Important: should be the same in p4 code (fixed at compile time)
        self.initial_idle_list = [0, 1, 2, 3]
        
        self.initial_idle_count = len(self.initial_idle_list)
        self.intitial_qlen_state = [0] * len(self.initial_idle_list)
        self.wid_port_mapping = {0:36, 1:44, 2:20, 3:52, 4:28, 5:20, 110:56, 111: 58, 100:28}
        
        if self.setup == 'b':
            self.qlen_unit = [4] * len(self.initial_idle_list) # assuming 3bit showing fraction and 5bit decimal: 0.125 (8 workers in each rack)
        elif self.setup == 's':
            self.qlen_unit = [8, 8, 4, 1] # proportional to 1/workers in the rack
        
        self.intitial_deferred_state = [0] * len(self.initial_idle_list)
        self.num_valid_ds_elements = len(self.initial_idle_list) # num available leaves this vcluster (the number in hardware will be 2^W)

        self.leaf_start_idx = self.TEST_VCLUSTER_ID * self.MAX_VCLUSTER_LEAVES
    
    def set_tables(self):
        # Insert idle_list values (wid of idle workers)
        for i in range(self.initial_idle_count):
            register_write(self.target,
                self.register_idle_list,
                register_name='SpineIngress.idle_list.f1',
                index=self.leaf_start_idx + i,
                register_value=self.initial_idle_list[i])
            
            register_write(self.target,
                self.register_idle_list_idx_mapping,
                register_name="SpineIngress.idle_list_idx_mapping.f1",
                index=self.initial_idle_list[i],
                register_value=self.leaf_start_idx + i)

        register_write(self.target,
            self.register_idle_count,
            register_name='SpineIngress.idle_count.f1',
            index=self.TEST_VCLUSTER_ID,
            register_value=self.initial_idle_count)
        test_register_read(self.target,
            self.register_idle_count,
            'SpineIngress.idle_count.f1',
            self.pipe_id,
            self.TEST_VCLUSTER_ID)
        
        
        for i, qlen in enumerate(self.intitial_qlen_state):
            register_write(self.target,
                    self.register_queue_len_list_1,
                    register_name='SpineIngress.queue_len_list_1.f1',
                    index=self.leaf_start_idx + i,
                    register_value=qlen)
            register_write(self.target,
                    self.register_queue_len_list_2,
                    register_name='SpineIngress.queue_len_list_2.f1',
                    index=self.leaf_start_idx + i,
                    register_value=qlen)
            register_write(self.target,
                    self.register_deferred_list_1,
                    register_name='SpineIngress.deferred_queue_len_list_1.f1',
                    index=self.leaf_start_idx + i,
                    register_value=self.intitial_deferred_state[i])
            register_write(self.target,
                    self.register_deferred_list_2,
                    register_name='SpineIngress.deferred_queue_len_list_2.f1',
                    index=self.leaf_start_idx + i,
                    register_value=self.intitial_deferred_state[i])

        # Table entries
        print("********* Populating Table Entires *********")
        for wid in self.wid_port_mapping.keys():
            self.forward_saqr_switch_dst.entry_add(
                self.target,
                [self.forward_saqr_switch_dst.make_key([client.KeyTuple('hdr.saqr.dst_id', wid)])],
                [self.forward_saqr_switch_dst.make_data([client.DataTuple('port', self.wid_port_mapping[wid])],
                                             'SpineIngress.act_forward_saqr')]
            )
        
        for idx, leaf_id in enumerate(self.initial_idle_list):
            self.set_queue_len_unit_1.entry_add(
                self.target,
                [self.set_queue_len_unit_1.make_key([client.KeyTuple('saqr_md.random_id_1', leaf_id), client.KeyTuple('hdr.saqr.cluster_id', TEST_VCLUSTER_ID)])],
                [self.set_queue_len_unit_1.make_data([client.DataTuple('cluster_unit', self.qlen_unit[idx])],
                                             'SpineIngress.act_set_queue_len_unit_1')]
            )
            self.set_queue_len_unit_2.entry_add(
                self.target,
                [self.set_queue_len_unit_2.make_key([client.KeyTuple('saqr_md.random_id_2', leaf_id), client.KeyTuple('hdr.saqr.cluster_id', TEST_VCLUSTER_ID)])],
                [self.set_queue_len_unit_2.make_data([client.DataTuple('cluster_unit', self.qlen_unit[idx])],
                                             'SpineIngress.act_set_queue_len_unit_2')]
            )
            self.get_rand_leaf_id_1.entry_add(
                self.target,
                [self.get_rand_leaf_id_1.make_key([client.KeyTuple('saqr_md.random_ds_index_1', idx), client.KeyTuple('hdr.saqr.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_rand_leaf_id_1.make_data([client.DataTuple('leaf_id', leaf_id)],
                                             'SpineIngress.act_get_rand_leaf_id_1')]
            )
            self.get_rand_leaf_id_2.entry_add(
                self.target,
                [self.get_rand_leaf_id_2.make_key([client.KeyTuple('saqr_md.random_ds_index_2', idx), client.KeyTuple('hdr.saqr.cluster_id', TEST_VCLUSTER_ID)])],
                [self.get_rand_leaf_id_2.make_data([client.DataTuple('leaf_id', leaf_id)],
                                             'SpineIngress.act_get_rand_leaf_id_2')]
            )

        self.get_cluster_num_valid.entry_add(
                self.target,
                [self.get_cluster_num_valid.make_key([client.KeyTuple('hdr.saqr.cluster_id', self.TEST_VCLUSTER_ID)])],
                [self.get_cluster_num_valid.make_data([client.DataTuple('num_leafs', self.num_valid_ds_elements)],
                                             'SpineIngress.act_get_cluster_num_valid_leafs')]
            )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_queue_signals', 2)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_1')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_queue_signals', 4)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_2')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_queue_signals', 5)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_2')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_queue_signals', 16)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_4')]
        )
        self.adjust_random_range_ds.entry_add(
            self.target,
            [self.adjust_random_range_ds.make_key([client.KeyTuple('saqr_md.cluster_num_valid_queue_signals', 256)])],
            [self.adjust_random_range_ds.make_data([], 'SpineIngress.adjust_random_leaf_index_8')]
        )

    def read_reg_stats(self):
        if DEBUG_DUMP_REGS:
            for i in range(4):
                test_register_read(self.target,
                    self.register_idle_list,
                    'SpineIngress.idle_list.f1',
                    self.pipe_id,
                    i)
            for i in range(1):
                test_register_read(self.target,
                    self.register_idle_count,
                    'SpineIngress.idle_count.f1',
                    self.pipe_id,
                    i)
            for i in range(4):
                test_register_read(self.target,
                    self.register_idle_list_idx_mapping,
                    "SpineIngress.idle_list_idx_mapping.f1",
                    self.pipe_id,
                    i)
            print("\n")

        resub_tot = test_register_read(self.target,
            self.register_stat_count_resub,
            'SpineIngress.stat_count_resub.f1',
            self.pipe_id,
            TEST_VCLUSTER_ID)
        print ("Total resubmissions at Spine (Task resub + Idle remove resub): %d" %(resub_tot))

        task_tot = test_register_read(self.target,
            self.register_stat_count_task,
            'SpineIngress.stat_count_task.f1',
            self.pipe_id,
            0)
        print ("Total tasks arrived at Spine: %d" %(task_tot))
        delay_list = []
        if (task_tot >= 10000):
            for i in range(10000):
                ingress_tstamp = test_register_read(self.target,
                self.register_ingress_tstamp,
                'SpineIngress.ingress_tstamp.f1',
                self.pipe_id,
                i, 
                print_reg=False)
                egress_tstamp = test_register_read(self.target,
                self.register_egress_tstamp,
                'SpineEgress.egress_tstamp.f1',
                self.pipe_id,
                i,
                print_reg=False)
                delay_list.append(egress_tstamp - ingress_tstamp)
            print (delay_list)
            np.savetxt('spine_latency.csv', [np.array(delay_list)], delimiter=', ', fmt='%d')

        # for i in range(2):
        #     test_register_read(self.target,
        #         self.register_queue_len_list_1,
        #         'SpineIngress.queue_len_list_1.f1',
        #         self.pipe_id,
        #         i)
        #     # test_register_read(self.target,
        #     #     self.register_queue_len_list_2,
        #     #     'SpineIngress.queue_len_list_2.f1',
        #     #     self.pipe_id,
        #     #     i)
        #     test_register_read(self.target,
        #         self.register_deferred_list_1,
        #         'SpineIngress.deferred_queue_len_list_1.f1',
        #         self.pipe_id,
        #         i)
        #     test_register_read(self.target,
        #         self.register_deferred_list_2,
        #         'SpineIngress.deferred_queue_len_list_2.f1',
        #         self.pipe_id,
        #         i)

if __name__ == "__main__":
    setup = str(sys.argv[1])
    if setup == 'b':
        print("*****Using the balanced worker placement*****")
        print("**Four racks with eight workers each**")
    elif setup == 's':
        print("*****Using the skewed worker placement*****")
        print("**Two racks with four workers, one rack with eight workers, one rack with 32 workers**")
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

    spine_controller = SpineController(target, bfrt_info, setup=setup)
    spine_controller.set_tables()

    leaf_controller = LeafController(target, bfrt_info, setup=setup)
    leaf_controller.set_tables()
    
    while(True):
        spine_controller.read_reg_stats()
        print ("\n")
        leaf_controller.read_reg_stats()
        print("#################")
        print("\n")
        time.sleep(2)


