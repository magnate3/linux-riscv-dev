'''
# Monitoring/Tofino2
# Version 2.0.0
# Authors: Mateus, Alireza
# Lab: LERIS
# Date: 2025-04-14
'''

import os
import sys
import pdb
import time

# Principle Setup ***********************************************************************
SDE_INSTALL   = os.environ['SDE_INSTALL']
SDE_PYTHON2   = os.path.join(SDE_INSTALL, 'lib', 'python2.7', 'site-packages')
sys.path.append(SDE_PYTHON2)
sys.path.append(os.path.join(SDE_PYTHON2, 'tofino'))

PYTHON3_VER   = '{}.{}'.format(
    sys.version_info.major,
    sys.version_info.minor)
SDE_PYTHON3   = os.path.join(SDE_INSTALL, 'lib', 'python' + PYTHON3_VER,
                             'site-packages')
sys.path.append(SDE_PYTHON3)
sys.path.append(os.path.join(SDE_PYTHON3, 'tofino'))
sys.path.append(os.path.join(SDE_PYTHON3, 'tofino', 'bfrt_grpc'))

## Import bfrt API ***********************************************************************
import bfrt_grpc.client as gc
# Connect to the BF Runtime Server
for bfrt_client_id in range(10):
    try:
        interface = gc.ClientInterface(
            grpc_addr = 'localhost:50052',
            client_id = bfrt_client_id,
            device_id = 0,
            num_tries = 1)
        print('Connected to BF Runtime Server as client', bfrt_client_id)
        break;
    except:
        print('Could not connect to BF Runtime server')
        quit

# Get the information about the running program
bfrt_info = interface.bfrt_info_get()
print('The target runs the program ', bfrt_info.p4_name_get())

# Using the Connection to bfrt API
if bfrt_client_id == 0:
    interface.bind_pipeline_config(bfrt_info.p4_name_get())

################### You can now use BFRT CLIENT ###########################
from tabulate import tabulate

# Print the list of tables in the "pipe" node
dev_tgt = gc.Target(0, pipe_id=1)

### Reading a register ###
# print(dir(bfrt_info))
sorted_tables = bfrt_info.table_list_sorted

table_flow_id_reg = bfrt_info.table_get("pipe.Ingress.flow_id_reg")
table_packet_size_reg = bfrt_info.table_get("pipe.Ingress.packet_size_reg")
table_ipg_cloned_packets_reg = bfrt_info.table_get("pipe.Ingress.ipg_cloned_packets_reg")
table_frame_size_reg = bfrt_info.table_get("pipe.Ingress.frame_size_reg")
table_ifg_reg = bfrt_info.table_get("pipe.Ingress.ifg_reg")
table_metadata_classT1 = bfrt_info.table_get("pipe.Ingress.metadata_classT1")
table_metadata_classT2 = bfrt_info.table_get("pipe.Ingress.metadata_classT2")
table_metadata_classT3 = bfrt_info.table_get("pipe.Ingress.metadata_classT3")
table_metadata_classT4 = bfrt_info.table_get("pipe.Ingress.metadata_classT4")
table_metadata_classT5 = bfrt_info.table_get("pipe.Ingress.metadata_classT5")
table_ingress_host_ifg_reg = bfrt_info.table_get("pipe.Ingress.ingress_host_ifg_reg")

table_frame_counter_threshhold = bfrt_info.table_get("pipe.Egress.frame_counter_threshhold")
table_packet_counter_threshhold = bfrt_info.table_get("pipe.Egress.packet_counter_threshhold")


print("FlowID, PS, IPG, FS, IFG t1, t2, t3, t4, t5, host_ifg")

while True:

    #FLOWID
    key_flow_id_reg = table_flow_id_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)])
    flow_id_reg_read_result = table_flow_id_reg.entry_get(dev_tgt, [key_flow_id_reg], {"from_hw": True}) 
    flow_id_reg_entry_data, _ = next(flow_id_reg_read_result)
    flow_id_reg_entry_dict = flow_id_reg_entry_data.to_dict()
    flow_id_reg_f1_value = flow_id_reg_entry_dict["Ingress.flow_id_reg.f1"]
    flow_id_reg_value = flow_id_reg_f1_value[0]
    #print(flow_id_reg_value)

    #t1
    key_metadata_classT1 = table_metadata_classT1.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    metadata_classT1_read_result = table_metadata_classT1.entry_get(dev_tgt, [key_metadata_classT1], {"from_hw": True}) #, "print_zero": False})
    metadata_classT1_entry_data, _ = next(metadata_classT1_read_result)
    metadata_classT1_entry_dict = metadata_classT1_entry_data.to_dict()
    metadata_classT1_f1_value = metadata_classT1_entry_dict["Ingress.metadata_classT1.f1"]
    metadata_classT1_value = metadata_classT1_f1_value[0]
    #t2
    key_metadata_classT2 = table_metadata_classT2.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    metadata_classT2_read_result = table_metadata_classT2.entry_get(dev_tgt, [key_metadata_classT2], {"from_hw": True}) 
    metadata_classT2_entry_data, _ = next(metadata_classT2_read_result)
    metadata_classT2_entry_dict = metadata_classT2_entry_data.to_dict()
    metadata_classT2_f1_value = metadata_classT2_entry_dict["Ingress.metadata_classT2.f1"]
    metadata_classT2_value = metadata_classT2_f1_value[0]

    #t3
    key_metadata_classT3 = table_metadata_classT3.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    metadata_classT3_read_result = table_metadata_classT3.entry_get(dev_tgt, [key_metadata_classT3], {"from_hw": True}) 
    metadata_classT3_entry_data, _ = next(metadata_classT3_read_result)
    metadata_classT3_entry_dict = metadata_classT3_entry_data.to_dict()
    metadata_classT3_f1_value = metadata_classT3_entry_dict["Ingress.metadata_classT3.f1"]
    metadata_classT3_value = metadata_classT3_f1_value[0]

    #t4
    key_metadata_classT4 = table_metadata_classT4.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    metadata_classT4_read_result = table_metadata_classT4.entry_get(dev_tgt, [key_metadata_classT4], {"from_hw": True}) 
    metadata_classT4_entry_data, _ = next(metadata_classT4_read_result)
    metadata_classT4_entry_dict = metadata_classT4_entry_data.to_dict()
    metadata_classT4_f1_value = metadata_classT4_entry_dict["Ingress.metadata_classT4.f1"]
    metadata_classT4_value = metadata_classT4_f1_value[0]

    #t5
    key_metadata_classT5 = table_metadata_classT5.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    metadata_classT5_read_result = table_metadata_classT5.entry_get(dev_tgt, [key_metadata_classT5], {"from_hw": True}) 
    metadata_classT5_entry_data, _ = next(metadata_classT5_read_result)
    metadata_classT5_entry_dict = metadata_classT5_entry_data.to_dict()
    metadata_classT5_f1_value = metadata_classT5_entry_dict["Ingress.metadata_classT5.f1"]
    metadata_classT5_value = metadata_classT5_f1_value[0]


    #PACKET SIZE
    key_packet_size_reg = table_packet_size_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    packet_size_reg_read_result = table_packet_size_reg.entry_get(dev_tgt, [key_packet_size_reg], {"from_hw": True}) 
    packet_size_reg_entry_data, _ = next(packet_size_reg_read_result)
    packet_size_reg_entry_dict = packet_size_reg_entry_data.to_dict()
    packet_size_reg_f1_value = packet_size_reg_entry_dict["Ingress.packet_size_reg.f1"]
    packet_size_reg_value = packet_size_reg_f1_value[0]

    # #IPG
    key_ipg_cloned_packets_reg = table_ipg_cloned_packets_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    ipg_cloned_packets_reg_read_result = table_ipg_cloned_packets_reg.entry_get(dev_tgt, [key_ipg_cloned_packets_reg], {"from_hw": True}) #, "print_zero": False})
    ipg_cloned_packets_reg_entry_data, _ = next(ipg_cloned_packets_reg_read_result)
    ipg_cloned_packets_reg_entry_dict = ipg_cloned_packets_reg_entry_data.to_dict()
    ipg_cloned_packets_reg_f1_value = ipg_cloned_packets_reg_entry_dict["Ingress.ipg_cloned_packets_reg.f1"]
    ipg_cloned_packets_reg_value = ipg_cloned_packets_reg_f1_value[0]

    # #FRAME SIZE
    key_frame_size_reg = table_frame_size_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    frame_size_reg_read_result = table_frame_size_reg.entry_get(dev_tgt, [key_frame_size_reg], {"from_hw": True}) #, "print_zero": False})
    frame_size_reg_entry_data, _ = next(frame_size_reg_read_result)
    frame_size_reg_entry_dict = frame_size_reg_entry_data.to_dict()
    frame_size_reg_f1_value = frame_size_reg_entry_dict["Ingress.frame_size_reg.f1"]
    frame_size_reg_value = frame_size_reg_f1_value[0]

    # #ifg
    # print(dir(table_ifg_reg))
    key_ifg_reg = table_ifg_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    ifg_reg_read_result = table_ifg_reg.entry_get(dev_tgt, [key_ifg_reg], {"from_hw": True}) #, "print_zero": False})
    ifg_reg_entry_data, _ = next(ifg_reg_read_result)
    ifg_reg_entry_dict = ifg_reg_entry_data.to_dict()
    ifg_reg_f1_value = ifg_reg_entry_dict["Ingress.ifg_reg.f1"]
    ifg_reg_value = ifg_reg_f1_value[0]


    # Alireza added
    #host_ifg
    key_ingress_host_ifg_reg = table_ingress_host_ifg_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', flow_id_reg_value)])
    ingress_host_ifg_reg_read_result = table_ingress_host_ifg_reg.entry_get(dev_tgt, [key_ingress_host_ifg_reg], {"from_hw": True}) #, "print_zero": False})
    ingress_host_ifg_reg_entry_data, _ = next(ingress_host_ifg_reg_read_result)
    ingress_host_ifg_reg_entry_dict = ingress_host_ifg_reg_entry_data.to_dict()
    ingress_host_ifg_reg_f1_value = ingress_host_ifg_reg_entry_dict["Ingress.ingress_host_ifg_reg.f1"]
    ingress_host_ifg_reg_value = ingress_host_ifg_reg_f1_value[0]
    # Alireza Added
    print(f"{flow_id_reg_value}, {packet_size_reg_value}, {ipg_cloned_packets_reg_value}, {frame_size_reg_value}, {ifg_reg_value}, {metadata_classT1_value}, {metadata_classT2_value}, {metadata_classT3_value}, {metadata_classT4_value}, {metadata_classT5_value},{ingress_host_ifg_reg_value/90000}")


    #Setiing Frame Number Threshold 
    # Keys
    keys_frame_counter_threshhold = [
        table_frame_counter_threshhold.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)]),
        table_frame_counter_threshhold.make_key([gc.KeyTuple('$REGISTER_INDEX', 1)]),
        table_frame_counter_threshhold.make_key([gc.KeyTuple('$REGISTER_INDEX', 2)]),
        table_frame_counter_threshhold.make_key([gc.KeyTuple('$REGISTER_INDEX', 3)])
   
    ]
    # Values
    data_reset_frame_counter_threshhold = [ # Q0_136, Q0_137, Q1_136, Q1_137 (order)
        table_frame_counter_threshhold.make_data([gc.DataTuple('Egress.frame_counter_threshhold.f1', 4)]),
        table_frame_counter_threshhold.make_data([gc.DataTuple('Egress.frame_counter_threshhold.f1', 4)]),
        table_frame_counter_threshhold.make_data([gc.DataTuple('Egress.frame_counter_threshhold.f1', 4)]),
        table_frame_counter_threshhold.make_data([gc.DataTuple('Egress.frame_counter_threshhold.f1', 4)])
       
    ]

    table_frame_counter_threshhold.entry_mod(dev_tgt, key_list=keys_frame_counter_threshhold, data_list=data_reset_frame_counter_threshhold)


    #packet_counter_threshhold
    #Setiing Frame Number Threshold 
    # Keys
    keys_packet_counter_threshhold = [
        table_packet_counter_threshhold.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)]),
        table_packet_counter_threshhold.make_key([gc.KeyTuple('$REGISTER_INDEX', 1)]),
        table_packet_counter_threshhold.make_key([gc.KeyTuple('$REGISTER_INDEX', 2)]),
        table_packet_counter_threshhold.make_key([gc.KeyTuple('$REGISTER_INDEX', 3)])
    ]
    # Values
    data_reset_packet_counter_threshhold = [ # Q0_136, Q0_137, Q1_136, Q1_137 (order)
        table_packet_counter_threshhold.make_data([gc.DataTuple('Egress.packet_counter_threshhold.f1', 20)]),
        table_packet_counter_threshhold.make_data([gc.DataTuple('Egress.packet_counter_threshhold.f1', 20)]),
        table_packet_counter_threshhold.make_data([gc.DataTuple('Egress.packet_counter_threshhold.f1', 20)]),
        table_packet_counter_threshhold.make_data([gc.DataTuple('Egress.packet_counter_threshhold.f1', 20)])
       
    ]
    table_packet_counter_threshhold.entry_mod(dev_tgt, key_list=keys_packet_counter_threshhold, data_list=data_reset_frame_counter_threshhold)

