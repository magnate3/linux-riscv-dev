#!/usr/bin/env python3
import sys
import os
import argparse
import time
import scapy
from scapy.all import Ether, RandMAC

sde_install = os.environ['SDE_INSTALL']
sys.path.append('%s/lib/python2.7/site-packages/tofino'%(sde_install))
sys.path.append('%s/lib/python2.7/site-packages/p4testutils'%(sde_install))
sys.path.append('%s/lib/python2.7/site-packages'%(sde_install))

# Assumes valid PYTHONPATH
import bfrt_grpc.client as gc

# For getting the dev_ports from the front panel ports itself
def get_devport(frontpanel, lane):
    port_hdl_info = bfrt_info.table_get("$PORT_HDL_INFO")
    key = port_hdl_info.make_key(
        [gc.KeyTuple("$CONN_ID", frontpanel), gc.KeyTuple("$CHNL_ID", lane)]
    )
    for data, _ in port_hdl_info.entry_get(target, [key], {"from_hw": False}):
        devport = data.to_dict()["$DEV_PORT"]
        if devport:
            return devport

# Connect to the BF Runtime server
for bfrt_client_id in range(10):
    try:
        interface = gc.ClientInterface(
            grpc_addr="localhost:50052",
            client_id=bfrt_client_id,
            device_id=0,
            num_tries=1,
        )
        print("Connected to BF Runtime Server as client", bfrt_client_id)
        break
    except:
        print("Could not connect to BF Runtime Server")
        quit

# Get information about the running program
bfrt_info = interface.bfrt_info_get()
print("The target is running the P4 program: {}".format(bfrt_info.p4_name_get()))

# Establish that you are the "main" client
if bfrt_client_id == 0:
    interface.bind_pipeline_config(bfrt_info.p4_name_get())

# Get the target device, currently setup for all pipes
target = gc.Target(device_id=0, pipe_id=0xffff)

parser = argparse.ArgumentParser(description="Test Configuration")
parser.add_argument('-r', type=int, help="Line rate (Gbps),           Default :  0.08", default=0.08)
parser.add_argument('-b', type=int, help="Duration (s),         Default :  30", default=30)
parser.add_argument('-s', type=int, help="Size of the packets in (B)  Default :  256", default=256)
args = parser.parse_args()

# Getting the pktgen tables
pktgen_buffer = bfrt_info.table_get("tf1.pktgen.pkt_buffer")
pktgen_port = bfrt_info.table_get("tf1.pktgen.port_cfg")
pktgen_app = bfrt_info.table_get("tf1.pktgen.app_cfg")

# sport_value = 1234
src_mac = RandMAC()
dst_mac = RandMAC()
p=Ether(src=src_mac, dst=dst_mac, type=0x8122)/(b'\x01'*( args.s - 22 ))
packet_len = len(p) - 6

# Configuring pktgen port
pktgen_port_key = pktgen_port.make_key([gc.KeyTuple('dev_port', 196)])
pktgen_port_action_data = pktgen_port.make_data([gc.DataTuple('pktgen_enable', bool_val=True)])
pktgen_port.entry_mod(target,[pktgen_port_key],[pktgen_port_action_data])

# Configuring pktgen buffer
offset = 0
pktgen_pkt_buf_key = pktgen_buffer.make_key([gc.KeyTuple('pkt_buffer_offset', offset),gc.KeyTuple('pkt_buffer_size', packet_len)])
pktgen_pkt_buf_action_data = pktgen_buffer.make_data([gc.DataTuple('buffer', bytearray(bytes(p)[6:]))])
pktgen_buffer.entry_mod(target,[pktgen_pkt_buf_key],[pktgen_pkt_buf_action_data])

## Configuring pktgen parameters
padding = args.s % 4
overhead = 98 + padding
INTER_PACKET_GAP_NS = round((packet_len + overhead) * 8 / args.r)

## Configuring pktgen app
pktgen_app_key = pktgen_app.make_key([gc.KeyTuple('app_id', 0)])
pktgen_app_action_data = pktgen_app.make_data([
    gc.DataTuple('timer_nanosec', 10),
    gc.DataTuple('app_enable', bool_val=True),
    gc.DataTuple('pkt_len', packet_len),
    gc.DataTuple('pkt_buffer_offset', 0),
    gc.DataTuple('pipe_local_source_port', 68),
    gc.DataTuple('increment_source_port', bool_val=False),
    gc.DataTuple('batch_count_cfg', 0),
    gc.DataTuple('packets_per_batch_cfg', 1),
    gc.DataTuple('ibg', 0),
    gc.DataTuple('ibg_jitter', 0),
    gc.DataTuple('ipg', INTER_PACKET_GAP_NS),
    gc.DataTuple('ipg_jitter', 0),
    gc.DataTuple('batch_counter', 0),
    gc.DataTuple('pkt_counter', 0),
    gc.DataTuple('trigger_counter', 0)], 'trigger_timer_periodic')
pktgen_app.entry_mod(target,[pktgen_app_key],[pktgen_app_action_data])
print("Packet generation is completed")

port31 = get_devport(31, 0)
port32 = get_devport(32, 0)
dev_ports=[port31,port32]
print(port31)
print(port32)

port_stat_table = bfrt_info.table_get("$PORT_STAT")
keys = [ port_stat_table.make_key([gc.KeyTuple('$DEV_PORT', dp)]) for dp in dev_ports ]

# Getting the rates
while 1:
    resp = list(port_stat_table.entry_get(target, keys, {'from_hw': False}, None))
    data_dict0 = resp[0][0].to_dict()
    tx_pps_31 = data_dict0['$TX_PPS']
    rx_pps_31 = data_dict0['$RX_PPS']
    tx_rate_31 = data_dict0['$TX_RATE']
    rx_rate_31 = data_dict0['$RX_RATE']

    data_dict1 = resp[1][0].to_dict()
    tx_pps_32 = data_dict1['$TX_PPS']
    rx_pps_32 = data_dict1['$RX_PPS']
    tx_rate_32 = data_dict1['$TX_RATE']
    rx_rate_32 = data_dict1['$RX_RATE']

    # print("For port 31, Tx rate = "+str(tx_rate_31)+" Tx Pps = "+str(tx_pps_31))
    print("For port 32, Tx rate = "+str(tx_rate_32)+" Tx Pps = "+str(tx_pps_32))

    time.sleep(1)

# pkt_out = bfrt_info.table_get("pipe.Ingress.reg")
# key = [pkt_out.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)])]
# data = [pkt_count.make_data([gc.DataTuple('SwitchIngress.active.f1', 0)])]

# for data, key in pkt_out.entry_get(target, key, {"from_hw": True}):
#     out = data.to_dict()["Ingress.reg.f1"]

# print("The packet out of 1c is",out[0])


# pkt_count = bfrt_info.table_get("pipe.Ingress.reg_2")
# key = [pkt_count.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)])]
# data = [pkt_count.make_data([gc.DataTuple('SwitchIngress.active.f1', 0)])]

# for data, key in pkt_count.entry_get(target, key, {"from_hw": True}):
#     count = data.to_dict()["Ingress.reg_2.f1"]

# print("The packet in to 1c is",count[0])

# if((out[0]-count[0])>1):
#     print("packet loss")

#pktgen_app_action_data=pktgen_app.make_data([gc.DataTuple('app_enable',bool_val=False)])                                            
#pktgen_app.entry_mod(target,[pktgen_app_key],[pktgen_app_action_data])
#print("packet gen is stopped")

# if()
# time.sleep(1) # Sleep for 1 second
# pktgen_app_action_data=pktgen_app.make_data([gc.DataTuple('app_enable',bool_val=True)])                                            
# pktgen_app.entry_mod(target,[pktgen_app_key],[pktgen_app_action_data])
# print("packet gen is stopped")


# If we ever need a forwarding logic at 1c
    # forwarding = bfrt_info.table_get("Ingress.forwarding")
    # forwarding_keys = [
    #     forwarding.make_key([gc.KeyTuple("ig_intr_md.ingress_port", port15)])
    # ]
    # forwarding_data = [
    #     forwarding.make_data(
    #         [gc.DataTuple("egress_port", port16)], "Ingress.set_egress_port"
    #     )
    # ]
    # forwarding.entry_add(target, forwarding_keys, forwarding_data)
    # print("Programmed Forwarding Table")



# target = gc.Target(device_id=0, pipe_id=0xffff)
# global global_grpc_comm_interface
# bfrt_info = global_grpc_comm_interface.bfrt_info_get("tcp_fsm")
# pktgen_buffer = bfrt_info.table_get("tf1.pktgen.pkt_buffer")
# pktgen_port = bfrt_info.table_get("tf1.pktgen.port_cfg")
# pktgen_app = bfrt_info.table_get("tf1.pktgen.app_cfg")

# sport_value = 1234
# iface=ports.CPU_NETWORK_INTERFACE
# src_mac = "00:AA:BB:CC:DD:EE"
# dst_mac = "00:EE:DD:CC:BB:AA"
# p=Ether(src=src_mac, dst=dst_mac)/Dot1Q(vlan=7)/IP(src="42.42.42.42" , dst="1.1.1.1")/TCP(dport=443, sport=sport_value,flags='R')
# p.show()
# packet_len = len(p)
# ## Configuring pktgen port
# pktgen_port_key = pktgen_port.make_key([gc.KeyTuple('dev_port', 68)])
# pktgen_port_action_data = pktgen_port.make_data([gc.DataTuple('pktgen_enable', bool_val=True)])
# pktgen_port.entry_add(target,[pktgen_port_key],[pktgen_port_action_data])
# ## Configuring pktgen buffer
# offset = 0
# pktgen_pkt_buf_key = pktgen_buffer.make_key([gc.KeyTuple('pkt_buffer_offset', offset),gc.KeyTuple('pkt_buffer_size', packet_len)])
# pktgen_pkt_buf_action_data = pktgen_buffer.make_data([gc.DataTuple('buffer', bytearray(bytes(p)))])
# pktgen_buffer.entry_add(target,[pktgen_pkt_buf_key],[pktgen_pkt_buf_action_data])

# ## Configuring pktgen app
# pktgen_app_key = pktgen_app.make_key([gc.KeyTuple('app_id', 0)])
# pktgen_app_action_data = pktgen_app.make_data([gc.DataTuple('timer_nanosec', 500000000),
#                                                     gc.DataTuple('app_enable', bool_val=True),
#                                                     gc.DataTuple('pkt_len', packet_len),
#                                                     gc.DataTuple('pkt_buffer_offset', 0),
#                                                     gc.DataTuple('pipe_local_source_port', 68),
#                                                     gc.DataTuple('increment_source_port', bool_val=False),
#                                                     gc.DataTuple('batch_count_cfg', 0),
#                                                     gc.DataTuple('packets_per_batch_cfg', 1),
#                                                     gc.DataTuple('ibg', 10000),
#                                                     gc.DataTuple('ibg_jitter', 0),
#                                                     gc.DataTuple('ipg', 500),
#                                                     gc.DataTuple('ipg_jitter', 1000),
#                                                     gc.DataTuple('batch_counter', 0),
#                                                     gc.DataTuple('pkt_counter', 0),
#                                                     gc.DataTuple('trigger_counter', 0)],
#                                                     'trigger_timer_periodic')
# pktgen_app.entry_add(target,[pktgen_app_key],[pktgen_app_action_data])


# def connect():
#     # Connect to BfRt Server
#     interface = gc.ClientInterface(grpc_addr='localhost:50052', client_id=0, device_id=0)
#     target = gc.Target(device_id=0, pipe_id=0xFFFF)
#     # print('Connected to BfRt Server!')

#     # Get the information about the running program
#     bfrt_info = interface.bfrt_info_get()
#     # print('The target is running the', bfrt_info.p4_name_get())

#     # Establish that you are working with this program
#     interface.bind_pipeline_config(bfrt_info.p4_name_get())
#     return interface, target, bfrt_info

# def disable(connection):
#     interface = connection[0]
#     target = connection[1]
#     bfrt_info = connection[2]
#     active_reg = bfrt_info.table_get('pipe.SwitchIngress.active')
#     key = [active_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)])]
#     data = [active_reg.make_data([gc.DataTuple('SwitchIngress.active.f1', 0)])]
#     active_reg.entry_mod(target, key, data)
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     print('OPERATION: Switching is disabled! :(')
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

# def enable(connection):
#     interface = connection[0]
#     target = connection[1]
#     bfrt_info = connection[2]
#     active_reg = bfrt_info.table_get('pipe.SwitchIngress.active')
#     key = [active_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)])]
#     data = [active_reg.make_data([gc.DataTuple('SwitchIngress.active.f1', 1)])]
#     active_reg.entry_mod(target, key, data)
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     print('OPERATION: Switching is enabled! :D')
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

# def configure(connection, sequence, sess_num):
#     interface = connection[0]
#     target = connection[1]
#     bfrt_info = connection[2]
#     sess_reg = bfrt_info.table_get('pipe.SwitchIngress.session')
#     key = [sess_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)])]
#     data = [sess_reg.make_data([gc.DataTuple('SwitchIngress.session.f1', sess_num)])]
#     sess_reg.entry_mod(target, key, data)
#     counter_reg = bfrt_info.table_get('pipe.SwitchIngress.counter')
#     key = [counter_reg.make_key([gc.KeyTuple('$REGISTER_INDEX', 0)])]
#     data = [counter_reg.make_data([gc.DataTuple('SwitchIngress.counter.f1', sequence)])]
#     counter_reg.entry_mod(target, key, data)
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     print('OPERATION: Updated session number to ' + str(sess_num) + ' and message sequence number to ' + str(sequence))
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


# def main():
#     parser1 = argparse.ArgumentParser()
#     group = parser1.add_mutually_exclusive_group()
#     group.add_argument(
#         '--disable', default=False, action='store_true',
#         help='disable switch forwarding'
#     )
#     group.add_argument(
#         '--enable', default=False, action='store_true',
#         help='enable switch forwarding'
#     )
#     parser2 = argparse.ArgumentParser()
#     subparsers = parser2.add_subparsers()
#     subparser1 = subparsers.add_parser('config')

#     subparser1.add_argument(

#         '--sequence', type=int, default=1, required=True,

#         help='specify starting message sequence number'

#     )

#     subparser1.add_argument(

#         '--session', type=int, default=0, required=True,

#         help='specify current session number'

#     )

#     args, extras = parser1.parse_known_args()
#     to_disable = args.disable
#     to_enable = args.enable
    
#     if to_disable or to_enable:

#         pprint(args)

#         if len(extras) > 0:

#             print('PARSER: Remaining arguments are omitted.')

#     else:

#         if len(extras) > 0 and extras[0] in ['config']:

#             args = parser2.parse_args(extras, namespace=args)

#             pprint(args)

#     sequence = args.sequence if 'sequence' in args else None

#     sess_num = args.session if 'session' in args else None

#     if to_disable:

#         disable(connect())

#         return

#     if to_enable:

#         enable(connect())

#         return

#     if not sequence == None and not sess_num == None:

#         configure(connect(), sequence, sess_num)

#         return

#     print('Nothing was done. :)')

# if __name__ == '__main__':
#     main()
