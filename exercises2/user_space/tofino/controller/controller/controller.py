#!/usr/bin/env python3

import argparse
import ipaddress
from tofinoManager import TofinoManager
import bfrt_grpc.client as gc


class Controller:
    def __init__(self, tm):
        self.tm = tm
    
    def get_cap_table(self):
        return self.tm.get_tables()["cap_table"]


    # Stolen from https://github.com/nsg-ethz/ACC-Turbo/blob/86869689a511567be5b42b4e556f3f6dc53f14be/tofino/python_controller/core.py
    # This is a simple helper method that takes a list of entries and programs
    # them in a specified table
    #
    # Each entry is a tuple, consisting of 3 elements:
    #  key         -- a list of tuples for each element of the key
    #                 @signature (name, value=None, mask=None, prefix_len=None, low=None, high=None)
    #  action_name -- the action to use. Must use full name of the action
    #  data        -- a list (may be empty) of the tuples for each action parameter
    #                 @signature (name, value=None) [for complex use cases refer to bfrt client.py]
    # 
    # Examples:
    # --------------------------------
    # self.program_table("ipv4_host", [
    #         ([("hdr.ipv4.dst_addr", "192.168.1.1")],
    #          "Ingress.send", [("port", 1)])
    # ]
    # self.programTable("ipv4_lpm", [
    #       ([("hdr.ipv4.dst_addr", "192.168.1.0", None, 24)],
    #         "Ingress.send", [("port", 64)]),

    def program_table(self, table_name, entries):
        table = self.tm.get_tables()[table_name]
        key_list=[]
        data_list=[]
        for k, a, d in entries:
            key_list.append(table.make_key([gc.KeyTuple(*f)   for f in k]))
            data_list.append(table.make_data([gc.DataTuple(*p) for p in d], a))
        try:
            table.entry_add(self.tm.dev_tgt, key_list, data_list)
        except:
            table.entry_mod(self.tm.dev_tgt, key_list, data_list)



    def insert_capability(self, obj_id, owner_host, delegatee_host):
        self.program_table("cap_table", [([("cap_id", obj_id), ("src_addr", int(ipaddress.IPv4Address(owner_host)))], "Ingress.capAllow_forward", [])])

    def revoke_capability(self, obj_id, owner_host, delegatee_host):
        self.program_table("cap_table", [([("cap_id", obj_id), ("src_addr", int(ipaddress.IPv4Address(owner_host)))], "Ingress.capRevoked", [])])

    def delegate_capability(self, obj_id, owner_host, delegatee_host, delegater_host):

        # TODO(@jkrbs): check if the delegator currently has the cap
        # This might be racy

        self.program_table("cap_table", [([("cap_id", obj_id), ("src_addr", int(ipaddress.IPv4Address(owner_host)))], "Ingress.capRevoked", [])])

if __name__ == "__main__":
    tm = TofinoManager()
    c = Controller(tm)
    c.tm.print_table_info("pipe.Ingress.cap_table")
    c.insert_capability(42, "10.0.0.1", "10.0.0.0.2")
    c.tm.print_table_info("pipe.Ingress.cap_table")
    c.revoke_capability(42, "10.0.0.1", "10.0.0.0.2")
    c.tm.print_table_info("pipe.Ingress.cap_table")
    print(c.get_cap_table())