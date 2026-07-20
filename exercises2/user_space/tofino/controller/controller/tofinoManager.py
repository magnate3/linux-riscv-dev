
#!/usr/bin/python
import pprint
import os
import sys
sys.path.append(f"{os.environ['SDE_INSTALL']}/lib/python3.10/site-packages/tofino")

import bfrt_grpc.client as gc


# Initialization Code is copied fro
# https://github.com/nsg-ethz/ACC-Turbo/blob/86869689a511567be5b42b4e556f3f6dc53f14be/tofino/python_controller/core.py
class TofinoManager():
    def __init__(self, client_id=0, p4_name=None, grpc_addr='localhost:50052'):
        try:
            self.setup_grpc(client_id, p4_name, grpc_addr)
        except Exception as e:
            print("Error init: {}".format(e))

    def setup_grpc(self, client_id, p4_name, grpc_addr):
        self.client_id = client_id
        self.dev      = 0
        self.dev_tgt  = gc.Target(self.dev, pipe_id=0xFFFF)
        self.bfrt_info = None

        self.interface = gc.ClientInterface(grpc_addr, client_id=client_id,
                device_id=self.dev, notifications=None)

        if not p4_name:
            self.bfrt_info = self.interface.bfrt_info_get()
            p4_name = self.bfrt_info.p4_name_get()

        self.interface.bind_pipeline_config(p4_name)
        self.p4_name = p4_name

        print("    Connected to Device: {}, Program: {}, ClientId: {}".format(
            self.dev, self.p4_name, self.client_id))

    def insert(self, table_name, index, action):
        table  = self.bfrt_info.table_dict[table_name]

        table.entry_add(
            self.dev_tgt,
            table.make_key([gc.KeyTuple('cap_id', index[0]),
                            gc.KeyTuple('src_addr', index[1])]),
            table.make_data([], action_name=action)
        )

    def get_table_names(self) -> [str]:
        return self.bfrt_info.table_dict.keys()
    def get_tables(self):
        return self.bfrt_info.table_dict

    def print_table_info(self, table_name):
        print("====Table Info===")
        t = self.bfrt_info.table_dict[table_name]
        print("{:<30}: {}".format("TableName", t.info.name_get()))
        print("{:<30}: {}".format("Size", t.info.size_get()))
        print("{:<30}: {}".format("Actions", t.info.action_name_list_get()))
        print("{:<30}:".format("KeyFields"))
        for field in sorted(t.info.key_field_name_list_get()):
            print("  {:<28}: {} => {}".format(field, t.info.key_field_type_get(field), t.info.key_field_match_type_get(field)))
        print("{:<30}:".format("DataFields"))
        for field in t.info.data_field_name_list_get():
            print("  {:<28}: {} {}".format(
                "{} ({})".format(field, t.info.data_field_id_get(field)), 
                # type(t.info.data_field_allowed_choices_get(field)), 
                t.info.data_field_type_get(field),
                t.info.data_field_size_get(field),
                ))
        print("================")

if __name__ == "__main__":
    tm = TofinoManager()
    print(tm.get_table_names())

    print("+++++++")
    tm.print_table_info("pipe.Ingress.cap_table")

    print("Performing Insert")
    tm.insert("pipe.Ingress.cap_table", [42,  0x2324], "Ingress.drop")
