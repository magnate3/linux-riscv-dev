# bfrt_controller/controller.py

# import bfrt_grpc.bfruntime_pb2_grpc
# import bfrt_grpc.client as gc

from bfrt_controller.bfrt_grpc import bfruntime_pb2_grpc
from bfrt_controller.bfrt_grpc import client as gc
from tabulate import tabulate

from .ports import PortManager
from .logger import log
from .utils import is_valid_ip, format_value

class Controller:
    def __init__(self, bfrt_ip="localhost", bfrt_port="50052", pipe_id=0xFFFF):
        self.log = log
        self.bfrt_ip = bfrt_ip
        self.bfrt_port = bfrt_port
        self.device_id = 0
        self.interface = gc.ClientInterface(f"{self.bfrt_ip}:{self.bfrt_port}", client_id=0, device_id=0)
        self.bfrt_info = self.interface.bfrt_info_get()
        self.target = gc.Target(self.device_id, pipe_id=pipe_id)
        self.p4_name = self.bfrt_info.p4_name_get()
        self.log.info(f"Connected to {self.p4_name}")
        self.interface.bind_pipeline_config(self.p4_name)
        self.port_manager = PortManager(self.target, gc, self.bfrt_info)

    def setup_tables(self, table_names):
        self.tables = {}
        for t in table_names:
            self.tables[t] = self.bfrt_info.table_get(t)

    def get_entries(self, table_name, print_entries=False):
        entries = []
        t = self.tables[table_name]
        entry_output = ""

        if print_entries:
            header = "======== %s ========\n" % t.info.name_get()
            print(header)
            entry_output += header

        for d, k in t.entry_get(self.target):
            key_dict = k.to_dict()
            data_dict = d.to_dict()
            if print_entries:
                entry_str = self._print_entry(key_dict, data_dict)
                print(entry_str)
                entry_output += entry_str
            entries.append((key_dict, data_dict))

        if print_entries:
            return entry_output.strip()  # Remove any trailing whitespace/newlines
        return entries

    def list_tables(self):
        self.log.info(", ".join(sorted(self.bfrt_info.table_dict.keys())))

    def _print_entry(self, keys, data):
        if len(keys) == 0:
            return ""

        # Prepare keys for tabular output
        key_rows = []
        for k, v in keys.items():
            if len(v) == 1:
                value_display = format_value(k, v["value"])
            else:
                if "prefix_len" in v:
                    value_display = f"{v['value']}/{v['prefix_len']}"
                elif "mask" in v:
                    value_display = f"{v['value']}/{v['mask']}"
                elif "low" in v:
                    value_display = f"{v['low']} .. {v['high']}"
                else:
                    value_display = str(v)
            key_rows.append([k, value_display])

        # Prepare data for tabular output
        data_rows = [[k, v] for k, v in data.items()]

        # Generate tabular output as strings
        output = "\n== Entry ==\n"
        if key_rows:
            output += "Keys:\n"
            output += tabulate(key_rows, tablefmt="plain") + "\n"
        if data_rows:
            output += "\nData:\n"
            output += tabulate(data_rows, tablefmt="plain") + "\n"

        return output

    def table_exists(self, table_name):
        if table_name not in self.tables:
            print("{} not found in tables. Did you pass it to setup_tables?".format(table_name))
            return False
        return True

    def print_table_info(self, table_name):
        if not self.table_exists(table_name):
            return
        print("====Table Info===")
        t = self.tables[table_name]
        print("{:<30}: {}".format("TableName", t.info.name_get()))
        print("{:<30}: {}".format("Size", t.info.size_get()))
        print("{:<30}: {}".format("Actions", t.info.action_name_list_get()))
        print("{:<30}:".format("KeyFields"))
        for field in sorted(t.info.key_field_name_list_get()):
            print(
                "  {:<28}: {} => {}".format(
                    field, t.info.key_field_type_get(field), t.info.key_field_match_type_get(field)
                )
            )
        print("{:<30}:".format("DataFields"))
        for field in t.info.data_field_name_list_get():
            print(
                "  {:<28}: {} {}".format(
                    "{} ({})".format(field, t.info.data_field_id_get(field)),
                    # type(t.info.data_field_allowed_choices_get(field)),
                    t.info.data_field_type_get(field),
                    t.info.data_field_size_get(field),
                )
            )
        print("================")

    def clear_tables(self):
        try:
            for table_name in self.tables:
                t = self.tables[table_name]
                print("Clearing Table {}".format(t.info.name_get()))
                keys = []
                for d, k in t.entry_get(self.target):
                    if k is not None:
                        keys.append(k)
                try:
                    t.entry_del(self.target, keys)
                except:
                    pass
                # Not all tables support default entry
                try:
                    t.default_entry_reset(self.target)
                except:
                    pass
        except Exception as e:
            self.log.error("Error cleaning up: {}".format(e))

    def add_annotation(self, table_name, field, annotation):
        try:
            self.tables[table_name].info.key_field_annotation_add(field, annotation)
        except Exception as e:
            self.log.error("Exc in add_annotation:", e)

    #
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
        table = self.tables[table_name]
        key_list = []
        data_list = []
        for k, a, d in entries:
            key_list.append(table.make_key([gc.KeyTuple(*f) for f in k]))
            data_list.append(table.make_data([gc.DataTuple(*p) for p in d], a))
        try:
            table.entry_add(self.target, key_list, data_list)
        except:
            table.entry_mod(self.target, key_list, data_list)

    # ALWAYS call tear down at the end
    def tear_down(self):
        self.interface.tear_down_stream()

    def add_multicast_node(self, mc_node_id, port_list):
        mc_node_table = self.bfrt_info.table_get("$pre.node")
        mc_node_table.entry_add(
            self.target,
            [mc_node_table.make_key([gc.KeyTuple("$MULTICAST_NODE_ID", mc_node_id)])],
            [
                mc_node_table.make_data(
                    [gc.DataTuple("$MULTICAST_RID", 0), gc.DataTuple("$DEV_PORT", int_arr_val=port_list)]
                )
            ],
        )

    def add_multicast_nodes(self, entries):
        for mc_node_id, port_list in entries:
            try:
                self.add_multicast_node(mc_node_id, port_list)
            except gc.BfruntimeReadWriteRpcException as e:
                self.log.error(e)

    def add_multicast_group(self, mc_grp_id, mc_node_id):
        mc_mgid_table = self.bfrt_info.table_get("$pre.mgid")
        mc_mgid_table.entry_add(
            self.target,
            [mc_mgid_table.make_key([gc.KeyTuple("$MGID", mc_grp_id)])],
            [
                mc_mgid_table.make_data(
                    [
                        gc.DataTuple("$MULTICAST_NODE_ID", int_arr_val=[mc_node_id]),
                        gc.DataTuple("$MULTICAST_NODE_L1_XID_VALID", bool_arr_val=[False]),
                        gc.DataTuple("$MULTICAST_NODE_L1_XID", int_arr_val=[0]),
                    ]
                )
            ],
        )

    def add_mirror_entry(self, session_id, egress_port, max_pkt_len=16384, direction="INGRESS"):
        mirror_cfg_table = self.bfrt_info.table_get("$mirror.cfg")
        mirror_cfg_table.entry_add(
            self.target,
            [mirror_cfg_table.make_key([gc.KeyTuple("$sid", session_id)])],
            [
                mirror_cfg_table.make_data(
                    [
                        gc.DataTuple("$direction", str_val=direction),
                        gc.DataTuple("$ucast_egress_port", egress_port),
                        gc.DataTuple("$ucast_egress_port_valid", bool_val=True),
                        gc.DataTuple("$session_enable", bool_val=True),
                        gc.DataTuple("$max_pkt_len", max_pkt_len),
                    ],
                    "$normal",
                )
            ],
        )

    def read_counter(self, table_name, index=None):
        if not self.table_exists(table_name):
            log.warning(f"Table {table_name} is not setup!")
            return
        counter_table = self.bfrt_info.table_get(table_name)
        size = counter_table.info.size
        # print(f"Counter size: {size}")

        def get_counter_at_index(idx):
            resp = counter_table.entry_get(
                self.target, [counter_table.make_key([gc.KeyTuple("$COUNTER_INDEX", idx)])], {"from_hw": True}, None
            )
            # parse resp to get the counter
            data_dict = next(resp)[0].to_dict()
            pkts = data_dict["$COUNTER_SPEC_PKTS"]
            bytes = data_dict["$COUNTER_SPEC_BYTES"]
            return pkts, bytes

        if index is not None:
            return get_counter_at_index(index)
        else:
            results = []
            for idx in range(size):
                pkts, bytes = get_counter_at_index(idx)
                results.append((idx, pkts, bytes))
            return results

    def read_register(self, reg_name, index=None, pipe=0):
        if not self.table_exists(reg_name):
            log.warning(f"Table {reg_name} is not setup!")
            return
        register_table = self.bfrt_info.table_get(reg_name)

        def get_register_value_at_index(idx):
            resp = register_table.entry_get(
                self.target, [register_table.make_key([gc.KeyTuple("$REGISTER_INDEX", idx)])], {"from_hw": True}
            )

            data_dict = next(resp)[0].to_dict()
            if reg_name + ".f1" in data_dict.keys():
                return data_dict[reg_name + ".f1"][pipe]
            else:
                return (data_dict[reg_name + ".first"][pipe], data_dict[reg_name + ".second"][pipe])

        if index is not None:
            return get_register_value_at_index(index)
        else:
            results = []
            for idx in range(register_table.info.size):
                results.append((idx, get_register_value_at_index(idx)))
            return results
