# bfrt_controller/helpers.py

"""
helpers.py

This module contains helper functions for configuring a P4 Tofino switch using the Controller API.

NOTE:
- These helpers assume a specific P4 program structure (e.g., the presence of Ingress.Dmac.broadcast_table).
- The port configuration reflects the topology and wiring at the University of Waterloo testbed.
- These functions are not portable across arbitrary P4 programs or hardware setups without modification.
"""

import logging
from bfrt_controller.controller import Controller
from bfrt_controller.bfrt_grpc.client import BfruntimeReadWriteRpcException

def setup_ports(c: Controller):
    logging.info("Setting up ports")
    ports = [
        (14, 0, 10, "none", "disable"),
        (14, 1, 10, "none", "disable"),
        (14, 2, 10, "none", "disable"),
        (33, 0, 10, "none", "disable"),
        (33, 2, 10, "none", "enable"),
        (33, 3, 10, "none", "enable"),
        (15, 0, 100, "none", "enable"),
        (16, 0, 100, "none", "enable"),
        (17, 0, 40, "none", "enable"),
        (18, 0, 40, "none", "enable"),
        (19, 0, 100, "none", "disable"),
        (20, 0, 100, "none", "disable"),
        (21, 0, 100, "none", "disable"),
        (22, 0, 100, "none", "disable"),
    ]
    c.port_manager.add_ports(port_list=ports)

def configure_multicast(c: Controller, port_list):
    logging.info(f"Configuring multicast for ports: {port_list}")
    multicast_config = {}
    mc_grp_id = 1
    for ingress_port in port_list:
        other_ports = [p for p in port_list if p != ingress_port]
        multicast_config[ingress_port] = (mc_grp_id, other_ports)
        mc_grp_id += 1

    entries = [
        ([("ig_intr_md.ingress_port", ingress)], "Ingress.Dmac.set_mcast_grp", [("mcast_grp", group)])
        for ingress, (group, _) in multicast_config.items()
    ]

    try:
        logging.info("Programming broadcast_table")
        c.program_table("Ingress.Dmac.broadcast_table", entries)
    except BfruntimeReadWriteRpcException as e:
        logging.error(f"broadcast_table error: {e}")

    try:
        logging.info("Programming multicast nodes and groups")
        for group_id, ports in multicast_config.values():
            c.add_multicast_node(group_id, ports)
            c.add_multicast_group(group_id, group_id)
    except BfruntimeReadWriteRpcException as e:
        logging.error(f"multicast group error: {e}")
