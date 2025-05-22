"""
Basic port and forwarding setup for our Tofino switch.

This script initializes ports and installs multicast groups and IPv4 forwarding rules
based on the physical topology of our testbed. The forwarding entries reflect our specific testbed configuration.

Functions:
    - setup_ports: Add and configure physical ports (front panel, internal, loopback)
    - configure_multicast: Create per-port multicast groups and populate the DMAC broadcast table
    - program_ipv4_forwarding: Install static IP-to-port forwarding entries
    - mac_to_byte_array / ip_to_byte_array: Utility functions for address formatting

"""

import os
import sys

sys.path.append("/home/n6saha/bfrt_controller")
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


from bfrt_controller import Controller
# from bfrt_controller.bfrt_grpc.client import BfruntimeReadWriteRpcException
from bfrt_controller.helpers import setup_ports, configure_multicast

def program_ipv4_forwarding(c: Controller):
    """Install static forwarding rules to route IPs to testbed ports."""
    logging.info("Programming ipv4_host_table")
    c.add_annotation("Ingress.Forward.ipv4_host_table", "hdr.ipv4.dst_addr", "ipv4")

    entries = [
        ([("hdr.ipv4.dst_addr", "192.168.44.101")], "Ingress.Forward.send", [("port", 66)]),
        ([("hdr.ipv4.dst_addr", "192.168.44.102")], "Ingress.Forward.send", [("port", 67)]),
        ([("hdr.ipv4.dst_addr", "192.168.44.13")], "Ingress.Forward.send", [("port", 64)]),
        ([("hdr.ipv4.dst_addr", "192.168.44.18")], "Ingress.Forward.send", [("port", 16)]),
        ([("hdr.ipv4.dst_addr", "192.168.44.203")], "Ingress.Forward.send", [("port", 17)]),
        ([("hdr.ipv4.dst_addr", "192.168.44.201")], "Ingress.Forward.send", [("port", 18)]),
    ]

    c.program_table("Ingress.Forward.ipv4_host_table", entries)


def mac_to_byte_array(mac_str):
    return bytearray(int(b, 16) for b in mac_str.split(":"))


def ip_to_byte_array(ip_str):
    return bytearray(int(b) for b in ip_str.split("."))


def main():
    c = Controller()

    # Set up ports
    setup_ports(c)

    # Set up required tables
    c.setup_tables(
        [
            "Ingress.Dmac.broadcast_table",
            "Ingress.Forward.ipv4_host_table",
        ]
    )

    # Configure multicast and forwarding rules
    configure_multicast(c, port_list=[16, 17, 18, 64, 66, 67])
    program_ipv4_forwarding(c)

    c.tear_down()


if __name__ == "__main__":
    main()
