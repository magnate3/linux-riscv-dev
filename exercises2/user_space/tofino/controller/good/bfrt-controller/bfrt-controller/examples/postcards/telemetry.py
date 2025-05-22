"""
Configures INT watchlist, postcard generation, and mirror session for telemetry export on Tofino.

This script sets up in-band telemetry using the INT-XD (eXport Data) postcard model. It configures:
    - Mirror session to export telemetry
    - Egress.IntWatchList.int_watchlist_table to mark selected packets
    - Egress.IntPostcard.int_postcard_table to generate and export INT postcards
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

# # Add custom controller module to path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# controller_dir = os.path.join(current_dir, "../../../")
sys.path.append("/home/n6saha/bfrt_controller")

from bfrt_controller.controller import Controller


def mac_to_byte_array(mac_str):
    """Convert colon-separated MAC string to bytearray."""
    return bytearray(int(b, 16) for b in mac_str.split(":"))


def ip_to_byte_array(ip_str):
    """Convert dotted-decimal IP string to bytearray."""
    return bytearray(int(b) for b in ip_str.split("."))


def configure_mirror_session(controller):
    """Install egress mirror session to export truncated postcards."""
    session_id = 1
    egress_port = 17
    max_pkt_len = 64  # Ethernet (14) + IP (20) + UDP (8) + Postcard (22)

    logging.info(f"Installing mirror session (ID={session_id})")
    controller.add_mirror_entry(
        session_id=session_id, egress_port=egress_port, max_pkt_len=max_pkt_len, direction="EGRESS"
    )


def program_watchlist_table(controller):
    """Program INT watchlist table to mark packets for telemetry export."""
    logging.info("Programming int_watchlist_table")

    controller.add_annotation("Egress.IntWatchList.int_watchlist_table", "hdr.ipv4.dst_addr", "ipv4")
    controller.add_annotation("Egress.IntWatchList.int_watchlist_table", "hdr.ipv4.src_addr", "ipv4")

    entries = [
        (
            [
                ("hdr.ipv4.src_addr", "192.168.44.13"),
                ("hdr.ipv4.dst_addr", "192.168.44.18"),
                ("hdr.ipv4.protocol", 17),
                ("meta.src_port", 0, 0),
                ("meta.dst_port", 0, 0),
            ],
            "Egress.IntWatchList.mark_to_report",
            [],
        ),
        (
            [
                ("hdr.ipv4.src_addr", "192.168.44.201"),
                ("hdr.ipv4.dst_addr", "192.168.44.18"),
                ("hdr.ipv4.protocol", 17),
                ("meta.src_port", 0, 0),
                ("meta.dst_port", 0, 0),
            ],
            "Egress.IntWatchList.mark_to_report",
            [],
        ),
    ]
    controller.program_table("Egress.IntWatchList.int_watchlist_table", entries)


def program_postcard_table(controller):
    """Program postcard generation table with export metadata."""
    logging.info("Programming int_postcard_table")

    entries = [
        (
            [("meta.pkt_type", 2)],
            "Egress.IntPostcard.generate_postcard",
            [
                ("src_mac", mac_to_byte_array("00:1A:2B:3C:4D:5E")),
                ("src_ip", ip_to_byte_array("192.168.44.44")),
                ("collector_mac", mac_to_byte_array("e4:1d:2d:09:c7:50")),
                ("collector_ip", ip_to_byte_array("192.168.44.203")),
                ("collector_port", 4567),
            ],
        )
    ]
    controller.program_table("Egress.IntPostcard.int_postcard_table", entries)


def main():
    c = Controller()

    c.setup_tables(
        [
            "Egress.IntWatchList.int_watchlist_table",
            "Egress.IntPostcard.int_postcard_table",
        ]
    )

    configure_mirror_session(c)
    program_watchlist_table(c)
    program_postcard_table(c)

    c.tear_down()


if __name__ == "__main__":
    main()
