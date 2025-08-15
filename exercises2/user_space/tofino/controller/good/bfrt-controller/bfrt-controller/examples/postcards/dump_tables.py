"""
Dumps P4 table entries from a Tofino switch and writes them to a file.
"""

import sys
sys.path.append("/home/n6saha/bfrt_controller")

import os
import logging
from bfrt_controller import Controller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def write_entries_to_file(entries, file_path):
    with open(file_path, "w") as file:
        for table_name, table_output in entries.items():
            file.write(f"== {table_name} ==\n")
            file.write(table_output)
            file.write("\n\n")


if __name__ == "__main__":
    controller = Controller()

    table_names = [
        "Ingress.Forward.ipv4_host_table",
        "Ingress.Dmac.dmac_table",
        "Ingress.Dmac.broadcast_table",
        "Egress.IntWatchList.int_watchlist_table",
        "Egress.IntPostcard.int_postcard_table",
        "Ingress.QoSMeter.meter_table",
        "Ingress.QoSMeter.qos_table",
    ]
    controller.setup_tables(table_names)

    all_entries = {}
    for table in table_names:
        all_entries[table] = controller.get_entries(table, print_entries=True)

    output_file = os.path.join(os.path.dirname(__file__), "table_entries.txt")
    write_entries_to_file(all_entries, output_file)

    controller.tear_down()

    logging.info(f"Entries written to {output_file}")