#!/usr/bin/env python3

"""
Dumps queue scheduling policy entries (sched_cfg and sched_shaping) from a Tofino switch and writes them to a file.
"""

import sys
sys.path.append("/home/n6saha/bfrt_controller")

import os
import logging
from bfrt_controller import Controller
from bfrt_controller.bfrt_grpc import client as gc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DEV_PORTS = [16]
QIDS = [0, 1, 2, 3]

def collect_sched_entries(controller, dev_port, qids):
    results = {}

    port_cfg = controller.bfrt_info.table_get("tf1.tm.port.cfg")
    key = port_cfg.make_key([gc.KeyTuple("dev_port", dev_port)])
    data, _ = next(port_cfg.entry_get(controller.target, [key], {"from_hw": True}))
    pg_id = data.to_dict()["pg_id"]
    qid_map = data.to_dict()["egress_qid_queues"]

    sched_cfg = controller.bfrt_info.table_get("tf1.tm.queue.sched_cfg")
    sched_shaping = controller.bfrt_info.table_get("tf1.tm.queue.sched_shaping")

    for qid in qids:
        if qid >= len(qid_map): continue
        physical_qid = qid_map[qid]

        entry_name = f"dev_port:{dev_port} qid:{qid} → pg_id:{pg_id} pg_queue:{physical_qid}"
        cfg_key = sched_cfg.make_key([
            gc.KeyTuple("pg_id", pg_id),
            gc.KeyTuple("pg_queue", physical_qid),
        ])
        cfg_data, _ = next(sched_cfg.entry_get(controller.target, [cfg_key], {"from_hw": True}))

        shaping_key = sched_shaping.make_key([
            gc.KeyTuple("pg_id", pg_id),
            gc.KeyTuple("pg_queue", physical_qid),
        ])
        shaping_data, _ = next(sched_shaping.entry_get(controller.target, [shaping_key], {"from_hw": True}))

        results[entry_name] = {
            "sched_cfg": cfg_data.to_dict(),
            "sched_shaping": shaping_data.to_dict(),
        }

    return results

def write_to_file(entries, output_file):
    with open(output_file, "w") as f:
        for key, value in entries.items():
            f.write(f"== {key} ==\n")
            f.write("[sched_cfg]\n")
            f.write(f"{value['sched_cfg']}\n")
            f.write("[sched_shaping]\n")
            f.write(f"{value['sched_shaping']}\n\n")

if __name__ == "__main__":
    c = Controller(pipe_id=0x0000)
    c.setup_tables(["tf1.tm.port.cfg", "tf1.tm.queue.sched_cfg", "tf1.tm.queue.sched_shaping"])

    all_results = {}
    for dev_port in DEV_PORTS:
        entries = collect_sched_entries(c, dev_port, QIDS)
        all_results.update(entries)

    output_path = os.path.join(os.path.dirname(__file__), "sched_entries.txt")
    write_to_file(all_results, output_path)

    logging.info(f"Scheduling policy entries written to {output_path}")
    c.tear_down()
