"""
Programs token bucket meters on a Tofino switch using QFI-based traffic classes.

Each TEID is matched exactly in the meter table, and inherits shaping parameters
based on its QFI. This setup uses the RFC 2697 trTCM model.

Usage:
    python3 metering.py --map teid_qfi_map.json
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
sys.path.append("/home/n6saha/bfrt_controller")

from bfrt_controller.controller import Controller

QFI_METER_PARAMS = {
    1: {  # QFI 1 – Competitive real-time gaming (e.g., Battlegrounds)
        "CIR_KBPS": 1000,    # Very low committed rate (just control/predictive updates)
        "PIR_KBPS": 1500,   # Occasional bursts (e.g., explosions, team actions)
        "CBS_KBITS": 100,   # Short consistent bursts allowed
        "PBS_KBITS": 200    # Slightly higher peak bursts, prevent excessive queuing, keep latency low
    },
    2: {  # QFI 2 – Cloud gaming (e.g., GeForce Now)
        "CIR_KBPS": 3000,   # Moderate base rate to sustain 720p/1080p stream
        "PIR_KBPS": 6000,   # Can spike with action scenes or bitrate jumps
        "CBS_KBITS": 500,   # Tolerate regular short bursts
        "PBS_KBITS": 1000    # Allow larger spikes
    },
    3: {  # QFI 3 – Conferencing / interactive video (e.g., Zoom, Teams)
        "CIR_KBPS": 2000,   # High committed rate — real-time requirements
        "PIR_KBPS": 4000,   # Some headroom for quality adaptation
        "CBS_KBITS": 150,   # Conferencing uses small, frequent packets
        "PBS_KBITS": 300    # Limited extra buffering, reduce latency
    },
    5: {  # QFI 5 – Streaming (e.g., Netflix, YouTube Live)
        "CIR_KBPS": 4000,   # Moderate baseline (buffered delivery)
        "PIR_KBPS": 8000,   # Allows high-throughput initial buffering
        "CBS_KBITS": 1000,   # High buffer-friendly burst capacity
        "PBS_KBITS": 2000   # Supports fast-start or adaptive resolution shifts
    },
    9: {  # QFI 9 – Best-effort (web, email, social)
        "CIR_KBPS": 2000,   # Minimal guaranteed rate
        "PIR_KBPS": 8000,   # Allow bulk
        "CBS_KBITS": 500,   # Modest sustained burst
        "PBS_KBITS": 800    # Occasional heavy asset loads tolerated
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Configure QoS meters based on QFI mapping.")
    parser.add_argument("--teid-map", required=True, help="Path to TEID-to-QFI JSON mapping file")
    return parser.parse_args()


def load_teid_qfi_map(path):
    """Load and parse the TEID→QFI map from JSON."""
    if not os.path.exists(path):
        logging.error(f"Mapping file not found: {path}")
        sys.exit(1)
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in {path}: {e}")
        sys.exit(1)


def build_meter_entries(teid_map):
    """Generate meter table entries from TEID→QFI map."""
    entries = []
    teid_entries = teid_map["teids"] if "teids" in teid_map else teid_map

    for entry in teid_entries:
        teid = entry["teid"]
        qfi = entry.get("qfi")
        if qfi not in QFI_METER_PARAMS:
            logging.warning(f"Skipping TEID {teid} — unknown QFI {qfi}")
            continue

        params = QFI_METER_PARAMS[qfi]
        entries.append((
            [("hdr.gtpu.teid", teid)],
            "Ingress.QoSMeter.set_color",
            [
                ("$METER_SPEC_CIR_KBPS", params["CIR_KBPS"]),
                ("$METER_SPEC_PIR_KBPS", params["PIR_KBPS"]),
                ("$METER_SPEC_CBS_KBITS", params["CBS_KBITS"]),
                ("$METER_SPEC_PBS_KBITS", params["PBS_KBITS"]),
            ],
        ))
    return entries


def main():
    args = parse_args()
    teid_map = load_teid_qfi_map(args.teid_map)

    c = Controller()
    c.setup_tables(["Ingress.QoSMeter.meter_table"])
    c.add_annotation("Ingress.QoSMeter.meter_table", "hdr.gtpu.teid", "hex")

    meter_entries = build_meter_entries(teid_map)
    logging.info(f"Installing {len(meter_entries)} meter entries")
    c.program_table("Ingress.QoSMeter.meter_table", meter_entries)

    c.tear_down()


if __name__ == "__main__":
    main()
