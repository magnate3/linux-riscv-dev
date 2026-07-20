#!/usr/bin/env python3

"""
bfrt_connect.py

Connects to a Tofino switch via gRPC using the BFRT interface.

Usage:
    python3 bfrt_connect.py [--verbose]
"""

import grpc
import argparse
import logging
from bfrt_controller.bfrt_grpc import bfruntime_pb2
from bfrt_controller.bfrt_grpc import bfruntime_pb2_grpc
from bfrt_controller.bfrt_grpc import client as gc

# --- Configuration ---
bfrt_ip = "localhost"
bfrt_port = 50052
device_id = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description="Connect to a BFRT gRPC server")
    parser.add_argument("--verbose", action="store_true", help="Print available BFRT tables")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    try:
        target = f"{bfrt_ip}:{bfrt_port}"
        logger.info(f"Connecting to BFRT at {target} (device_id={device_id})...")

        interface = gc.ClientInterface(target, client_id=0, device_id=device_id)
        bfrt_info = interface.bfrt_info_get()

        logger.info("Successfully connected.")
        if args.verbose:
            logger.info("Available BFRT tables:")
            for table in bfrt_info.table_name_list_get():
                logger.info(f"  - {table}")

    except grpc.RpcError as e:
        logger.error(f"gRPC connection failed: {e.details()} (code: {e.code().name})")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
