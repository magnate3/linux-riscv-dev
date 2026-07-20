# SPDX-License-Identifier: AGPL-3.0-or-later
import bfrt_grpc.client as gc
import ptf
import time
import scapy
from ptf.testutils import group
from scapy.layers.inet import Ether, UDP
from scapy_scion.layers.scion import SCION
from bfruntime_client_base_tests import BaseTest, BfRuntimeTest
from p4testutils.misc_utils import *
from ptf.mask import Mask

from controller.pktgen import StatelessTrafficGenerator
from controller.ts_header import TimestampHdr


PACKET_PATH = "ptf-tests/scion_packet.bin"

def _make_port(pipe: int, port: int) -> int:
    """Make port number from pipe ID and port within the pipe."""
    return (pipe << 7) | port


@group("default")
class TrafficGeneratorTest(BfRuntimeTest):
    client_id = 0
    p4_name = "pktgen"
    config = {
        "apps": {
            # Up to 16 applications (0-15)
            0: {
                "enabled": True,                 # Enable the application
                "packet": PACKET_PATH,           # Path to file containing packet data
                "pktgen_port": _make_port(1, 0), # Pktgen port to use (0-7 of each pipe)
                "timer_nanosec": 1e6,            # Delay after trigger
                "batch_count": 1,                # Number of batches
                "packets_per_batch": 1,          # Packets per batch
                "ibg": 0,                        # Inter-batch gap
                "ibg_jitter": 0,                 # Inter-batch gap jitter
                "ipg": 0,                        # Inter-packet gap
                "ipg_jitter": 0,                 # Inter-packet gap jitter
                "eg_port_group": 0               # Device egress ports (references eg_port_groups)
            },
            1: {
                "enabled": True,
                "packet": PACKET_PATH,
                "pktgen_port": _make_port(1, 2),
                "timer_nanosec": 1e6,
                "batch_count": 1,
                "packets_per_batch": 1,
                "ibg": 0,
                "ibg_jitter": 0,
                "ipg": 0,
                "ipg_jitter": 0,
                "eg_port_group": 1
            }
        },
        "eg_port_groups": {
            0: [
                _make_port(1, 8), # Replicate packet on multiple ports
                _make_port(1, 9)
            ],
            1: [_make_port(0, 8)] # Send to a single port in a different pipeline
        }
    }

    def setUp(self):
        BfRuntimeTest.setUp(self, self.client_id, self.p4_name)
        self.interface.clear_all_tables()
        self.controller = StatelessTrafficGenerator(self.interface, pipe=1)
        with open(PACKET_PATH, "rb") as file:
            self.pkt = Ether(file.read(1500))

    def tearDown(self):
        super().tearDown()

    def runTest(self):
        logger.info("Program tables")
        self.controller.program_tables(self.config)

        PKTGEN_HDR_SIZE = 6
        FCS_SIZE = 4

        try:
            exp = Mask(self.pkt)
            exp.set_do_not_care_scapy(UDP, "chksum")
            exp.set_do_not_care_scapy(TimestampHdr, "Timestamp")
            logger.info("Trigger generator")
            self.controller.trigger()
            verify_packet(self, exp, _make_port(1, 8))
            verify_packet(self, exp, _make_port(1, 9))
            verify_packet(self, exp, _make_port(0, 8))

            logger.info("Trigger generator again")
            self.controller.trigger(enable=False)
            self.controller.trigger()
            verify_packet(self, exp, _make_port(1, 8))
            verify_packet(self, exp, _make_port(1, 9))
            verify_packet(self, exp, _make_port(0, 8))

            logger.info("Get TX counters")
            tx_bytes, tx_pkts = self.controller.get_tx_counters()
            self.assertEqual(tx_bytes, 4 * (len(self.pkt) + PKTGEN_HDR_SIZE + FCS_SIZE))
            self.assertEqual(tx_pkts, 4)

            logger.info("Get timestamps")
            send_packet(self, _make_port(1, 8), self.pkt)
            time.sleep(0.5)
            (tx_first, tx_last), (rx_first, rx_last) = self.controller.get_timestamps()
            self.assertNotEqual(tx_first, -1)
            self.assertNotEqual(tx_last, -1)
            self.assertNotEqual(rx_first, -1)
            self.assertNotEqual(rx_last, -1)
            self.assertLessEqual((tx_last - tx_first) % 2**48, 1000000000)
            self.assertEqual(rx_first, rx_last)

            send_packet(self, _make_port(1, 8), self.pkt)
            time.sleep(0.5)
            (tx_first, tx_last), (rx_first, rx_last) = self.controller.get_timestamps()
            self.assertNotEqual(rx_first, rx_last)

            logger.info("Reset timestamps")
            self.controller.reset_timestamps()
            (tx_first, tx_last), (rx_first, rx_last) = self.controller.get_timestamps()
            self.assertEqual(tx_first, -1)
            self.assertEqual(tx_last, -1)
            self.assertEqual(rx_first, -1)
            self.assertEqual(rx_last, -1)

            logger.info("Read histogram")
            for i in range(6):
                send_packet(self, _make_port(1, 8), self.pkt)
                time.sleep(0.5) # Tofino model drops packets if rate is too high

            _, bytes, packets = self.controller.get_histogram()
            ovfl_bytes, ovfl_pkts = self.controller.get_hist_overflow()
            self.assertEqual(sum(packets) + ovfl_pkts, 8)
            self.assertEqual(sum(bytes) + ovfl_bytes, 8 * (len(self.pkt) + FCS_SIZE))

            logger.info("Reset histogram")
            self.controller.reset_histogram()
            _, bytes, packets = self.controller.get_histogram()
            ovfl_bytes, ovfl_pkts = self.controller.get_hist_overflow()
            self.assertEqual(sum(packets) + ovfl_pkts, 0)
            self.assertEqual(sum(bytes) + ovfl_bytes, 0)

        finally:
            logger.info("Cleanup")
            self.controller.program_tables(self.config, cleanup=True)


@group("rewrite")
class RewriteTest(BfRuntimeTest):
    client_id = 0
    p4_name = "pktgen"

    config = {
        "apps": {
            0: {
                "enabled": True,
                "packet": PACKET_PATH,
                "pktgen_port": _make_port(1, 0),
                "timer_nanosec": 1e6,
                "batch_count": 1,
                "packets_per_batch": 4,
                "ibg": 0,
                "ibg_jitter": 0,
                "ipg": 0,
                "ipg_jitter": 0,
                "eg_port_group": 0
            }
        },
        "eg_port_groups": {
            0: [_make_port(0, 8)]
        },
        "port_map": [
            {
                # Key
                "app": {"value": 0, "mask": 0},
                "batch": {"value": 0, "mask": 0},
                "packet": {"value": 0, "mask": 0x3},
                # Action
                "action": "set", # "set" or "random"
                "src": 50000,    # Source port
                "dst": 50000     # Destination port
            },
            {
                "app": {"value": 0, "mask": 0},
                "batch": {"value": 0, "mask": 0},
                "packet": {"value": 1, "mask": 0x3},
                "action": "set",
                "src": 50001,
                "dst": 50001
            },
            {
                "app": {"value": 0, "mask": 0},
                "batch": {"value": 0, "mask": 0},
                "packet": {"value": 2, "mask": 0x3},
                "action": "set",
                "src": 50002,
                "dst": 50002
            },
            {
                "app": {"value": 0, "mask": 0},
                "batch": {"value": 0, "mask": 0},
                "packet": {"value": 3, "mask": 0x3},
                "action": "set",
                "src": 50003,
                "dst": 50003
            }
        ],
        "flow_map": [
            {
                # Key
                "app": {"value": 0, "mask": 0xf},
                "batch": {"value": 0, "mask": 0},
                "packet": {"value": 0, "mask": 0xffffffff},
                # Action
                "action": "set", # "set" or "random"
                "flow_id": 0xdead
            },
            {
                "app": {"value": 0, "mask": 0xf},
                "batch": {"value": 0, "mask": 0},
                "packet": {"value": 0, "mask": 0},
                "action": "random"
            }
        ]
    }

    def setUp(self):
        BfRuntimeTest.setUp(self, self.client_id, self.p4_name)
        self.interface.clear_all_tables()
        self.controller = StatelessTrafficGenerator(self.interface, pipe=1)
        with open(PACKET_PATH, "rb") as file:
            self.pkt = Ether(file.read(1500))

    def tearDown(self):
        super().tearDown()

    def runTest(self):
        logger.info("Program tables")
        self.controller.program_tables(self.config)

        try:
            exp = Mask(self.pkt)
            exp.set_do_not_care_scapy(UDP, "chksum")
            exp.set_do_not_care_scapy(SCION, "FlowID")
            exp.set_do_not_care_scapy(TimestampHdr, "Timestamp")
            logger.info("Trigger generator")
            self.controller.trigger()
            exp.exp_pkt.getlayer("UDP", 1).sport = 50000
            exp.exp_pkt.getlayer("UDP", 1).dport = 50000
            verify_packet(self, exp, _make_port(0, 8))
            exp.exp_pkt.getlayer("UDP", 1).sport = 50001
            exp.exp_pkt.getlayer("UDP", 1).dport = 50001
            verify_packet(self, exp, _make_port(0, 8))
            exp.exp_pkt.getlayer("UDP", 1).sport = 50002
            exp.exp_pkt.getlayer("UDP", 1).dport = 50002
            verify_packet(self, exp, _make_port(0, 8))
            exp.exp_pkt.getlayer("UDP", 1).sport = 50003
            exp.exp_pkt.getlayer("UDP", 1).dport = 50003
            verify_packet(self, exp, _make_port(0, 8))

        finally:
            logger.info("Cleanup")
            self.controller.program_tables(self.config, cleanup=True)
