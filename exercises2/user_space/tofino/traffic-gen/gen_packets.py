# SPDX-License-Identifier: AGPL-3.0-or-later
from scapy.layers.inet import IP, UDP, Ether
from scapy.packet import Raw
from scapy_scion.layers.scion import SCION, HopField, InfoField, SCIONPath

from controller.ts_header import TimestampHdr


def create_scion_packet():
    pkt = Ether(dst="ff:ff:ff:ff:ff:ff", src="00:00:00:00:00:00")
    pkt = pkt / IP(dst="127.0.0.1", src="127.0.0.1")
    pkt = pkt / UDP() / SCION(
        Path=SCIONPath(
            Seg0Len=2,
            Seg1Len=2,
            InfoFields=[InfoField(), InfoField()],
            HopFields=[HopField(), HopField(), HopField(), HopField()]
        )
    )
    pkt = pkt / UDP() / TimestampHdr()
    pkt = pkt / Raw(34 * b"\x00")

    with open("ptf-tests/scion_packet.bin", "bw") as file:
        file.write(bytes(pkt))


if __name__ == "__main__":
    create_scion_packet()
