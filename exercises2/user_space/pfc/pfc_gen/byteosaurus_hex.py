#!/usr/bin/python3

#################################################################################################################
# Yet another packet generator based on Scapy
# For installation and usage see README https://github.com/UchihaItachiSama/scapy-cli/blob/main/README.md
#################################################################################################################

#################################################################################################################
# Libraries
import codecs
import sys
from scapy.all import *
from scapy.error import Scapy_Exception
from collections import Counter
from tabulate import tabulate
import gc
import multiprocessing
from os import urandom
from random import randint
from scapy.contrib.igmp import *
from scapy.contrib.igmpv3 import *
from scapy.contrib.mac_control import *
from scapy.contrib.mpls import *
import re
import logging


#################################################################################################################
def requires(module):
    req_arr = {
        "ICMP": [
            "Source MAC (de:ad:be:ef:ca:fe)", "Destination MAC", "Source IP",
            "Destination IP", "TTL", "Tag (y/n)"
        ],
        "ARP": [
            "Source MAC (de:ad:be:ef:ca:fe)", "Destination MAC", "Sender MAC",
            "Sender IP", "Target MAC", "Target IP", "Tag (y/n)"
        ],
        "IGMP": [
            "Sender MAC (de:ad:be:ef:ca:fe)", "Sender IP", "Multicast Address",
            "Source Address", "Tag (y/n)"
        ],
        "PCAP": [
            "Source MAC (de:ad:be:ef:ca:fe)", "Destination MAC", "Source IP",
            "Destination IP"
        ],
        "MCAST": [
            "Source MAC (de:ad:be:ef:ca:fe)", "Source IP", "Destination IP",
            "UDP Source Port", "UDP Destination Port", "Tag (y/n)"
        ],
        "UDP": [
            "Source MAC (de:ad:be:ef:ca:fe)", "Destination MAC", "Source IP",
            "Destination IP", "UDP Source Port", "UDP Destination Port",
            "Tag (y/n)"
        ],
        "TCP": [
            "Source MAC (de:ad:be:ef:ca:fe)", "Destination MAC", "Source IP",
            "Destination IP", "TCP Source Port", "TCP Destination Port",
            "Tag (y/n)"
        ],
        "VXLAN": [
            "Outer Source MAC (de:ad:be:af:ca:fe)", "Outer Destination MAC",
            "Outer Source IP", "Outer Destination IP", "Outer UDP Source Port",
            "Outer UDP Destination Port (default 4789)", "VNI"
        ],
        "LLFC": [
            "Source MAC (de:ad:be:ef:ca:fe)", "Time in Quanta (0-65535)"
        ],
        "PFC": [
            "Source MAC (de:ad:be:ef:ca:fe)", "Enable Pause for Class (C0-C7)",
            "Time in Quanta for Class (0-65535)"
        ],
        "MPLS": [
            "Outer Source MAC (de:ad:be:af:ca:fe)", "Outer Destination MAC",
            "Labels in comma delimited form (Top to Bottom)", "Control Word (y/n)"
        ],
        "common": ["Count (c for continous)", "Source Interface"]
    }
    return req_arr[module], req_arr["common"]


#################################################################################################################
def build_icmp():
    # Gettting the input parameters
    icmp_pkt = None
    input_param, common_param = requires("ICMP")
    fuzzy = (input("Random ICMP Packet? (y/n) > ").strip()).lower()
    if fuzzy == "y":
        icmp_type = (input("ICMP Type (req/reply) > ").strip()).lower()
        inputs = []
        # Common parameters
        for i in range(0, len(common_param)):
            inputs.insert(i, input("{} > ".format(common_param[i])))
        icmp_pkt = icmp_packet(fuzzy, 'ICMP', icmp_type, inputs)
        if icmp_pkt != None:
            logger.info("ICMP packet built")
            icmp_pkt.show()
            return icmp_pkt, inputs[0], inputs[1]
        else:
            return None
    elif fuzzy == "n":
        icmp_type = (input("ICMP Type (req/reply) > ").strip()).lower()
        # Getting input parameters
        inputs = []
        dot1q_prio = []
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid choice, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))
        # Based on the provided VLAN tag return tagged/untagged icmp packet
        if not (inputs[5]):
            icmp_pkt = icmp_packet(fuzzy, 'ICMP', icmp_type, inputs)
        else:
            vlans = (inputs[5]).strip().split(",")
            cos = (dot1q_prio[0]).strip().split(",")
            try:
                vlans = [int(i) for i in vlans]
            except ValueError:
                logger.critical(
                    "Invalid vlan id'{}' Expected integer".format(vlans))
                logger.critical(ValueError, exc_info=True)
                return None
            icmp_pkt = icmp_packet(fuzzy, 'ICMP', icmp_type, inputs)
            cos = validate_cos(cos, vlans)
            if icmp_pkt != None and cos != None:
                icmp_pkt = add_vlan(icmp_pkt, vlans, cos)
            else:
                return None
        if icmp_pkt != None:
            logger.info("ICMP Packet built")
            icmp_pkt.show()
            return icmp_pkt, inputs[6], inputs[7]
        else:
            return None
    else:
        logger.critical(
            "Invalid input '{}' Expected string (y/n)".format(fuzzy))
        return None


#################################################################################################################
def build_arp():
    # Getting input parameters
    input_param, common_param = requires("ARP")
    inputs = [None] * len(input_param)
    arp_pkt = None
    fuzzy = (input("Generate random ARP Packet? (y/n) > ").strip()).lower()
    if fuzzy == "y":
        arp_type = (input("ARP Type (req/resp) > ").strip()).lower()
        inputs = []
        # Common parameters
        for i in range(0, len(common_param)):
            inputs.insert(i, input("{} > ".format(common_param[i])))
        arp_pkt = arp_packet(fuzzy, 'ARP', arp_type, inputs)
        if arp_pkt != None:
            logger.info("ARP packet built")
            arp_pkt.show()
            return arp_pkt, inputs[0], inputs[1]
        else:
            return None
    elif fuzzy == "n":
        arp_type = (input("ARP Type (req/resp) > ").strip()).lower()
        inputs = []
        dot1q_prio = []
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid choice, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))
        if not (inputs[6]):
            arp_pkt = arp_packet(fuzzy, 'ARP', arp_type, inputs)
        else:
            vlans = inputs[6].split(",")
            cos = (dot1q_prio[0]).strip().split(",")
            try:
                vlans = [int(i) for i in vlans]
            except ValueError:
                logger.critical(
                    "Invalid vlan id'{}' Expected integer".format(vlans))
                logger.critical(ValueError, exc_info=True)
                return None
            arp_pkt = arp_packet(fuzzy, 'ARP', arp_type, inputs)
            cos = validate_cos(cos, vlans)
            if arp_pkt != None and cos != None:
                arp_pkt = add_vlan(arp_pkt, vlans, cos)
            else:
                return None
        if arp_pkt != None:
            logger.info("ARP packet built")
            arp_pkt.show()
            return arp_pkt, inputs[7], inputs[8]
        else:
            return None
    else:
        logger.critical(
            "Invalid input '{}' Expected string (y/n)".format(fuzzy))
        return None


#################################################################################################################
def build_group_records(msg_type):
    final_grp_arr = []
    if msg_type == "M_R":
        try:
            num_records = int(input("Number of group records > ").strip())
            for index in range(0, num_records):
                print("\nGroup record {}:".format(index + 1))
                mcast_addr = input("Multicast Address > ").strip()
                record_type = int(
                    input(
                        "\nRecord Type:\n\n1 -- Mode Is Include\n2 -- Mode Is Exclude\n3 -- Change To Include Mode\n4 -- Change To Exclude Mode\n5 -- Allow New Sources\n6 -- Block Old Sources\n\nEnter your choice (1-6) > "
                    ).strip())
                src_addrs = input("Source addresses (IP1,IP2) > ").split(",")
                if len(src_addrs) == 1 and src_addrs[0] == '':
                    gr1 = IGMPv3gr(rtype=4, maddr=mcast_addr, numsrc=0)
                    final_grp_arr.append(gr1)
                else:
                    gr1 = IGMPv3gr(rtype=record_type,
                                   maddr=mcast_addr,
                                   numsrc=len(src_addrs),
                                   srcaddrs=src_addrs)
                    final_grp_arr.append(gr1)
        except ValueError:
            logger.critical(
                "Invalid input for num_records:'{}'. Expecting integer value".
                format(num_records))
            logger.critical(ValueError, exc_info=True)
            return None
    elif msg_type == "L_G":
        try:
            num_records = int(input("Number of group records > ").strip())
            for index in range(0, num_records):
                print("\nGroup record {}:".format(index + 1))
                mcast_addr = input("Multicast Address > ").strip()
                #record_type = int(input("\nRecord Type \n{1: 'Mode Is Include'\n2: 'Mode Is Exclude'\n3: 'Change To Include Mode'\n4: 'Change To Exclude Mode'\n5: 'Allow New Sources'\n6: 'Block Old Sources'}\n Input > ").strip())
                src_addrs = input("Source addresses (IP1,IP2) > ").split(",")
                if len(src_addrs) == 1 and src_addrs[0] == '':
                    gr1 = IGMPv3gr(rtype=3, maddr=mcast_addr, numsrc=0)
                    final_grp_arr.append(gr1)
                else:
                    gr1 = IGMPv3gr(rtype=6,
                                   maddr=mcast_addr,
                                   numsrc=len(src_addrs),
                                   srcaddrs=src_addrs)
                    final_grp_arr.append(gr1)
        except ValueError:
            logger.critical(
                "Invalid input for num_records:'{}'. Expecting integer value".
                format(num_records))
            logger.critical(ValueError, exc_info=True)
            return None
    return final_grp_arr


#################################################################################################################
def build_igmp(msg_type, version):
    if msg_type == "M_Q_G" and (version == "v1" or version == "v2" or version == "v3"):
        # Gettting the input parameters
        input_param, common_param = requires("IGMP")
        del input_param[2:4]
        inputs = []
        dot1q_prio = []
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid input, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))
        # Building IGMPv1 Membership Query
        if version == "v1":
            if not (inputs[2]):
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x11, gaddr="0.0.0.0", mrcode=0)
                if not (p[IGMP].igmpize()):
                    logger.critical("Failed building IGMPv1 Membership Query")
                    return None
            else:
                vlans = inputs[2].split(",")
                cos = (dot1q_prio[0]).strip().split(",")
                try:
                    vlans = [int(i) for i in vlans]
                except ValueError:
                    logger.critical(
                        "Invalid vlan '{}' Expected integer".format(vlans))
                    logger.critical(ValueError, exc_info=True)
                    return None
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x11, gaddr="0.0.0.0", mrcode=0)
                if not (p[IGMP].igmpize()):
                    logger.critical("Failed building IGMPv1 Membership Query")
                    return None
                cos = validate_cos(cos, vlans)
                if cos != None:
                    p = add_vlan(p, vlans, cos)
                else:
                    return None
            logger.info("IGMPv1 Membership Query Built")
            p.show()
            return p, inputs[3], inputs[4]
        # Building IGMPv2 Membership Query, General
        elif version == "v2":
            if not (inputs[2]):
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x11, gaddr="0.0.0.0")
                if not (p[IGMP].igmpize()):
                    logger.critical("Failed building IGMPv2 Membership Query")
                    return None
            else:
                vlans = inputs[2].split(",")
                cos = (dot1q_prio[0]).strip().split(",")
                try:
                    vlans = [int(i) for i in vlans]
                except ValueError:
                    logger.critical(
                        "Invalid vlan '{}' Expected integer".format(vlans))
                    logger.critical(ValueError, exc_info=True)
                    return None
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x11, gaddr="0.0.0.0")
                if not (p[IGMP].igmpize()):
                    logger.critical("Failed building IGMPv2 Membership Query")
                    return None
                cos = validate_cos(cos, vlans)
                if cos != None:
                    p = add_vlan(p, vlans, cos)
                else:
                    return None
            logger.info("IGMPv2 Membership Query Built")
            p.show()
            return p, inputs[3], inputs[4]
        # Building IGMPv3 Membership Query, General
        elif version == "v3":
            if not (inputs[2]):
                p = Ether(src=inputs[0]) / IP(
                    src=inputs[1]) / IGMPv3() / IGMPv3mq(gaddr="0.0.0.0")
                if not (p[IGMPv3].igmpize()):
                    logger.critical("Failed building IGMPv3 Membership Query")
                    return None
            else:
                vlans = inputs[2].split(",")
                cos = (dot1q_prio[0]).strip().split(",")
                try:
                    vlans = [int(i) for i in vlans]
                except ValueError:
                    logger.critical(
                        "Invalid vlan '{}' Expected integer".format(vlans))
                    logger.critical(ValueError, exc_info=True)
                    return None
                p = Ether(src=inputs[0]) / IP(
                    src=inputs[1]) / IGMPv3() / IGMPv3mq(gaddr="0.0.0.0")
                if not (p[IGMPv3].igmpize()):
                    logger.critical("Failed building IGMPv3 Membership Query")
                    return None
                cos = validate_cos(cos, vlans)
                if cos != None:
                    p = add_vlan(p, vlans, cos)
                else:
                    return None
            logger.info("IGMPv3 Membership Query Built")
            p.show()
            return p, inputs[3], inputs[4]
        else:
            logger.critical(
                "Invalid version: '{}' Expected value (v1/v2/v3)".format(
                    version))
            return None
    elif msg_type == "M_Q_GS" and (version == "v2" or version == "v3"):
        # Gettting the input parameters
        input_param, common_param = requires("IGMP")
        del input_param[3:4]
        inputs = []
        dot1q_prio = []
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid input, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))

        # Building IGMP Membership query, Group specific
        if version == "v2":
            if not (inputs[3]):
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x11, gaddr=inputs[2])
                if not (p[IGMP].igmpize()):
                    logger.critical(
                        "Failed building IGMPv2 Membership Query, Group specific"
                    )
                    return None
            else:
                vlans = inputs[3].split(",")
                cos = (dot1q_prio[0]).strip().split(",")
                try:
                    vlans = [int(i) for i in vlans]
                except ValueError:
                    logger.critical(
                        "Invalid vlan '{}' Expected integer".format(vlans))
                    logger.critical(ValueError, exc_info=True)
                    return None
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x11, gaddr=inputs[2])
                if not (p[IGMP].igmpize()):
                    logger.critical(
                        "Failed building IGMPv2 Membership Query, Group specific"
                    )
                    return None
                cos = validate_cos(cos, vlans)
                if cos != None:
                    p = add_vlan(p, vlans, cos)
                else:
                    return None
            logger.info("IGMPv2 Membership Query, Group specific")
            p.show()
            return p, inputs[4], inputs[5]
        elif version == "v3":
            if not (inputs[3]):
                p = Ether(src=inputs[0]) / IP(
                    src=inputs[1]) / IGMPv3() / IGMPv3mq(gaddr=inputs[2])
                if not (p[IGMPv3].igmpize()):
                    logger.critical(
                        "Failed building IGMPv3 Membership Query, Group specific"
                    )
                    return None
            else:
                vlans = inputs[3].split(",")
                cos = (dot1q_prio[0]).strip().split(",")
                try:
                    vlans = [int(i) for i in vlans]
                except ValueError:
                    logger.critical(
                        "Invalid vlan '{}' Expected integer".format(vlans))
                    logger.critical(ValueError, exc_info=True)
                    return None
                p = Ether(src=inputs[0]) / IP(
                    src=inputs[1]) / IGMPv3() / IGMPv3mq(gaddr=inputs[2])
                if not (p[IGMPv3].igmpize()):
                    logger.critical(
                        "Failed building IGMPv3 Membership Query, Group specific"
                    )
                    return None
                cos = validate_cos(cos, vlans)
                if cos != None:
                    p = add_vlan(p, vlans, cos)
                else:
                    return None
            logger.info("IGMPv3 Membership Query, Group specific")
            p.show()
            return p, inputs[4], inputs[5]
        else:
            logger.critical(
                "Invalid version: '{}' Expected value (v1/v2/v3)".format(
                    version))
            return None
    elif msg_type == "M_Q_G_SS" and (version == "v3"):
        # Gettting the input parameters
        input_param, common_param = requires("IGMP")
        inputs = []
        dot1q_prio = []
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid input, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))

        # Building IGMP Membership query, Group and Source specific
        if version == "v3":
            src_addrs = inputs[3].split(",")
            for index in range(len(src_addrs)):
                src_addrs[index] = src_addrs[index].strip()

            if not (inputs[4]):
                p = Ether(src=inputs[0]) / IP(
                    src=inputs[1]) / IGMPv3() / IGMPv3mq(gaddr=inputs[2],
                                                         numsrc=len(src_addrs),
                                                         srcaddrs=src_addrs)
                if not (p[IGMPv3].igmpize()):
                    logger.critical(
                        "Failed building IGMPv3 Membership Query, Group & Source specific"
                    )
                    return None
            else:
                vlans = inputs[4].split(",")
                cos = (dot1q_prio[0]).strip().split(",")
                try:
                    vlans = [int(i) for i in vlans]
                except ValueError:
                    logger.critical(
                        "Invalid vlan '{}' Expected integer".format(vlans))
                    logger.critical(ValueError, exc_info=True)
                    return None
                p = Ether(src=inputs[0]) / IP(
                    src=inputs[1]) / IGMPv3() / IGMPv3mq(gaddr=inputs[2],
                                                         numsrc=len(src_addrs),
                                                         srcaddrs=src_addrs)
                if not (p[IGMPv3].igmpize()):
                    logger.critical(
                        "Failed building IGMPv3 Membership Query, Group & Source specific"
                    )
                    return None
                cos = validate_cos(cos, vlans)
                if cos != None:
                    p = add_vlan(p, vlans, cos)
                else:
                    return None
            logger.info("IGMPv3 Membership Query, Group & Source specific")
            p.show()
            return p, inputs[5], inputs[6]
        else:
            logger.critical(
                "Invalid version: '{}' Expected value (v1/v2/v3)".format(
                    version))
            return None
    elif msg_type == "M_R" and (version == "v1" or version == "v2"):
        # Gettting the input parameters
        input_param, common_param = requires("IGMP")
        del input_param[3:4]
        inputs = []
        dot1q_prio = []
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid input, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))

        # Building IGMPv1 Membership Report
        if version == "v1":
            if not (inputs[3]):
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x12, gaddr=inputs[2])
                if not (p[IGMP].igmpize()):
                    logger.critical("Failed building IGMPv1 Membership Report")
                    return None
            else:
                vlans = inputs[3].split(",")
                cos = (dot1q_prio[0]).strip().split(",")
                try:
                    vlans = [int(i) for i in vlans]
                except ValueError:
                    logger.critical(
                        "Invalid vlan '{}' Expected integer".format(vlans))
                    logger.critical(ValueError, exc_info=True)
                    return None
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x12, gaddr=inputs[2])
                if not (p[IGMP].igmpize()):
                    logger.critical("Failed building IGMPv1 Membership Report")
                    return None
                cos = validate_cos(cos, vlans)
                if cos != None:
                    p = add_vlan(p, vlans, cos)
                else:
                    return None
            logger.info("IGMPv1 Membership Report")
            p.show()
            return p, inputs[4], inputs[5]
        # Building IGMPv2 Membership Report
        elif version == "v2":
            if not (inputs[3]):
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x16, gaddr=inputs[2])
                if not (p[IGMP].igmpize()):
                    logger.critical("Failed building IGMPv2 Membership Report")
                    return None
            else:
                vlans = inputs[3].split(",")
                cos = (dot1q_prio[0]).strip().split(",")
                try:
                    vlans = [int(i) for i in vlans]
                except ValueError:
                    logger.critical(
                        "Invalid vlan '{}' Expected integer".format(vlans))
                    logger.critical(ValueError, exc_info=True)
                    return None
                p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(
                    type=0x16, gaddr=inputs[2])
                if not (p[IGMP].igmpize()):
                    logger.critical("Failed building IGMPv2 Membership Report")
                    return None
                cos = validate_cos(cos, vlans)
                if cos != None:
                    p = add_vlan(p, vlans, cos)
                else:
                    return None
            logger.info("IGMPv2 Membership Report")
            p.show()
            return p, inputs[4], inputs[5]
        else:
            logger.critical(
                "Invalid version: '{}' Expected value (v1/v2/v3)".format(
                    version))
            return None
    elif msg_type == "M_R" and version == "v3":
        # Gettting the input parameters
        input_param, common_param = requires("IGMP")
        inputs = []
        dot1q_prio = []
        del input_param[2:4]
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid input, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Build group records
        group_rec = build_group_records("M_R")
        if group_rec == None:
            return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))

        # Building the IGMPv3 Membership Record with group records
        if not (inputs[2]):
            p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMPv3() / IGMPv3mr(
                numgrp=len(group_rec), records=group_rec)
            if not (p[IGMPv3].igmpize()):
                logger.critical("Failed building IGMPv3 Membership Report")
                return None
        else:
            vlans = inputs[2].split(",")
            cos = (dot1q_prio[0]).strip().split(",")
            try:
                vlans = [int(i) for i in vlans]
            except ValueError:
                logger.critical(
                    "Invalid vlan '{}' Expected integer".format(vlans))
                logger.critical(ValueError, exc_info=True)
                return None
            p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMPv3() / IGMPv3mr(
                numgrp=len(group_rec), records=group_rec)
            if not (p[IGMPv3].igmpize()):
                logger.critical(
                    "Failed building IGMPv3 Membership Query, Group & Source specific"
                )
                return None
            cos = validate_cos(cos, vlans)
            if cos != None:
                p = add_vlan(p, vlans, cos)
            else:
                return None
        logger.info("IGMPv3 Membership Report")
        p.show()
        return p, inputs[3], inputs[4]
    elif msg_type == "L_G" and version == "v2":
        # Gettting the input parameters
        input_param, common_param = requires("IGMP")
        inputs = []
        dot1q_prio = []
        del input_param[3:4]
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid input, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))
        #Building the IGMPv2 Leave Message
        if not (inputs[3]):
            p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(type=0x17,
                                                                gaddr=inputs[2])
            if not (p[IGMP].igmpize()):
                logger.critical("Failed building IGMPv2 Leave Message")
                return None
        else:
            vlans = inputs[3].split(",")
            cos = (dot1q_prio[0]).strip().split(",")
            try:
                vlans = [int(i) for i in vlans]
            except ValueError:
                logger.critical(
                    "Invalid vlan '{}' Expected integer".format(vlans))
                logger.critical(ValueError, exc_info=True)
                return None
            p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMP(type=0x17,
                                                                gaddr=inputs[2])
            if not (p[IGMP].igmpize()):
                logger.critical("Failed building IGMPv2 Leave Message")
                return None
            cos = validate_cos(cos, vlans)
            if cos != None:
                p = add_vlan(p, vlans, cos)
            else:
                return None
        logger.info("IGMPv2 Leave Message")
        p.show()
        return p, inputs[4], inputs[5]
    elif msg_type == "L_G" and version == "v3":
        # Gettting the input parameters
        input_param, common_param = requires("IGMP")
        inputs = []
        dot1q_prio = []
        del input_param[2:4]
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid input, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Build group records
        group_rec = build_group_records("L_G")
        if group_rec == None:
            return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))

        # Building the IGMPv3 Leave message with group records
        if not (inputs[2]):
            p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMPv3() / IGMPv3mr(
                numgrp=len(group_rec), records=group_rec)
            if not (p[IGMPv3].igmpize()):
                logger.critical("Failed building IGMPv3 Leave Message")
                return None
        else:
            vlans = inputs[2].split(",")
            cos = (dot1q_prio[0]).strip().split(",")
            try:
                vlans = [int(i) for i in vlans]
            except ValueError:
                logger.critical(
                    "Invalid vlan '{}' Expected integer".format(vlans))
                logger.critical(ValueError, exc_info=True)
                return None
            p = Ether(src=inputs[0]) / IP(src=inputs[1]) / IGMPv3() / IGMPv3mr(
                numgrp=len(group_rec), records=group_rec)
            if not (p[IGMPv3].igmpize()):
                logger.critical("Failed building IGMPv3 Leave Message")
                return None
            cos = validate_cos(cos, vlans)
            if cos != None:
                p = add_vlan(p, vlans, cos)
            else:
                return None
        logger.info("IGMPv3 Leave Message")
        p.show()
        return p, inputs[3], inputs[4]
    else:
        logger.critical(
            "Invalid msg_type: '{}' or version: '{}' provided".format(
                msg_type, version))
        return None


#################################################################################################################
def igmp():
    # Getting IGMP version and message types
    igmp_ver = input("IGMP Version (v1/v2/v3) > ").strip().lower()
    if igmp_ver == "v1":
        try:
            msg_type = int(
                input(
                    "\nIGMP Message Type:\n\n1 -- {}\n2 -- {}\n\nEnter your choice (1-2) > "
                    .format("Membership Query", "Membership Report")).strip())
            if msg_type == 1:  #Membership Query
                return build_igmp("M_Q_G", igmp_ver)
            elif msg_type == 2:  #Membership Report
                return build_igmp("M_R", igmp_ver)
            else:
                logger.critical(
                    "Invalid msg_type: '{}' Expected integer (1-2)".format(
                        msg_type))
                return None
        except ValueError:
            logger.critical("Invalid input, expected integer (1-2).")
            return None
    elif igmp_ver == "v2":
        try:
            msg_type = int(
                input(
                    "\nIGMP Message Type:\n\n1 -- {}\n2 -- {}\n3 -- {}\n4 -- {}\n\nEnter your choice (1-4) > "
                    .format("Membership Query, General",
                            "Membership Query, Group-Specific",
                            "Membership Report", "Leave Group")).strip())
            if msg_type == 1:  #Membership Query, General
                return build_igmp("M_Q_G", igmp_ver)
            elif msg_type == 2:  #Membership Query, Group-Specific
                return build_igmp("M_Q_GS", igmp_ver)
            elif msg_type == 3:  #Membership Report
                return build_igmp("M_R", igmp_ver)
            elif msg_type == 4:  #Leave Group
                return build_igmp("L_G", igmp_ver)
            else:
                logger.critical(
                    "Invalid msg_type: '{}' Expected integer (1-4)".format(
                        msg_type))
                return None
        except ValueError:
            logger.critical("Invalid input, expected integer (1-4).")
            return None
    elif igmp_ver == "v3":
        try:
            msg_type = int(
                input(
                    "\nIGMP Message Type:\n\n1 -- {}\n2 -- {}\n3 -- {}\n4 -- {}\n5 -- {}\n\nEnter your choice (1-5) > "
                    .format("Membership Query, General",
                            "Membership Query, Group-Specific",
                            "Membership Query, Group-and-Source-Specific",
                            "Membership Report", "Leave Group")).strip())
            if msg_type == 1:  #Membership Query, General
                return build_igmp("M_Q_G", igmp_ver)
            elif msg_type == 2:  #Membership Query, Group-Specific
                return build_igmp("M_Q_GS", igmp_ver)
            elif msg_type == 3:  #Membership Query, Group-and-Source-Specific
                return build_igmp("M_Q_G_SS", igmp_ver)
            elif msg_type == 4:  #Membership Report
                return build_igmp("M_R", igmp_ver)
            elif msg_type == 5:  #Leave Group
                return build_igmp("L_G", igmp_ver)
            else:
                logger.critical(
                    "Invalid msg_type: '{}' Expected integer (1-5)".format(
                        msg_type))
                return None
        except ValueError:
            logger.critical("Invalid input, expected integer (1-5).")
            return None
    else:
        logger.critical(
            "Invalid igmp_ver: '{}' Expected string (v1/v2/v3)".format(
                igmp_ver))
        return None


#################################################################################################################
def convert_multicast_ip_to_mac(ip_address):
    try:
        ip_binary = socket.inet_pton(socket.AF_INET, ip_address)
        ip_bit_string = ''.join(['{0:08b}'.format(x) for x in ip_binary])
    except socket.error:
        raise RuntimeError('Invalid IP Address to convert.')
    lower_order_23 = ip_bit_string[-23:]
    high_order_25 = '0000000100000000010111100'
    mac_bit_string = high_order_25 + lower_order_23
    final_string = '{0:012x}'.format(int(mac_bit_string, 2))
    mac_string = ':'.join('%02x' % b for b in (codecs.decode(final_string, 'hex')))
    return mac_string.lower()


#################################################################################################################
def build_mcast():
    # Gettting the input parameters
    input_param, common_param = requires("MCAST")
    udp_pkt = None
    fuzzy = (input("Random Multicast Packet? (y/n) > ").strip()).lower()
    if fuzzy == "y":
        inputs = []
        # Common parameters
        for i in range(0, len(common_param)):
            inputs.insert(i, input("{} > ".format(common_param[i])))
        udp_pkt = udp_packet(fuzzy, 'MCAST', inputs)
        if udp_pkt != None:
            logger.info("Multicast Packet built")
            udp_pkt.show()
            return udp_pkt, inputs[0], inputs[1]
        else:
            return None
    elif fuzzy == "n":
        inputs = []
        dot1q_prio = []
        for i in range(0, len(input_param)):
            temp_input = input("{} > ".format(input_param[i]))
            if "Tag" in input_param[i] and temp_input.lower() == "y":
                inputs.insert(i, input("VLAN Tag (x,y) > "))
                dot1q_prio.insert(0, input("CoS (x,y | default 0) > "))
            elif "Tag" in input_param[i] and temp_input.lower() == "n":
                inputs.insert(i, False)
            elif "Tag" not in input_param[i]:
                inputs.insert(i, temp_input)
            else:
                logger.critical(
                    "Invalid choice, got '{}' expected values (y/n)".format(
                        temp_input))
                return None
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))
        # Based on the provided VLAN tag return tagged/untagged UDP packet
        if not (inputs[5]):
            udp_pkt = udp_packet(fuzzy, 'MCAST', inputs)
        else:
            vlans = (inputs[5]).strip().split(",")
            cos = (dot1q_prio[0]).strip().split(",")
            try:
                vlans = [int(i) for i in vlans]
            except ValueError:
                logger.critical(
                    "Invalid vlan id'{}' Expected integer".format(vlans))
                logger.critical(ValueError, exc_info=True)
                return None
            udp_pkt = udp_packet(fuzzy, 'MCAST', inputs)
            cos = validate_cos(cos, vlans)
            if udp_pkt != None and cos != None:
                udp_pkt = add_vlan(udp_pkt, vlans, cos)
            else:
                return None
        if udp_pkt != None:
            logger.info("UDP Packet built")
            udp_pkt.show()
            return udp_pkt, inputs[6], inputs[7]
        else:
            return None
    else:
        logger.critical(
            "Invalid input '{}' Expected string (y/n)".format(fuzzy))
        return None


#################################################################################################################
def add_dot1q(vlan_list, cos_list, layer):
    if (len(vlan_list) == 1):
        dot1q = Dot1Q(vlan=vlan_list[0], prio=cos_list[0])
        vlan_list.pop(0)
        cos_list.pop(0)
        dot1q.add_payload(layer.payload)
        return dot1q
    else:
        dot1q = Dot1Q(vlan=vlan_list[0], prio=cos_list[0], type=33024)
        vlan_list.pop(0)
        cos_list.pop(0)
        dot1q.add_payload(add_dot1q(vlan_list, cos_list, layer))
        return dot1q


#################################################################################################################
def add_vlan(packet, vlans, cos):
    layer = packet.firstlayer()
    while not isinstance(layer, NoPayload):
        if 'chksum' in layer.default_fields:
            del layer.chksum
        if (type(layer) is Ether):
            layer.type = 33024
            dot1q = add_dot1q(vlans, cos, layer)
            layer.remove_payload()
            layer.add_payload(dot1q)
            layer = dot1q
        layer = layer.payload
    return packet


#################################################################################################################
def validate_cos(cos, vlans):
    if len(cos) > len(vlans):
        logger.critical("Mismatched CoS and vlans input")
        return None
    elif len(cos) <= len(vlans):
        if len(cos) == 1 and cos[0] == "":
            _ = cos.pop(0)
        cos.extend([0] * (len(vlans) - len(cos)))
        try:
            cos = [int(i) for i in cos]
        except ValueError:
            logger.critical("Invalid CoS id'{}' Expected integer".format(cos))
            logger.critical(ValueError, exc_info=True)
            return None
        if (any((i > 7) or (i < 0) for i in cos)):
            logger.critical(
                "Invalid CoS values'{}' Expected value between 0 and 7".format(cos))
            return None
        else:
            return cos


#################################################################################################################
def customAction(packets, input_vars, edit_type):
    final_pkts = PacketList()
    if edit_type == "ALL":
        for p in packets:
            if not (p.haslayer(IP)):
                if len(input_vars[0]) > 0:
                    p[Ether].src = input_vars[0]
                if len(input_vars[1]) > 0:
                    p[Ether].dst = input_vars[1]
                final_pkts.append(p)
            elif (p.haslayer(IP)):
                if len(input_vars[0]) > 0:
                    p[Ether].src = input_vars[0]
                if len(input_vars[1]) > 0:
                    p[Ether].dst = input_vars[1]
                if len(input_vars[2]) > 0:
                    p[IP].src = input_vars[2]
                if len(input_vars[3]) > 0:
                    p[IP].dst = input_vars[3]
                final_pkts.append(p)
            else:
                pass
    elif edit_type == "IP":
        for p in packets:
            if (p.haslayer(IP)):
                if len(input_vars[0]) > 0:
                    p[Ether].src = input_vars[0]
                if len(input_vars[1]) > 0:
                    p[Ether].dst = input_vars[1]
                if len(input_vars[2]) > 0:
                    p[IP].src = input_vars[2]
                if len(input_vars[3]) > 0:
                    p[IP].dst = input_vars[3]
                final_pkts.append(p)
            else:
                pass  # Currently for IP only flows not adding the non-IP packets
    elif edit_type == "NON-IP":
        for p in packets:
            if not (p.haslayer(IP)):
                if len(input_vars[0]) > 0:
                    p[Ether].src = input_vars[0]
                if len(input_vars[1]) > 0:
                    p[Ether].dst = input_vars[1]
                final_pkts.append(p)
            else:
                pass  # Currently for non-IP only flows not adding the IP packets
    else:
        logger.critical(
            "Invalid edit_type: '{}'. Expected 'ALL/IP/NON-IP".format(
                edit_type))
        return None
    return final_pkts


#################################################################################################################
def getFlow(packets, flows, filter_type):
    curr_flow = PacketList()
    if filter_type == "IP":
        for p in packets:
            if (p.haslayer(IP)):
                if p[IP].src == flows[0] and p[IP].dst == flows[1]:
                    curr_flow.append(p)
            else:
                pass
    elif filter_type == "NON-IP":
        for p in packets:
            if not (p.haslayer(IP)):
                if p[Ether].src == flows[0] and p[Ether].dst == flows[1]:
                    curr_flow.append(p)
            else:
                pass
    else:
        logger.critical(
            "Invalid edit_type: '{}'. Expected 'IP/NON-IP".format(filter_type))
        sys.exit(1)
    return curr_flow


#################################################################################################################
def threaded_sendp(packets, count, intf):
    # Sending out the packet based on the packet count and egress interface
    try:
        pkt_count = int(count)
        sendp(packets, iface=intf, count=pkt_count, verbose=0)
    except ValueError:
        if count == "c":
            sendp(packets, iface=intf, loop=1, verbose=0)
        else:
            logger.critical(
                "Invalid packet count '{}' of type '{}' Expected int or 'c'".
                format(count, type(count)))
            return None


#################################################################################################################
def send_flows(flow_arr):
    all_processes = []
    for flow in flow_arr:
        if flow != None:
            process = multiprocessing.Process(target=threaded_sendp,
                                              args=(flow[0], flow[1], flow[-1]))
            process.start()
            all_processes.append([process, flow[1]])
    loop_counter = 0
    for i in range(len(all_processes)):
        if all_processes[i][-1] != "c":
            all_processes[i][0].join()
        else:
            loop_counter += 1
    if loop_counter > 0:
        print("\nPress Ctrl+C to break continous flows.\n")
        try:
            input("> ")
        except KeyboardInterrupt:
            for i in range(len(all_processes)):
                if all_processes[i][-1] == "c":
                    process.terminate()
            logger.info("Done sending all flows.")
    else:
        logger.info("Done sending all flows")


#################################################################################################################
def pcap_mod():
    # Get the PCAP file path and verify if its a valid file.
    file_path = input("Path to pcap file > ").strip()
    try:
        pkts = sniff(offline=file_path)
    except Scapy_Exception as msg:
        logger.critical("Not a pcap capture file (bad magic)")
        return None
    logger.info("Pcap file loaded successfully!")

    # Get the action to be performed.
    try:
        action = int(
            input(
                "\nSelect action:\n\n1 -- Edit and replay all packets\n2 -- Edit and replay specific IP flows\n3 -- Edit and replay specific non-IP flows\n4 -- Replay the same pcap\n\nEnter your choice (1-4) > "
            ).strip())
        # Edit and replay all packets
        if action == 1:
            logger.info("Changing all packets in the capture file.")

            # Getting input parameters
            input_param, common_param = requires("PCAP")
            inputs = []
            for i in range(0, len(input_param)):
                temp_input = input("{} > ".format(input_param[i])).strip()
                inputs.insert(i, temp_input)
            for j in range(0, len(common_param)):
                i = i + 1
                inputs.insert(i, input("{} > ".format(common_param[j])))

            # Modify the packets based on the inputs provided.
            mod_pkts = customAction(pkts, inputs, "ALL")
            # Cleaning up old packets and releasing memory
            del pkts
            gc.collect()
            logger.info("Packets after change...")
            mod_pkts.nsummary()

            # Sending out the packet based on the packet count and egress interface
            pkt_count = inputs[4].strip()
            send_flows([(mod_pkts, pkt_count, inputs[5])])
            del mod_pkts
            gc.collect()

        elif action == 2:
            logger.info("Following unique flows found in capture...")
            ip_packet_counts = Counter()
            for ip_pkt in pkts:
                if (ip_pkt.haslayer(IP)):
                    key = tuple(([ip_pkt[IP].src, ip_pkt[IP].dst]))
                    ip_packet_counts.update([key])
            index = 0
            ip_flows = []
            flow_table = []
            flow_table_headers = [
                'Index', 'Source', 'Destination', 'Packet Count'
            ]
            for key, count in ip_packet_counts.items():
                index += 1
                #print ("{} > {:<15} <----> {:<20} pkts:{:<10}".format(index, key[0], key[1], count))
                flow_table.append([index, key[0], key[1], count])
                ip_flows.append(key)
            print(
                tabulate(flow_table,
                         flow_table_headers,
                         tablefmt='github',
                         colalign=("center", "left", "left", "center")))

            flow_index = (input(
                "\nSelect flows to modify and replay (all to modify all IP flows) > "
            ).strip()).split(",")
            if len(flow_index) == 1 and flow_index[0] == "all":
                # Getting input parameters
                input_param, common_param = requires("PCAP")
                inputs = []
                for i in range(0, len(input_param)):
                    temp_input = input("{} > ".format(input_param[i])).strip()
                    inputs.insert(i, temp_input)
                for j in range(0, len(common_param)):
                    i = i + 1
                    inputs.insert(i, input("{} > ".format(common_param[j])))

                # Modify the packets based on the inputs provided.
                mod_pkts = customAction(pkts, inputs, "IP")
                # Cleaning up old packets and releasing memory
                del pkts
                gc.collect()
                logger.info("Packets after change...")
                mod_pkts.nsummary()

                # Sending out the packet based on the packet count and egress interface
                pkt_count = inputs[4].strip()
                send_flows([(mod_pkts, pkt_count, inputs[5])])
                # Cleaning up packets and releasing memory
                del mod_pkts
                gc.collect()
            else:
                final_flows = []
                for i in flow_index:
                    idx = int(i) - 1
                    print("\nModify flow [ ", ip_flows[idx], " ]\n")

                    # Getting input parameters
                    input_param, common_param = requires("PCAP")
                    inputs = []
                    for i in range(0, len(input_param)):
                        temp_input = input("{} > ".format(input_param[i])).strip()
                        inputs.insert(i, temp_input)
                    for j in range(0, len(common_param)):
                        i = i + 1
                        inputs.insert(i, input("{} > ".format(common_param[j])))

                    # Extract flow
                    curr_flow = getFlow(pkts, ip_flows[idx], "IP")

                    # Modify the packets based on the inputs provided.
                    mod_pkts = customAction(curr_flow, inputs, "IP")
                    # Cleaning up packets and releasing memory
                    del curr_flow
                    gc.collect()
                    logger.info("Packets after change...")
                    mod_pkts.nsummary()
                    final_flows.append((mod_pkts, inputs[4], inputs[5]))
                logger.info("Sending out modified flows...")
                send_flows(final_flows)

        elif action == 3:
            logger.info("Following flows found in capture...")
            non_ip_packet_counts = Counter()
            for non_ip_pkt in pkts:
                if not (non_ip_pkt.haslayer(IP)):
                    key = tuple(([non_ip_pkt[Ether].src,
                                  non_ip_pkt[Ether].dst]))
                    non_ip_packet_counts.update([key])
            index = 0
            non_ip_flows = []
            flow_table = []
            flow_table_headers = [
                'Index', 'Source MAC', 'Destination MAC', 'Packet Count'
            ]
            for key, count in non_ip_packet_counts.items():
                index += 1
                flow_table.append([index, key[0], key[1], count])
                #print ("{} > {:<15} <----> {:<20} pkts:{:<10}".format(index, key[0], key[1], count))
                non_ip_flows.append(key)
            print(
                tabulate(flow_table,
                         flow_table_headers,
                         tablefmt='github',
                         colalign=("center", "left", "left", "center")))

            flow_index = (input(
                "\nSelect flows to modify and replay (all to modify all flows) > "
            ).strip()).split(",")
            if len(flow_index) == 1 and flow_index[0] == "all":
                # Getting input parameters
                input_param, common_param = requires("PCAP")
                inputs = []
                for i in range(0, len(input_param) - 2):
                    temp_input = input("{} > ".format(input_param[i])).strip()
                    inputs.insert(i, temp_input)
                for j in range(0, len(common_param)):
                    i = i + 1
                    inputs.insert(i, input("{} > ".format(common_param[j])))

                # Modify the packets based on the inputs provided.
                mod_pkts = customAction(pkts, inputs, "NON-IP")
                # Cleaning up old packets and releasing memory
                del pkts
                gc.collect()
                logger.info("Packets after change...")
                mod_pkts.nsummary()

                # Sending out the packet based on the packet count and egress interface
                pkt_count = inputs[2]
                send_flows([(mod_pkts, pkt_count, inputs[3])])
                del mod_pkts
                gc.collect()
            else:
                final_flows = []
                for i in flow_index:
                    idx = int(i) - 1
                    print("\nModify flow [ ", non_ip_flows[idx], " ]\n")

                    # Getting input parameters
                    input_param, common_param = requires("PCAP")
                    inputs = []
                    for i in range(0, len(input_param) - 2):
                        temp_input = input("{} > ".format(input_param[i])).strip()
                        inputs.insert(i, temp_input)
                    for j in range(0, len(common_param)):
                        i = i + 1
                        inputs.insert(i, input("{} > ".format(common_param[j])))

                    # Extract flow
                    curr_flow = getFlow(pkts, non_ip_flows[idx], "NON-IP")

                    # Modify the packets based on the inputs provided.
                    mod_pkts = customAction(curr_flow, inputs, "NON-IP")
                    # Cleaning up packets and releasing memory
                    del curr_flow
                    gc.collect()
                    logger.info("Packets after change...")
                    mod_pkts.nsummary()
                    final_flows.append((mod_pkts, inputs[2], inputs[3]))
                logger.info("Sending out modified flows...")
                send_flows(final_flows)

        elif action == 4:
            # Get common parameters: count and egress interface
            _, common_param = requires("PCAP")
            inputs = []
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            print("-" * 50)
            pkts.nsummary()

            # Sending out the packet based on packet count and egress interface
            pkt_count = inputs[0]
            send_flows([(pkts, pkt_count, inputs[1])])
            logging.info("Replaying the same capture file...")
            del pkts
            gc.collect()
        logging.info("Done with PCAP module")
        pass
    except ValueError:
        logger.critical(
            "Invalid action: '{}' provided. Expected integer (1-4)".format(
                action))
        return None


#################################################################################################################
def arp_packet(fuzzy, module, arp_type, arp_inputs):
    final_packet = None
    if arp_type == 'req':
        op_code = "who-has"
    elif arp_type == 'resp':
        op_code = "is-at"
    else:
        logger.critical(
            "Invalid arp_type: '{}' Expected value (req/resp)".format(arp_type))
        return None
    if fuzzy == 'y':
        if module != None and len(arp_inputs) != 0:
            if module == 'VXLAN' or module  == 'MPLS':
                src_mac = sender_mac = RandMAC()._fix()
                sender_ip = RandIP("172.16.0.0/12")._fix()
                trgt_ip = RandIP("172.16.0.0/12")._fix()
                if arp_type == "req":
                    dst_mac = 'ff:ff:ff:ff:ff:ff'
                    trgt_mac = '00:00:00:00:00:00'
                elif arp_type == "resp":
                    dst_mac = trgt_mac = RandMAC()._fix()
            elif module == 'ARP':
                src_mac = sender_mac = get_if_hwaddr(arp_inputs[1])
                sender_ip = get_if_addr(arp_inputs[1])
                ip_pattern = re.compile(r'^0\.0\.0\.0$')
                if ip_pattern.match(sender_ip):  # If there is no IP on port generate random IP in 172.16.0.0/12
                    sender_ip = RandIP("172.16.0.0/12")._fix()
                trgt_ip = RandIP("172.16.0.0/12")._fix()
                if arp_type == "req":
                    dst_mac = 'ff:ff:ff:ff:ff:ff'
                    trgt_mac = '00:00:00:00:00:00'
                elif arp_type == 'resp':
                    dst_mac = trgt_mac = RandMAC()._fix()
            else:
                return None
    elif fuzzy == 'n':
        if module != None and len(arp_inputs) != 0:
            if module == 'ARP' or module == 'VXLAN' or module == 'MPLS':
                src_mac = arp_inputs[0]
                sender_mac = arp_inputs[2]
                sender_ip = arp_inputs[3]
                trgt_ip = arp_inputs[5]
                if arp_type == "req":
                    dst_mac = 'ff:ff:ff:ff:ff:ff'
                    trgt_mac = '00:00:00:00:00:00'
                elif arp_type == 'resp':
                    dst_mac = arp_inputs[1]
                    trgt_mac = arp_inputs[4]
        else:
            return None
    final_packet = (Ether(src=src_mac, dst=dst_mac)) / ARP(op=op_code,
                                                           hwsrc=sender_mac,
                                                           psrc=sender_ip,
                                                           hwdst=trgt_mac,
                                                           pdst=trgt_ip)
    return final_packet


#################################################################################################################
def icmp_packet(fuzzy, module, icmp_type, icmp_inputs):
    final_packet = None
    if icmp_type == 'req':
        pkt_type = 'echo-request'
    elif icmp_type == 'reply':
        pkt_type = 'echo-reply'
    else:
        logger.critical(
            "Invalid icmp_type: '{}' Expected value (req/reply)".format(
                icmp_type))
        return None
    if fuzzy == 'y':
        if module != None:
            if module == 'VXLAN' or module == 'MPLS':
                src_mac, dst_mac = RandMAC()._fix(), RandMAC()._fix()
                src_ip, dst_ip = RandIP("172.16.0.0/12")._fix(), RandIP("172.16.0.0/12")._fix()
                ttl = randint(10, 255)
            elif module == 'ICMP':
                src_mac, dst_mac = get_if_hwaddr(icmp_inputs[1]), RandMAC()._fix()
                src_ip, dst_ip = get_if_addr(icmp_inputs[1]), RandIP("172.16.0.0/12")._fix()
                ip_pattern = re.compile(r'^0\.0\.0\.0$')
                if ip_pattern.match(src_ip):  # If there is no IP on port generate random IP in 172.16.0.0/12
                    src_ip = RandIP("172.16.0.0/12")._fix()
                ttl = randint(10, 255)
            else:
                return None
        else:
            return None
    elif fuzzy == 'n':
        if module != None and len(icmp_inputs) != 0:
            if module == 'VXLAN' or module == 'MPLS': #Why are we not checking validity of MACs/IPs here
                src_mac, dst_mac = icmp_inputs[0], icmp_inputs[1]
                src_ip, dst_ip = icmp_inputs[2], icmp_inputs[3]
                ttl = int(icmp_inputs[4])
            elif module == 'ICMP':
                dst_mac = icmp_inputs[1]
                dst_ip = icmp_inputs[3]
                mac_pattern = re.compile(
                    r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
                ip_pattern = re.compile(
                    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
                )
                ip_pattern_2 = re.compile(r'^0\.0\.0\.0$')
                # If no valid source mac provided get it from source interface
                if mac_pattern.match(icmp_inputs[0].strip()):
                    src_mac = icmp_inputs[0]
                else:
                    logger.error(
                        "Invalid source MAC provided. Extracting source MAC from source interface."
                    )
                    src_mac = get_if_hwaddr(icmp_inputs[7])
                # If no valid source IP provided get it from source interface
                if not ip_pattern.match(icmp_inputs[2]):
                    logger.error(
                        "Invalid source IP provided. Extracting source IP from source interface."
                    )
                    src_ip = get_if_addr(icmp_inputs[7])
                    if ip_pattern_2.match(src_ip):
                        logger.critical(
                            "Source interface does not have a valid IP address")
                        return None
                else:
                    src_ip = icmp_inputs[2]
                ttl = int(icmp_inputs[4])
        else:
            return None
    random_data = urandom(64)  # This is for data payload
    random_id = randint(1, 2000)  # This is for identifier
    final_packet = Ether(src=src_mac, dst=dst_mac) / IP(
        src=src_ip, dst=dst_ip, ttl=ttl) / ICMP(
            id=random_id, type=pkt_type) / Raw(load=random_data)
    return final_packet


#################################################################################################################
def udp_packet(fuzzy, module, udp_inputs):
    final_packet = None
    if fuzzy == 'y':
        if module != None:
            if module == 'VXLAN' or module == 'MPLS':
                src_mac, dst_mac = RandMAC()._fix(), RandMAC()._fix()
                src_ip, dst_ip = RandIP("172.16.0.0/12")._fix(), RandIP("172.16.0.0/12")._fix()
                udp_dport = udp_sport = randint(49152, 65535)
            elif module == 'MCAST':
                src_mac, src_ip = get_if_hwaddr(udp_inputs[1]), get_if_addr(udp_inputs[1])
                ip_pattern = re.compile(r'^0\.0\.0\.0$')
                if ip_pattern.match(src_ip):  # If there is no IP on port generate random IP in 172.16.0.0/12
                    src_ip = RandIP("172.16.0.0/12")._fix()
                dst_ip = RandIP("239.0.0.0/8")._fix()
                dst_mac = convert_multicast_ip_to_mac(dst_ip)
                udp_dport = udp_sport = randint(49152, 65535)
        else:
            return None
    elif fuzzy == 'n':
        if module != None and len(udp_inputs) != 0:
            if module == 'VXLAN' or module == 'MPLS':
                src_mac, dst_mac = udp_inputs[0], udp_inputs[1]
                src_ip, dst_ip = udp_inputs[2], udp_inputs[3]
                udp_sport, udp_dport = int(udp_inputs[4]), int(udp_inputs[5])
            elif module == 'MCAST':
                mac_pattern = re.compile(
                    r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
                ip_pattern = re.compile(
                    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
                )
                ip_pattern_2 = re.compile(r'^0\.0\.0\.0$')
                if mac_pattern.match(udp_inputs[0].strip()):
                    src_mac = udp_inputs[0]
                else:
                    src_mac = get_if_hwaddr(udp_inputs[7])
                if not ip_pattern.match(udp_inputs[1].strip()):
                    logger.error(
                        "Invalid source IP provided. Extracting source IP from source interface."
                    )
                    src_ip = get_if_addr(udp_inputs[7])
                    if ip_pattern_2.match(src_ip):
                        logger.critical(
                            "Source interface does not have a valid IP address")
                        return None
                else:
                    src_ip = udp_inputs[1]
                if not ip_pattern.match(udp_inputs[2]):
                    logger.error("Invalid destination IP provided.")
                    return None
                else:
                    dst_ip = udp_inputs[2]
                dst_mac = convert_multicast_ip_to_mac(dst_ip)
                try:
                    if udp_inputs[4] and udp_inputs[3]:
                        udp_dport, udp_sport = int(udp_inputs[4]), int(udp_inputs[3])
                except ValueError:
                    logger.critical(
                        "Invalid udp_sport: '{}' and udp_dport: '{}' provided")
                    return None
        else:
            return None
    random_data = urandom(64)  # This is for data payload
    final_packet = Ether(src=src_mac, dst=dst_mac) / IP(
        src=src_ip, dst=dst_ip, ttl=randint(10, 255)) / UDP(
            sport=udp_sport, dport=udp_dport) / Raw(load=random_data)
    return final_packet


#################################################################################################################
def tcp_packet(fuzzy, module, tcp_inputs):
    final_packet = None
    if fuzzy == 'y':
        if module != None:
            if module == 'VXLAN' or module == 'MPLS':
                src_mac, dst_mac = RandMAC()._fix(), RandMAC()._fix()
                src_ip, dst_ip = RandIP("172.16.0.0/12")._fix(), RandIP("172.16.0.0/12")._fix()
                tcp_sport, tcp_dport = RandShort()._fix(), RandShort()
        else:
            return None
    elif fuzzy == 'n':
        if module != None and len(tcp_inputs) != 0:
            if module == 'VXLAN' or module == 'MPLS':
                src_mac, dst_mac = tcp_inputs[0], tcp_inputs[1]
                src_ip, dst_ip = tcp_inputs[2], tcp_inputs[3],
                tcp_sport, tcp_dport = int(tcp_inputs[4]), int(tcp_inputs[5])
        else:
            return None
    final_packet = Ether(src=src_mac, dst=dst_mac) / IP(
        src=src_ip, dst=dst_ip, ttl=randint(10, 255)) / TCP(
            sport=tcp_sport, dport=tcp_dport, flags='S')
    return final_packet


#################################################################################################################
def vxlan_packet(fuzzy, inner_pkt, vxlan_inputs):
    final_pkt = None
    if fuzzy == 'y':
        if inner_pkt != None and len(vxlan_inputs) != 0:
            outer_dst_mac, outer_src_mac = RandMAC()._fix(), get_if_hwaddr(vxlan_inputs[1])
            outer_src_ip, outer_dst_ip = RandIP("192.168.0.0/12")._fix(), RandIP("192.168.0.0/12")._fix()
            outer_src_port, outer_dst_port = randint(49152, 65535), 4789
            vni = randint(1, 16777215)
        else:
            return None
    elif fuzzy == 'n':
        if inner_pkt != None and len(vxlan_inputs) != 0:
            outer_dst_mac = vxlan_inputs[1]
            # If no valid outer source mac provided pull from source interface
            mac_pattern = re.compile(
                r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
            if mac_pattern.match(vxlan_inputs[0].strip()):
                outer_src_mac = vxlan_inputs[0]
            else:
                outer_src_mac = get_if_hwaddr(vxlan_inputs[8])
            # If no valid outer source IP provided assign random IP
            outer_src_ip = vxlan_inputs[2]
            ip_pattern = re.compile(
                r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
            )
            if not ip_pattern.match(outer_src_ip):
                outer_src_ip = RandIP("192.168.0.0/12")._fix()
            outer_dst_ip = vxlan_inputs[3]
            outer_src_port, outer_dst_port = int(vxlan_inputs[4]), int(vxlan_inputs[5])
            vni = int(vxlan_inputs[6])
        else:
            return None
    final_pkt = Ether(src=outer_src_mac, dst=outer_dst_mac) / IP(
        flags='DF', src=outer_src_ip, dst=outer_dst_ip, ttl=randint(
            10, 255)) / UDP(sport=outer_src_port, dport=outer_dst_port) / VXLAN(
                vni=vni, flags=8) / inner_pkt
    return final_pkt


#################################################################################################################
def build_vxlan(msg_type):
    if msg_type == 'VXLAN_ICMP':
        # Get input parameters
        vxlan_input_param, common_param = requires("VXLAN")
        icmp_input_param, _ = requires('ICMP')
        inner_icmp_pkt, vxlan_icmp_pkt = None, None
        fuzzy = (input("Generate random Vxlan ICMP Packet? (y/n) > ").strip()).lower()
        if fuzzy == 'y':
            # Build the inner ICMP packet
            icmp_type = (input("ICMP Type (req/reply) > ").strip()).lower()
            inputs = []
            # Common parameters
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            inner_icmp_pkt = icmp_packet(fuzzy, 'VXLAN', icmp_type, inputs)
            # Generate the outer Vxlan packet
            vxlan_icmp_pkt = vxlan_packet(fuzzy, inner_icmp_pkt, inputs)
            if inner_icmp_pkt != None and vxlan_icmp_pkt != None:
                logger.info("Inner ICMP packet built")
                logger.info("Vxlan ICMP Packet built")
                vxlan_icmp_pkt.show()
                return vxlan_icmp_pkt, inputs[0], inputs[1]
            else:
                return None
        elif fuzzy == 'n':
            icmp_type = (input("ICMP Type (req/reply) > ").strip()).lower()
            # Getting input parameters
            inputs = []
            del icmp_input_param[-1]  # Skipping VLAN tag for inner ICMP
            for i in range(0, len(icmp_input_param)):
                temp_input = input("Inner {} > ".format(icmp_input_param[i]))
                inputs.insert(i, temp_input)
            inner_icmp_pkt = icmp_packet(fuzzy, 'VXLAN', icmp_type, inputs)
            # Craft the outer Vxlan packet. Get vxlan input params
            vxlan_inputs = []
            for i in range(0, len(vxlan_input_param)):
                temp_input = input("{} > ".format(vxlan_input_param[i]))
                if "4789" in vxlan_input_param[i] and temp_input != True:
                    temp_input = '4789'
                vxlan_inputs.insert(i, temp_input)
            # Common parameters
            for j in range(0, len(common_param)):
                i = i + 1
                vxlan_inputs.insert(i, input("{} > ".format(common_param[j])))
            vxlan_icmp_pkt = vxlan_packet(fuzzy, inner_icmp_pkt, vxlan_inputs)
            if inner_icmp_pkt != None and vxlan_icmp_pkt != None:
                logger.info("Inner ICMP packet built")
                logger.info("Vxlan ICMP Packet built")
                vxlan_icmp_pkt.show()
                return vxlan_icmp_pkt, vxlan_inputs[7], vxlan_inputs[8]
            else:
                return None
        else:
            logger.critical(
                "Invalid input '{}' Expected string (y/n)".format(fuzzy))
            return None
    elif msg_type == 'VXLAN_UDP':
        # Get input parameters
        vxlan_input_param, common_param = requires("VXLAN")
        udp_input_param, _ = requires("UDP")
        inner_udp_pkt, vxlan_udp_pkt = None, None
        fuzzy = (input("Generate random Vxlan UDP Packet? (y/n) > ").strip()).lower()
        if fuzzy == 'y':
            # Build the inner UDP packet
            # Common parameters
            inputs = []
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            inner_udp_pkt = udp_packet(fuzzy, 'VXLAN', None)
            # Generate the outer Vxlan packet
            vxlan_udp_pkt = vxlan_packet(fuzzy, inner_udp_pkt, inputs)
            if inner_udp_pkt != None and vxlan_udp_pkt != None:
                logger.info("Inner UDP packet built")
                logger.info("Vxlan UDP Packet built")
                vxlan_udp_pkt.show()
                return vxlan_udp_pkt, inputs[0], inputs[1]
            else:
                return None
        elif fuzzy == 'n':
            # Get inputs parameters
            udp_inputs = []
            del udp_input_param[-1]  # Skipping VLAN Tag for inner UDP
            for i in range(0, len(udp_input_param)):
                temp_input = input("Inner {} > ".format(udp_input_param[i]))
                udp_inputs.insert(i, temp_input)
            inner_udp_pkt = udp_packet(fuzzy, 'VXLAN', udp_inputs)
            # Craft outer vxlan packet
            vxlan_inputs = []
            for i in range(0, len(vxlan_input_param)):
                temp_input = input("{} > ".format(vxlan_input_param[i]))
                if "4789" in vxlan_input_param[i] and temp_input != True:
                    temp_input = '4789'
                vxlan_inputs.insert(i, temp_input)
            # Common parameters
            for j in range(0, len(common_param)):
                i = i + 1
                vxlan_inputs.insert(i, input("{} > ".format(common_param[j])))
            vxlan_udp_pkt = vxlan_packet(fuzzy, inner_udp_pkt, vxlan_inputs)
            if inner_udp_pkt != None and vxlan_udp_pkt != None:
                logger.info("Inner UDP packet built")
                logger.info("Vxlan UDP Packet built")
                vxlan_udp_pkt.show()
                return vxlan_udp_pkt, vxlan_inputs[7], vxlan_inputs[8]
            else:
                return None
        else:
            logger.critical(
                "Invalid input '{}' Expected string (y/n)".format(fuzzy))
            return None
    elif msg_type == 'VXLAN_TCP':
        # Get input parameters
        vxlan_input_param, common_param = requires("VXLAN")
        tcp_input_param, _ = requires("TCP")
        inner_tcp_pkt, vxlan_tcp_pkt = None, None
        fuzzy = (input("Generate random Vxlan TCP Packet? (y/n) > ").strip()).lower()
        if fuzzy == 'y':
            # Build the inner TCP packet
            # Common Parameters
            inputs = []
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            inner_tcp_pkt = tcp_packet(fuzzy, 'VXLAN', None)
            # Generate the outer Vxlan packet
            vxlan_tcp_pkt = vxlan_packet(fuzzy, inner_tcp_pkt, inputs)
            if inner_tcp_pkt != None and vxlan_tcp_pkt != None:
                logger.info("Inner TCP packet built")
                logger.info("Vxlan TCP Packet built")
                vxlan_tcp_pkt.show()
                return vxlan_tcp_pkt, inputs[0], inputs[1]
            else:
                return None
        elif fuzzy == 'n':
            # Get input prarameters
            tcp_inputs = []
            del tcp_input_param[-1]  # Skipping VLAN tag for inner TCP
            for i in range(0, len(tcp_input_param)):
                temp_input = input("Inner {} > ".format(tcp_input_param[i]))
                tcp_inputs.insert(i, temp_input)
            inner_tcp_pkt = tcp_packet(fuzzy, 'VXLAN', tcp_inputs)
            # Craft outer Vxlan packet
            vxlan_inputs = []
            for i in range(0, len(vxlan_input_param)):
                temp_input = input("{} > ".format(vxlan_input_param[i]))
                if "4789" in vxlan_input_param[i] and temp_input != True:
                    temp_input = '4789'
                vxlan_inputs.insert(i, temp_input)
            # Common parameters
            for j in range(0, len(common_param)):
                i = i + 1
                vxlan_inputs.insert(i, input("{} > ".format(common_param[j])))
            vxlan_tcp_pkt = vxlan_packet(fuzzy, inner_tcp_pkt, vxlan_inputs)
            if inner_tcp_pkt != None and vxlan_tcp_pkt != None:
                logger.info("Inner TCP packet built")
                logger.info("Vxlan TCP Packet built")
                vxlan_tcp_pkt.show()
                return vxlan_tcp_pkt, vxlan_inputs[7], vxlan_inputs[8]
            else:
                return None
        else:
            logger.critical(
                "Invalid input '{}' Expected string (y/n)".format(fuzzy))
            return None
    elif msg_type == 'VXLAN_ARP':
        # Get input parameters
        vxlan_input_param, common_param = requires("VXLAN")
        arp_input_param, _ = requires('ARP')
        inner_arp_pkt, vxlan_arp_pkt = None, None
        fuzzy = (input("Generate random Vxlan ARP Packet? (y/n) > ").strip()).lower()
        if fuzzy == 'y':
            # Build inner ARP Packet
            arp_type = (input("ARP Type (req/resp) > ").strip()).lower()
            inputs = []
            # Common parameters
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            inner_arp_pkt = arp_packet(fuzzy, 'VXLAN', arp_type, inputs)
            # Generate outer Vxlan packet
            vxlan_arp_pkt = vxlan_packet(fuzzy, inner_arp_pkt, inputs)
            if inner_arp_pkt != None and vxlan_arp_pkt != None:
                logger.info("Inner ARP packet built")
                logger.info("VXLAN ARP packet built")
                vxlan_arp_pkt.show()
                return vxlan_arp_pkt, inputs[0], inputs[1]
        elif fuzzy == 'n':
            # Build inner ARP packet
            arp_type = (input("ARP Type (req/resp) > ").strip()).lower()
            inputs = []
            del arp_input_param[-1]  # Skipping VLAN tag for inner ARP
            for i in range(0, len(arp_input_param)):
                temp_input = input("Inner {} > ".format(arp_input_param[i]))
                inputs.insert(i, temp_input)
            inner_arp_pkt = arp_packet(fuzzy, 'VXLAN', arp_type, inputs)
            # Craft the outer Vxlan packet. Get vxlan input params
            vxlan_inputs = []
            for i in range(0, len(vxlan_input_param)):
                temp_input = input("{} > ".format(vxlan_input_param[i]))
                if "4789" in vxlan_input_param[i] and temp_input != True:
                    temp_input = '4789'
                vxlan_inputs.insert(i, temp_input)
                # Common parameters
            for j in range(0, len(common_param)):
                i = i + 1
                vxlan_inputs.insert(i, input("{} > ".format(common_param[j])))
            vxlan_arp_pkt = vxlan_packet(fuzzy, inner_arp_pkt, vxlan_inputs)
            if inner_arp_pkt != None and vxlan_arp_pkt != None:
                logger.info("Inner ARP packet built")
                logger.info("Vxlan ARP Packet built")
                vxlan_arp_pkt.show()
                return vxlan_arp_pkt, vxlan_inputs[7], vxlan_inputs[8]
        else:
            logger.critical(
                "Invalid input '{}' Expected string (y/n)".format(fuzzy))
            return None
    else:
        logger.critical("Invalid msg_type: '{}' provided.".format(msg_type))
        return None


#################################################################################################################
def vxlan():
    # Select Vxlan packet type
    avail_vxlan_mods = {
        1: 'Vxlan - Inner ICMP',
        2: 'Vxlan - Inner UDP',
        3: 'Vxlan - Inner TCP',
        4: 'Vxlan - Inner ARP',
    }
    print('Packet Type:\n')
    for key in avail_vxlan_mods.keys():
        print(key, '--', avail_vxlan_mods[key])
    try:
        msg_type = int(input("\nEnter your choice (1-3) > ").strip())
        if msg_type == 1:
            return build_vxlan("VXLAN_ICMP")
        elif msg_type == 2:
            return build_vxlan("VXLAN_UDP")
        elif msg_type == 3:
            return build_vxlan("VXLAN_TCP")
        elif msg_type == 4:
            return build_vxlan("VXLAN_ARP")
        else:
            logger.critical("Invalid msg_type, expected integer (1-3)")
            return None
    except ValueError:
        logger.critical("Invalid msg_type, expected integer (1-3)")
        return None
    
#################################################################################################################
def mpls_packet(fuzzy, inner_pkt, mpls_inputs): #TTL and cos value of labels are always set to 255 and 0 respectively
    final_pkt = None
    mpls_headers = None
    if fuzzy == 'y':
        mpls_labels_list = None
        mpls_label_count = 0
        if inner_pkt != None and len(mpls_inputs) != 0:
            outer_dst_mac, outer_src_mac = RandMAC()._fix(), get_if_hwaddr(mpls_inputs[1])
            #Number of labels
            mpls_label_count = int((input("No. of MPLS labels (maximum = 10) > ").strip()).lower())
            if mpls_label_count == 1:
                mpls_headers = MPLS(label=randint(16, 1048575), cos=0, s=1, ttl=255) # MPLS labels 0-15 are reserved, max MPLS label = 2^20 - 1
            elif 2 <= mpls_label_count <= 10:
                mpls_headers = MPLS(label=randint(16, 1048575), cos=0, s=0, ttl=255) 
                for i in range(1, mpls_label_count):
                    bos=0 #bottom_of_stack logic
                    if i==mpls_label_count-1:
                        bos=1
                    mpls_headers = mpls_headers / MPLS(label=randint(16, 1048575), cos=0, s=bos, ttl=255)                       
            else:
                logger.critical("Invalid input '{}' Expected value <= 10".format(mpls_label_count))
                return None
        else:
            return None  
        final_pkt = Ether(src=outer_src_mac, dst=outer_dst_mac, type=0x8847) / mpls_headers / inner_pkt #Skipping Control Word for random packet generation
        return final_pkt          
    elif fuzzy == 'n':
        mpls_labels_list = None
        mpls_label_count = 0
        if inner_pkt != None and len(mpls_inputs) != 0:
            outer_dst_mac = mpls_inputs[1]
            # If no valid outer source mac provided pull from source interface
            mac_pattern = re.compile(
                r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
            if mac_pattern.match(mpls_inputs[0].strip()):
                outer_src_mac = mpls_inputs[0]
            else:
                outer_src_mac = get_if_hwaddr(mpls_inputs[5])
            mpls_labels_list = mpls_inputs[2]
            mpls_label_count = len(mpls_labels_list)
            if mpls_label_count == 1:
                mpls_headers = MPLS(label=int(mpls_labels_list[0]), cos=0, s=1, ttl=255)
            elif mpls_label_count >= 2 :
                mpls_headers = MPLS(label=int(mpls_labels_list[0]), cos=0, s=0, ttl=255) 
                for i in range(1, mpls_label_count):
                    bos=0 #bottom_of_stack logic
                    if i==mpls_label_count-1:
                        bos=1
                    mpls_headers = mpls_headers / MPLS(label=int(mpls_labels_list[i]), cos=0, s=bos, ttl=255)
        else:
            return None
        #PW Control Word logic
        if mpls_inputs[3] is True:
            final_pkt = Ether(src=outer_src_mac, dst=outer_dst_mac, type=0x8847) / mpls_headers / EoMCW(zero=0, reserved=0, seq=0) / inner_pkt
        else:
            final_pkt = Ether(src=outer_src_mac, dst=outer_dst_mac, type=0x8847) / mpls_headers / inner_pkt
        return final_pkt  

#################################################################################################################
def build_mpls(msg_type):
    if msg_type == 'MPLS_ICMP':
        # Get input parameters
        mpls_input_param, common_param = requires("MPLS")
        icmp_input_param, _ = requires('ICMP') 
        inner_icmp_pkt, mpls_icmp_pkt = None, None
        fuzzy = (input("Generate random MPLS ICMP Packet? (y/n) > ").strip()).lower()
        if fuzzy == 'y':
            # Build the inner ICMP packet
            icmp_type = (input("ICMP Type (req/reply) > ").strip()).lower()
            inputs = []
            # Common parameters
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            inner_icmp_pkt = icmp_packet(fuzzy, 'MPLS', icmp_type, inputs)
            # Generate the Mpls packet
            mpls_icmp_pkt = mpls_packet(fuzzy, inner_icmp_pkt, inputs) 
            if inner_icmp_pkt != None and mpls_icmp_pkt != None:
                logger.info("Inner ICMP packet built")
                logger.info("MPLS ICMP Packet built")
                mpls_icmp_pkt.show() 
                return mpls_icmp_pkt, inputs[0], inputs[1]
            else:
                return None  
        elif fuzzy == 'n':
            labels = None
            icmp_type = (input("ICMP Type (req/reply) > ").strip()).lower()
            # Getting input parameters
            inputs = []
            del icmp_input_param[-1]  # Skipping VLAN tag for inner ICMP
            for i in range(0, len(icmp_input_param)):
                temp_input = input("Inner {} > ".format(icmp_input_param[i]))
                inputs.insert(i, temp_input)
            inner_icmp_pkt = icmp_packet(fuzzy, 'MPLS', icmp_type, inputs)
            # Craft the MPLS packet. Get mpls input parameters
            mpls_inputs = []
            for i in range(0, len(mpls_input_param)):
                temp_input = input("{} > ".format(mpls_input_param[i]))
                if "Labels" in mpls_input_param[i]:
                    #using a list comprehension with dummy variable x to create a list from the labels (which was a string)
                    labels = [x.strip() for x in temp_input.split(',')]
                    for k in range(0, len(labels)):
                        try:
                            if (0 <= int(labels[k]) <= 1048575) is False: #max MPLS label = 2^20
                                logger.error(
                                    "Invalid input '{}' Expected label in range (0-1048575)".format(labels[k]))
                                return None
                        except ValueError:
                            logger.error(
                                "Invalid input '{}' Expected comma separated values in range (0-1048575)".format(labels[k]))
                            return None                  
                    mpls_inputs.insert(i,labels)
                elif "Control Word" in mpls_input_param[i] and temp_input.lower() == "y":
                    mpls_inputs.insert(i, True)
                elif "Control Word" in mpls_input_param[i] and temp_input.lower() == "n":
                    mpls_inputs.insert(i, False)
                elif "Control Word" not in mpls_input_param[i]:
                    mpls_inputs.insert(i, temp_input)
                else:
                    logger.critical("Invalid choice, got '{}' expected values (y/n)".format(temp_input))
                    return None 
            # Common parameters
            for j in range(0, len(common_param)):
                i = i + 1
                mpls_inputs.insert(i, input("{} > ".format(common_param[j])))
            mpls_icmp_pkt = mpls_packet(fuzzy, inner_icmp_pkt, mpls_inputs)
            if inner_icmp_pkt != None and mpls_icmp_pkt != None:
                logger.info("Inner ICMP packet built")
                logger.info("MPLS ICMP Packet built")
                mpls_icmp_pkt.show()
                return mpls_icmp_pkt, mpls_inputs[4], mpls_inputs[5]
            else:
                return None
        else:
            logger.critical(
                "Invalid input '{}' Expected string (y/n)".format(fuzzy))
            return None     
    elif msg_type == 'MPLS_UDP':
        # Get input parameters
        mpls_input_param, common_param = requires("MPLS")
        udp_input_param, _ = requires("UDP")
        inner_udp_pkt, mpls_udp_pkt = None, None
        fuzzy = (input("Generate random MPLS UDP Packet? (y/n) > ").strip()).lower()
        if fuzzy == 'y':
            # Build the inner UDP packet
            # Common parameters
            inputs = []
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            inner_udp_pkt = udp_packet(fuzzy, 'MPLS', None)
            # Generate the Mpls packet
            mpls_udp_pkt = mpls_packet(fuzzy, inner_udp_pkt, inputs) 
            if inner_udp_pkt != None and mpls_udp_pkt != None:
                logger.info("Inner UDP packet built")
                logger.info("MPLS UDP Packet built")
                mpls_udp_pkt.show() 
                return mpls_udp_pkt, inputs[0], inputs[1]
            else:
                return None  
        elif fuzzy == 'n':
            labels = None
            # Getting input parameters
            inputs = []
            del udp_input_param[-1]  # Skipping VLAN Tag for inner UDP
            for i in range(0, len(udp_input_param)):
                temp_input = input("Inner {} > ".format(udp_input_param[i]))
                inputs.insert(i, temp_input)
            inner_udp_pkt = udp_packet(fuzzy, 'MPLS', inputs)
            # Craft the MPLS packet. Get mpls input parameters
            mpls_inputs = []
            for i in range(0, len(mpls_input_param)):
                temp_input = input("{} > ".format(mpls_input_param[i]))
                if "Labels" in mpls_input_param[i]:
                    #using a list comprehension with dummy variable x to create a list out of the labels (which was a string)
                    labels = [x.strip() for x in temp_input.split(',')]
                    for k in range(0, len(labels)):
                        try:
                            if (0 <= int(labels[k]) <= 1048575) is False: #max MPLS label = 2^20
                                logger.error(
                                    "Invalid input '{}' Expected label in range (0-1048575)".format(labels[k])) 
                                return None 
                        except ValueError:
                            logger.error(
                                "Invalid input '{}' Expected comma separated values in range (0-1048575)".format(labels[k]))
                            return None                  
                    mpls_inputs.insert(i,labels)
                elif "Control Word" in mpls_input_param[i] and temp_input.lower() == "y":
                    mpls_inputs.insert(i, True)
                elif "Control Word" in mpls_input_param[i] and temp_input.lower() == "n":
                    mpls_inputs.insert(i, False)
                elif "Control Word" not in mpls_input_param[i]:
                    mpls_inputs.insert(i, temp_input)
                else:
                    logger.critical("Invalid choice, got '{}' expected values (y/n)".format(temp_input))
                    return None 
            # Common parameters
            for j in range(0, len(common_param)):
                i = i + 1
                mpls_inputs.insert(i, input("{} > ".format(common_param[j])))
            mpls_udp_pkt = mpls_packet(fuzzy, inner_udp_pkt, mpls_inputs)
            if inner_udp_pkt != None and mpls_udp_pkt != None:
                logger.info("Inner UDP packet built")
                logger.info("MPLS UDP Packet built")
                mpls_udp_pkt.show()
                return mpls_udp_pkt, mpls_inputs[4], mpls_inputs[5]
            else:
                return None
        else:
            logger.critical(
                "Invalid input '{}' Expected string (y/n)".format(fuzzy))
            return None       
    elif msg_type == 'MPLS_TCP':
        # Get input parameters
        mpls_input_param, common_param = requires("MPLS")
        tcp_input_param, _ = requires("TCP")
        inner_tcp_pkt, mpls_tcp_pkt = None, None
        fuzzy = (input("Generate random MPLS TCP Packet? (y/n) > ").strip()).lower()
        if fuzzy == 'y':
            # Build the inner TCP packet
            # Common parameters
            inputs = []
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            inner_tcp_pkt = tcp_packet(fuzzy, 'MPLS', None)
            # Generate the Mpls packet
            mpls_tcp_pkt = mpls_packet(fuzzy, inner_tcp_pkt, inputs) 
            if inner_tcp_pkt != None and mpls_tcp_pkt != None:
                logger.info("Inner TCP packet built")
                logger.info("MPLS TCP Packet built")
                mpls_tcp_pkt.show() 
                return mpls_tcp_pkt, inputs[0], inputs[1]
            else:
                return None  
        elif fuzzy == 'n':
            labels = None
            # Getting input parameters
            inputs = []
            del tcp_input_param[-1]  # Skipping VLAN Tag for inner UDP
            for i in range(0, len(tcp_input_param)):
                temp_input = input("Inner {} > ".format(tcp_input_param[i]))
                inputs.insert(i, temp_input)
            inner_tcp_pkt = tcp_packet(fuzzy, 'MPLS', inputs)
            # Craft the MPLS packet. Get mpls input parameters
            mpls_inputs = []
            for i in range(0, len(mpls_input_param)):
                temp_input = input("{} > ".format(mpls_input_param[i]))
                if "Labels" in mpls_input_param[i]:
                    #using a list comprehension with dummy variable x to create a list out of the labels (which was a string)
                    labels = [x.strip() for x in temp_input.split(',')]
                    for k in range(0, len(labels)):
                        try:
                            if (0 <= int(labels[k]) <= 1048575) is False: #max MPLS label = 2^20
                                logger.error(
                                    "Invalid input '{}' Expected label in range (0-1048575)".format(labels[k]))
                                return None
                        except ValueError:
                            logger.error(
                                "Invalid input '{}' Expected comma separated values in range (0-1048575)".format(labels[k]))
                            return None                  
                    mpls_inputs.insert(i,labels)
                elif "Control Word" in mpls_input_param[i] and temp_input.lower() == "y":
                    mpls_inputs.insert(i, True)
                elif "Control Word" in mpls_input_param[i] and temp_input.lower() == "n":
                    mpls_inputs.insert(i, False)
                elif "Control Word" not in mpls_input_param[i]:
                    mpls_inputs.insert(i, temp_input)
                else:
                    logger.critical("Invalid choice, got '{}' expected values (y/n)".format(temp_input))
                    return None 
            # Common parameters
            for j in range(0, len(common_param)):
                i = i + 1
                mpls_inputs.insert(i, input("{} > ".format(common_param[j])))
            mpls_tcp_pkt = mpls_packet(fuzzy, inner_tcp_pkt, mpls_inputs)
            if inner_tcp_pkt != None and mpls_tcp_pkt != None:
                logger.info("Inner TCP packet built")
                logger.info("MPLS TCP Packet built")
                mpls_tcp_pkt.show()
                return mpls_tcp_pkt, mpls_inputs[4], mpls_inputs[5]
            else:
                return None
        else:
            logger.critical(
                "Invalid input '{}' Expected string (y/n)".format(fuzzy))
            return None
    elif msg_type == 'MPLS_ARP':
        # Get input parameters
        mpls_input_param, common_param = requires("MPLS")
        arp_input_param, _ = requires('ARP')
        inner_arp_pkt, mpls_arp_pkt = None, None
        fuzzy = (input("Generate random MPLS ARP Packet? (y/n) > ").strip()).lower()
        if fuzzy == 'y':
            # Build inner ARP Packet
            arp_type = (input("ARP Type (req/resp) > ").strip()).lower()
            inputs = []
            # Common parameters
            for i in range(0, len(common_param)):
                inputs.insert(i, input("{} > ".format(common_param[i])))
            inner_arp_pkt = arp_packet(fuzzy, 'MPLS', arp_type, inputs)
            # Generate MPLS packet
            mpls_arp_pkt = mpls_packet(fuzzy, inner_arp_pkt, inputs)
            if inner_arp_pkt != None and mpls_arp_pkt != None:
                logger.info("Inner ARP packet built")
                logger.info("MPLS ARP packet built")
                mpls_arp_pkt.show()
                return mpls_arp_pkt, inputs[0], inputs[1]
        elif fuzzy == 'n':
            # Build inner ARP packet
            arp_type = (input("ARP Type (req/resp) > ").strip()).lower()
            inputs = []
            del arp_input_param[-1]  # Skipping VLAN tag for inner ARP
            for i in range(0, len(arp_input_param)):
                if arp_type == 'req' and (arp_input_param[i] in ['Destination MAC', 'Target MAC']): #skipping Target & Destination MAC as known for ARP Request
                    inputs.insert(i, None)
                    continue
                temp_input = input("Inner {} > ".format(arp_input_param[i]))
                inputs.insert(i, temp_input)
            inner_arp_pkt = arp_packet(fuzzy, 'MPLS', arp_type, inputs)
            # Craft the MPLS packet. Get mpls input params
            mpls_inputs = []
            for i in range(0, len(mpls_input_param)):
                temp_input = input("{} > ".format(mpls_input_param[i]))
                if "Labels" in mpls_input_param[i]:
                    #using a list comprehension with dummy variable x to create a list out of the labels (which was a string)
                    labels = [x.strip() for x in temp_input.split(',')] 
                    for k in range(0, len(labels)):
                        try:
                            if (0 <= int(labels[k]) <= 1048575) is False: #max MPLS label = 2^20
                                logger.error(
                                    "Invalid input '{}' Expected label in range (0-1048575)".format(labels[k]))
                                return None
                        except ValueError:
                            logger.error(
                                "Invalid input '{}' Expected comma separated values in range (0-1048575)".format(labels[k]))
                            return None                  
                    mpls_inputs.insert(i,labels)
                elif "Control Word" in mpls_input_param[i] and temp_input.lower() == "y":
                    mpls_inputs.insert(i, True)
                elif "Control Word" in mpls_input_param[i] and temp_input.lower() == "n":
                    mpls_inputs.insert(i, False)
                elif "Control Word" not in mpls_input_param[i]:
                    mpls_inputs.insert(i, temp_input)
                else:
                    logger.critical("Invalid choice, got '{}' expected values (y/n)".format(temp_input))
                    return None 
            # Common parameters
            for j in range(0, len(common_param)):
                i = i + 1
                mpls_inputs.insert(i, input("{} > ".format(common_param[j])))
            mpls_arp_pkt = mpls_packet(fuzzy, inner_arp_pkt, mpls_inputs)
            if inner_arp_pkt != None and mpls_arp_pkt != None:
                logger.info("Inner ARP packet built")
                logger.info("MPLS ARP Packet built")
                mpls_arp_pkt.show()
                return mpls_arp_pkt, mpls_inputs[4], mpls_inputs[5]
        else:
            logger.critical(
                "Invalid input '{}' Expected string (y/n)".format(fuzzy))
            return None
    else:
        logger.critical("Invalid msg_type: '{}' provided.".format(msg_type))
        return None

#################################################################################################################
def mpls():
    avail_mpls_modules = {
        1: 'MPLS - Inner ICMP',
        2: 'MPLS - Inner UDP',
        3: 'MPLS - Inner TCP',
        4: 'MPLS - Inner ARP',
    }
    print('Packet Type:\n')
    for key in avail_mpls_modules.keys():
        print(key, '--', avail_mpls_modules[key])
    try:
        msg_type = int(input("\nEnter your choice (1-4) > ").strip())
        if msg_type == 1:
            return build_mpls("MPLS_ICMP")
        elif msg_type == 2:
            return build_mpls("MPLS_UDP")
        elif msg_type == 3:
            return build_mpls("MPLS_TCP")
        elif msg_type == 4:
            return build_mpls("MPLS_ARP")
        else:
            logger.critical("Invalid msg_type, expected integer (1-4)")
            return None
    except ValueError:
        logger.critical("Invalid msg_type, expected integer (1-4)")
        return None

#################################################################################################################
def flow_control_packet(fuzzy, module_type, module_inputs):
    final_frame = None
    if module_type not in ['LLFC', 'PFC']:
        logger.critical(
            "Invalid flow control type: '{}' Expected value (LLFC/PFC)".format(
                module_type))
        return None
    if fuzzy == "y":
        if module_type == 'LLFC':
            src_mac, dst_mac = RandMAC()._fix(), MACControl.DEFAULT_DST_MAC
            time_quanta = randint(0, 65535)
        elif module_type == 'PFC':
            src_mac, dst_mac = RandMAC()._fix(), MACControl.DEFAULT_DST_MAC
            # Randomly set the CEV
            class_enable_vector = [
                random.choice([True, False]),  #C0
                random.choice([True, False]),  #C1
                random.choice([True, False]),  #C2
                random.choice([True, False]),  #C3
                random.choice([True, False]),  #C4
                random.choice([True, False]),  #C5
                random.choice([True, False]),  #C6
                random.choice([True, False])   #C7
            ]
            # Randomly set pause time per class
            class_pause_times = [
                randint(1, 65535) if class_enable_vector[0] else 0,  #C0
                randint(1, 65535) if class_enable_vector[1] else 0,  #C1
                randint(1, 65535) if class_enable_vector[2] else 0,  #C2
                randint(1, 65535) if class_enable_vector[3] else 0,  #C3
                randint(1, 65535) if class_enable_vector[4] else 0,  #C4
                randint(1, 65535) if class_enable_vector[5] else 0,  #C5
                randint(1, 65535) if class_enable_vector[6] else 0,  #C6
                randint(1, 65535) if class_enable_vector[7] else 0   #C7
            ]
        else:
            return None
    if fuzzy == "n":
        if module_type == 'LLFC':
            mac_pattern = re.compile(
                r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
            # If no valid source mac provided get it from source interface
            if mac_pattern.match(module_inputs[0].strip()):
                src_mac = module_inputs[0]
            else:
                logger.error(
                    "Invalid source MAC provided. Extracting source MAC from source interface."
                )
                src_mac = get_if_hwaddr(module_inputs[3])
            # Destination MAC is well known reserved 01-80-C2-00-00-01
            dst_mac = MACControl.DEFAULT_DST_MAC
            try:
                time_quanta = int(module_inputs[1])
            except ValueError:
                logger.critical(
                    "Invalid input '{}' Expected integer (0-65535)".format(module_inputs[1]))
                logger.critical(ValueError, exc_info=True)
                return None
            if time_quanta < 0 or time_quanta > 65535:
                logger.critical(
                    "Invalid Quanta value provided '{}' Expected value in range (0-65535)".format(time_quanta))
                return None
        elif module_type == 'PFC':
            mac_pattern = re.compile(
                r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
            # If no valid source mac provided get it from source interface
            if mac_pattern.match(module_inputs[0].strip()):
                src_mac = module_inputs[0]
            else:
                logger.error(
                    "Invalid source MAC provided. Extracting source MAC from source interface."
                )
                src_mac = get_if_hwaddr(module_inputs[4])
            # Destination MAC is well known reserved 01-80-C2-00-00-01
            dst_mac = MACControl.DEFAULT_DST_MAC
            cev_input = (module_inputs[1].strip()).split(",")
            cev_input = [x.strip().upper() for x in cev_input]
            quanta_input = (module_inputs[2].strip()).split(",")
            quanta_input = [x.strip().upper() for x in quanta_input]
            class_enable_vector, class_pause_times = validate_cev_quanta(cev_input, quanta_input)
            if class_enable_vector == None or class_pause_times == None:
                return None
    if module_type == 'LLFC':
        final_frame = Ether(src=src_mac, dst=dst_mac) / MACControlPause(pause_time=time_quanta)
    elif module_type == 'PFC':
        final_frame = Ether(src=src_mac, dst=dst_mac, type=0x8808)
        pfc_opcode = 0x0101
        cev_value = sum(2**i if class_enable_vector[i] else 0 for i in range(8))
        class_pause_times_bytes = b''.join([time.to_bytes(2, byteorder='big') for time in class_pause_times])
        pfc_payload = struct.pack("!H", pfc_opcode) + cev_value.to_bytes(2, byteorder='big') + class_pause_times_bytes
        padding_length = 46 - len(pfc_payload)
        padding = Padding(load=b'\x00' * padding_length)
        final_frame = final_frame / pfc_payload / padding
    return final_frame


#################################################################################################################
def validate_cev_quanta(cev, quanta):
    class_names = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    default_cev = [False] * 8
    default_quanta = [0] * 8
    if len(quanta) > len(cev):
        logger.error("Mismatched Quanta and Class inputs")
        return None, None
    # Default if both inputs are empty
    elif len(quanta) <= len(cev):
        if (len(quanta) == 1 and len(cev) == 1) and (cev[0] == "" and quanta[0] == ""):
            return default_cev, default_quanta
        for class_id in cev:
            if class_id not in class_names:
                logger.error(
                    "Invalid Class value '{}' Expected comma separated values in range C0 - C7".format(class_id))
                return None, None
        if len(quanta) == 1 and quanta[0] == "":
            _ = quanta.pop(0)
        quanta.extend([0] * (len(cev) - len(quanta)))
        try:
            quanta = [int(i) for i in quanta]
        except ValueError:
            logger.critical(
                "Invalid Quanta value '{}' Expected integer in range 0 - 65535".format(quanta))
            logger.critical(ValueError, exc_info=True)
            return None, None
        if (any((i > 65535) or (i < 0) for i in quanta)):
            logger.error(
                "Invalid Quanta value '{}' Expected value in range 0 - 65535".format(quanta))
            return None, None
        for i in range(0, len(cev)):
            default_cev[class_names.index(cev[i])] = True
            default_quanta[class_names.index(cev[i])] = quanta[i]
        return default_cev, default_quanta


#################################################################################################################
def build_llfc():
    # Get input parameters
    llfc_pkt = None
    input_param, common_param = requires("LLFC")
    fuzzy = (input("Random 802.3x Pause Frame? (y/n) > ").strip()).lower()
    if fuzzy == "y":
        inputs = []
        # Common parameters
        for i in range(0, len(common_param)):
            inputs.insert(i, input("{} > ".format(common_param[i])))
        llfc_pkt = flow_control_packet(fuzzy, 'LLFC', inputs)
        if llfc_pkt != None:
            logger.info("802.3x Pause Frame built")
            llfc_pkt.show()
            return llfc_pkt, inputs[0], inputs[1]
        else:
            return None
    elif fuzzy == "n":
        inputs = []
        # Get input parameters
        for i in range(0, len(input_param)):
            inputs.insert(i, input("{} > ".format(input_param[i])))
        # Common parameters
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))
        llfc_pkt = flow_control_packet(fuzzy, 'LLFC', inputs)
        if llfc_pkt != None:
            logger.info("802.3x Pause Frame built")
            llfc_pkt.show()
            return llfc_pkt, inputs[2], inputs[3]
        else:
            return None
    else:
        logger.critical(
            "Invalid input '{}' Expected string (y/n)".format(fuzzy))
        return None


#################################################################################################################
def build_pfc():
    # Get input parameters
    pfc_pkt = None
    input_param, common_param = requires("PFC")
    fuzzy = (input("Random 802.1Qbb PFC Frame? (y/n) > ").strip()).lower()
    if fuzzy == "y":
        inputs = []
        # Common parameters
        for i in range(0, len(common_param)):
            inputs.insert(i, input("{} > ".format(common_param[i])))
        pfc_pkt = flow_control_packet(fuzzy, 'PFC', inputs)
        if pfc_pkt != None:
            logger.info("802.1Qbb PFC Frame built")
            pfc_pkt.show()
            return pfc_pkt, inputs[0], inputs[1]
        else:
            return None
    elif fuzzy == "n":
        inputs = []
        # Get inputs parameters
        for i in range(0, len(input_param)):
            inputs.insert(i, input("{} > ".format(input_param[i])))
        for j in range(0, len(common_param)):
            i = i + 1
            inputs.insert(i, input("{} > ".format(common_param[j])))
        pfc_pkt = flow_control_packet(fuzzy, 'PFC', inputs)
        if pfc_pkt != None:
            logger.info("802.1Qbb PFC Frame built")
            pfc_pkt.show()
            return pfc_pkt, inputs[3], inputs[4]
        else:
            return None
    else:
        logger.critical(
            "Invalid input '{}' Expected string (y/n)".format(fuzzy))
        return None


#################################################################################################################
def callModule(module_number):
    try:
        if 1 <= module_number <= 8:
            flow_arr = []
            flow_count = int(input("Enter the number of flows > ").strip())
            for index in range(0, flow_count):
                flow_instance = None
                print("\nBuilding flow number [ {} ]:\n".format(index + 1))
                if module_number == 1:
                    flow_instance = build_icmp()
                elif module_number == 2:
                    flow_instance = build_arp()
                elif module_number == 3:
                    flow_instance = igmp()
                elif module_number == 4:
                    flow_instance = build_mcast()
                elif module_number == 5:
                    flow_instance = vxlan()
                elif module_number == 6:
                    flow_instance = build_llfc()
                elif module_number == 7:
                    flow_instance = build_pfc()
                elif module_number == 8:
                    flow_instance = mpls()
                if flow_instance != None:
                    flow_arr.append(flow_instance)
            if len(flow_arr) > 0:
                logger.info("Sending out all flows")
                send_flows(flow_arr)
            else:
                logger.info("No valid flows found to send.")
            del flow_arr
        elif module_number == 9:
            _ = pcap_mod()
        else:
            logger.critical(
                "Invalid module number provided '{}'".format(module_number))
            return None
    except ValueError:
        logger.error("Invalid flow count. Expected integer")
        logger.error(ValueError, exc_info=True)
    gc.collect()
    logger.info("Module completed")


#################################################################################################################
def print_menu():
    available_mods = {
        1: 'ICMP',
        2: 'ARP',
        3: 'IGMP',
        4: 'Multicast',
        5: 'VXLAN',
        6: 'Pause Frame',
        7: 'Priority Flow Control',
        8: 'MPLS',
        9: 'Load PCAP File',
        10: 'Exit',
    }
    print("\n" + '=' * 50)
    print('Scapy based packet generator')
    print('=' * 50 + "\n")
    for key in available_mods.keys():
        print(key, '--', available_mods[key])


#################################################################################################################
class MyFormatter(logging.Formatter):
    err_fmt = "%(asctime)s: %(levelname)s: %(message)s, at line %(lineno)d, in %(funcName)s()"
    crit_fmt = "%(asctime)s: %(levelname)s: %(message)s, at line %(lineno)d, in %(funcName)s()"
    info_fmt = "%(asctime)s: %(levelname)s: %(message)s"

    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')

    def format(self, record):
        format_orig = self._style._fmt
        if record.levelno == logging.CRITICAL:
            self._style._fmt = MyFormatter.crit_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = MyFormatter.info_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = MyFormatter.err_fmt
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


fmt = MyFormatter()
hdlr = logging.StreamHandler(sys.stdout)
hdlr.setFormatter(fmt)
logging.root.addHandler(hdlr)
logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

#################################################################################################################
if __name__ == "__main__":
    call_mod = None
    while (call_mod != 10):
        print_menu()
        try:
            call_mod = int((input("\nEnter your choice (1-10): ")).strip())
            if 1 <= call_mod <= 9:
                callModule(call_mod)
            elif call_mod == 10:
                logger.info("See you later, alligator!")
                sys.exit(0)
            else:
                logger.info('Invalid input. Please select a number (1-10)')
        except ValueError:
            logger.info('Invalid input. Please select a number (1-10)')
