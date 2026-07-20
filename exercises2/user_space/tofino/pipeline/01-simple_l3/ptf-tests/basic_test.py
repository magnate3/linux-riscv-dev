################################################################################
# BAREFOOT NETWORKS CONFIDENTIAL & PROPRIETARY
#
# Copyright (c) 2018-2019 Barefoot Networks, Inc.

# All Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of
# Barefoot Networks, Inc. and its suppliers, if any. The intellectual and
# technical concepts contained herein are proprietary to Barefoot Networks,
# Inc.
# and its suppliers and may be covered by U.S. and Foreign Patents, patents in
# process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# Barefoot Networks, Inc.
#
# No warranty, explicit or implicit is provided, unless granted under a
# written agreement with Barefoot Networks, Inc.
#
#
###############################################################################

"""
Basic functional tests for the program simple_l3.p4
"""

# The simple_l3.py file in the same directory contains the foundational class,
# that is used for these individual tests

from simple_l3 import *


############################################################################
################# I N D I V I D U A L    T E S T S #########################
############################################################################

class HostForward(P4ProgramTest):
    """
    Basic forwarding (send() action) via ipv4_host table
    """
    
    # The runTest() method represents the test itself. 
    #
    # Typically you would want to configure the device (e.g. by populating
    # the tables), send some traffic and check the results.
    #
    # For more flexible checks, you can import unittest module and use
    # the provided methods, such as unittest.assertEqual()
    #
    # Do not enclose the code into try/except/finally -- this is done by
    # the framework itself
    
    def runTest(self):
        ingress_port = test_param_get("ingress_port",  0)
        egress_port  = test_param_get("egress_port",   4)
        ipv4_dst     = test_param_get("ipv4_dst",      "192.168.1.1")

        print("\n")
        print("Test Run")
        print("========")

        #
        # Program an entry in IPv4: ipv4_dst --> send(egress_port)
        #
        key_list = [
            self.ipv4_host.make_key([
                gc.KeyTuple('hdr.ipv4.dst_addr', ipv4_dst)])
            ]

        data_list = [
            self.ipv4_host.make_data([
                gc.DataTuple('port', egress_port)], "Ingress.send")
            ]

        self.ipv4_host.entry_add(self.dev_tgt, key_list, data_list);
        print("  Added an entry to ipv4_host: {} --> send({})".format(
            ipv4_dst, egress_port))

        #
        # Send a test packet
        #
        print("  Sending packet with IPv4 DST ADDR=%s into port %d" %
              (ipv4_dst, ingress_port))
        pkt = simple_tcp_packet(eth_dst="00:98:76:54:32:10",
                                eth_src='00:55:55:55:55:55',
                                ip_dst=ipv4_dst,
                                ip_id=101,
                                ip_ttl=64,
                                ip_ihl=5)
        send_packet(self, ingress_port, pkt)

        #
        # Wait for the egress packet and verify it
        #
        print("  Expecting the packet to be forwarded to port %d" % egress_port)
        verify_packet(self, pkt, egress_port)
        print("  Packet received of port %d" % egress_port)

        ############# That's it! ###############

class HostDrop(P4ProgramTest):
    """
    Basic test for dropping via ipv4_host table
    """
    def runTest(self):
        ingress_port = test_param_get("ingress_port",  0)
        ipv4_dst     = test_param_get("ipv4_dst",      "192.168.1.1")

        print("\n")
        print("Test Run")
        print("========")
        
        #
        # Program an entry in IPv4: ipv4_dst --> drop()
        #
        key_list = [
            self.ipv4_host.make_key([
                gc.KeyTuple('hdr.ipv4.dst_addr', ipv4_dst)])
            ]

        data_list = [
            self.ipv4_host.make_data([], "Ingress.drop")
            ]

        self.ipv4_host.entry_add(self.dev_tgt, key_list, data_list);
        print("  Added an entry to ipv4_host: {} --> drop()".format(
            ipv4_dst))

        #
        # Send a test packet
        #
        print("  Sending packet with IPv4 DST ADDR=%s into port %d" %
              (ipv4_dst, ingress_port))
        pkt = simple_tcp_packet(eth_dst="00:98:76:54:32:10",
                                eth_src='00:55:55:55:55:55',
                                ip_dst=ipv4_dst,
                                ip_id=101,
                                ip_ttl=64,
                                ip_ihl=5)
        send_packet(self, ingress_port, pkt)
        print("  Expecting No packets anywhere")
        
        #
        # Wait to make sure no packet egresses anywhere. self.swports is the
        # list of all ports the test is using. It is set up in the setUp()
        # method of the parent class
        #
        verify_no_packet_any(self, pkt, self.swports)
        print("  No packets received")

class LpmForward(P4ProgramTest):
    """
    Basic forwarding (send() action) via ipv4_lpm table.
    Test params:
       ingress_port -- the port used for the test packet injection
       lpm_addr     -- IP address/mask to program in the ipv4_lpm table
       ipv4_dst     -- destination IPv4 address for the packet
       egress_port  -- egress port for the send() action
    """
    def runTest(self):
        ingress_port = test_param_get("ingress_port",  0)
        egress_port  = test_param_get("egress_port",   4)
        lpm_addr     = test_param_get("lpm_addr",      "192.168.1.0/24")
        ipv4_dst     = test_param_get("ipv4_dst",      "192.168.1.1")
        egress_port  = test_param_get("egress_port",   4)

        (lpm_ipv4, lpm_p_len) = lpm_addr.split("/")
        
        print("\n")
        print("Test Run")
        print("========")
        
        #
        # Program an entry in IPv4: lpm_addr/prefix --> send(egress_port)
        # Note the extra named argument (prefix_len) to gc.KeyTuple(). 
        #
        key_list = [
            self.ipv4_lpm.make_key([
                gc.KeyTuple('hdr.ipv4.dst_addr',
                            value=lpm_ipv4, prefix_len=int(lpm_p_len))])
            ]

        data_list = [
            self.ipv4_lpm.make_data([
                gc.DataTuple('port', egress_port)], "Ingress.send")
            ]

        print("\n")
        print("Test Run")
        print("========")
        self.ipv4_lpm.entry_add(self.dev_tgt, key_list, data_list);
        print("  Added an entry to ipv4_lpm: {}/{} --> send({})".format(
            lpm_ipv4, lpm_p_len, egress_port))

        #
        # Send a test packet
        #
        print("  Sending packet with IPv4 DST ADDR=%s into port %d" %
              (ipv4_dst, ingress_port))
        pkt = simple_tcp_packet(eth_dst="00:98:76:54:32:10",
                                eth_src='00:55:55:55:55:55',
                                ip_dst=ipv4_dst,
                                ip_id=101,
                                ip_ttl=64,
                                ip_ihl=5)
        send_packet(self, ingress_port, pkt)
        
        #
        # Wait for the egress packet and verify it. Since we do not expect
        # any modifications, we use the same pkt. 
        #
        print("  Expecting the packet to be forwarded to port %d" % egress_port)
        verify_packet(self, pkt, egress_port)
        print("  Packet received of port %d" % egress_port)

class LpmDrop(P4ProgramTest):
    """
    Basic test for dropping via ipv4_lpm table
    """
    def runTest(self):
        ingress_port = test_param_get("ingress_port",  0)
        lpm_addr     = test_param_get("lpm_addr",      "192.168.1.0/24")
        ipv4_dst     = test_param_get("ipv4_dst",      "192.168.1.1")

        (lpm_ipv4, lpm_p_len) = lpm_addr.split("/")
        
        print("\n")
        print("Test Run")
        print("========")

        #
        # Program an entry in IPv4: lpm_addr/prefix --> drop()
        # Note the extra named argument (prefix_len) to gc.KeyTuple(). 
        #
        key_list = [
            self.ipv4_lpm.make_key([
                gc.KeyTuple('hdr.ipv4.dst_addr',
                            value=lpm_ipv4, prefix_len=int(lpm_p_len))])
            ]

        data_list = [
            self.ipv4_lpm.make_data([], "Ingress.drop")
            ]

        self.ipv4_lpm.entry_add(self.dev_tgt, key_list, data_list);
        print("  Added an entry to ipv4_host: {} --> drop()".format(
            ipv4_dst))

        print("  Sending packet with IPv4 DST ADDR=%s into port %d" %
              (ipv4_dst, ingress_port))
        pkt = simple_tcp_packet(eth_dst="00:98:76:54:32:10",
                                eth_src='00:55:55:55:55:55',
                                ip_dst=ipv4_dst,
                                ip_id=101,
                                ip_ttl=64,
                                ip_ihl=5)
        send_packet(self, ingress_port, pkt)
        print("  Expecting No packets anywhere")
        verify_no_packet_any(self, pkt, self.swports)
        print("  No packets received")

class NonIPv4(P4ProgramTest):
   """
   Basic test for dropping a non-IPv4 packet
   """
   def runTest(self):
       ingress_port = test_param_get("ingress_port",  0)
       
       print("\n")
       print("Test Run")
       print("========")
       
       print("  Sending IPv6 packet into port %d" %
             (ingress_port))
       
       pkt = simple_tcpv6_packet()
       
       send_packet(self, ingress_port, pkt)
       print("  Expecting No packets anywhere")
       verify_no_packet_any(self, pkt, self.swports)
       print("  No packets received")
