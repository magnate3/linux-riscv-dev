#!/usr/bin/python

# Copyright 2013-present Barefoot Networks, Inc. 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scapy.all import sniff, sendp
from scapy.all import Packet
from scapy.all import ShortField, IntField, LongField, BitField
from scapy.all import *

from falconpkts.pkts import *

import sys
import struct


class Worker:
    def __init__(self, ip_address, rack_local_id, global_id):
        self.ip_address = ip_address
        self.rack_local_id = rack_local_id
        self.global_id = global_id
        print (ip_address)

    def receive_pkt(self):
        sniff(filter=("ip src not %s and udp" % self.ip_address), prn = lambda x: self.handle_pkt(x))

    def handle_pkt(self, pkt):
        eth_kwargs = {
            'src_mac': '01:02:03:04:05:06',
            'dst_mac': 'AA:BB:CC:DD:EE:FF'
        }
        if not pkt[UDP]:
            print("Packet not UDP")
            return

        if (pkt[UDP].dport==1234):
            rcv_hdr = FalconPacket(pkt[UDP].payload)

            rcv_hdr.show()
            print ("received packet")
            
            #task_done_pkt = make_falcon_task_done_pkt(dst_ip=pkt[IP].src, cluster_id=rcv_hdr.cluster_id, local_cluster_id=rcv_hdr.local_cluster_id, src_id=int(self.rack_local_id), **eth_kwargs)
            
            #print('>> Task Done packet (size = %d bytes):' % len(task_done_pkt))
            
            #task_done_pkt.show()

            #send(task_done_pkt)

if __name__ == '__main__':
    host_ip = sys.argv[1]
    local_id = sys.argv[2]
    global_id = sys.argv[3]
    
    worker = Worker(host_ip, local_id, global_id)
    worker.receive_pkt()
