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

#from socket import *
from scapy.all import *
from scapy.all import sniff, sendp
from scapy.all import Packet
from scapy.all import ShortField, IntField, LongField, BitField

from falconpkts.pkts import *

import sys

num_tasks = 5

class SrcRoute(Packet):
    name = "SrcRoute"
    fields_desc = [
        LongField("preamble", 0),
        IntField("num_valid", 0)
    ]

def read_topo():
    nb_hosts = 0
    nb_switches = 0
    links = []
    with open("topo.txt", "r") as f:
        line = f.readline()[:-1]
        w, nb_switches = line.split()
        assert(w == "switches")
        line = f.readline()[:-1]
        w, nb_hosts = line.split()
        assert(w == "hosts")
        for line in f:
            if not f: break
            a, b = line.split()
            links.append( (a, b) )
    return int(nb_hosts), int(nb_switches), links

def main():
    eth_kwargs = {
        'src_mac': '01:02:03:04:05:06',
        'dst_mac': 'AA:BB:CC:DD:EE:FF'
    }

    _dst_ip = '10.0.2.101'

    for i in range(num_tasks):
        new_task_packet = make_falcon_task_pkt(dst_ip=_dst_ip, cluster_id=5, local_cluster_id=1, src_id=6, seq_num=0x10+i, **eth_kwargs)

        print('>> New Task packet (size = %d bytes):' % len(new_task_packet))
        #new_task_packet.show()

        send(new_task_packet)


if __name__ == '__main__':
    main()
