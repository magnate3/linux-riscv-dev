.. Copyright (c) <2016>, Intel Corporation
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.

   - Neither the name of Intel Corporation nor the names of its
     contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
   OF THE POSSIBILITY OF SUCH DAMAGE.

=======================
Generic filter/flow api
=======================

Prerequisites
=============

1. Hardware:
   Fortville and Niantic
  
2. software: 
   dpdk: http://dpdk.org/git/dpdk
   scapy: http://www.secdev.org/projects/scapy/

3. bind the pf to dpdk driver::

    ./usertools/dpdk-devbind.py -b igb_uio 05:00.0
 

Test case: Fortville ethertype
==============================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules::

    testpmd> flow validate 0 ingress pattern eth type is 0x0806 / end actions queue index 1 / end
    testpmd> flow create 0 ingress pattern eth type is 0x0806 / end actions queue index 2 / end
    testpmd> flow validate 0 ingress pattern eth type is 0x08bb / end actions queue index 16 / end
    testpmd> flow create 0 ingress pattern eth type is 0x88bb / end actions queue index 3 / end
    testpmd> flow create 0 ingress pattern eth dst is 00:11:22:33:44:55 type is 0x88e5 / end actions queue index 4 / end
    testpmd> flow create 0 ingress pattern eth type is 0x8864 / end actions drop / end
    testpmd> flow validate 0 ingress pattern eth type is 0x88cc / end actions queue index 5 / end
    testpmd> flow create 0 ingress pattern eth type is 0x88cc / end actions queue index 6 / end

   the i40e don't support the 0x88cc eth type packet. so the last two commands
   failed.

3. send packets::

    pkt1 = Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(pdst="192.168.1.1")
    pkt2 = Ether(dst="00:11:22:33:44:55", type=0x88BB)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55", type=0x88e5)/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:55", type=0x8864)/Raw('x' * 20)

   verify pkt1 to queue 2, and pkt2 to queue 3, pkt3 to queue 4, pkt4 dropped.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    verify pkt1 to queue 0, and pkt2 to queue 3, pkt3 to queue 4,
    testpmd> flow list 0
    testpmd> flow flush 0
    verify pkt1 to queue 0, and pkt2 to queue 0, pkt3 to queue 0, pkt4 to queue 0.
    testpmd> flow list 0


Test case: Fortville fdir for L2 payload
========================================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -w 05:00.0 --file-prefix=pf --socket-mem=1024,1024 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 1 / end actions queue index 1 / end
    testpmd> flow create 0 ingress pattern eth type is 0x0807 / end actions queue index 2 / end

3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=1)/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55", type=0x0807)/Dot1Q(vlan=1)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55", type=0x0807)/IP(src="192.168.0.5", dst="192.168.0.6")/Raw('x' * 20)

   check pkt1 to queue 1, pkt2 to queue 2, pkt3 to queue 2.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0


Test case: Fortville fdir for flexbytes
=======================================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -w 05:00.0 --file-prefix=pf --socket-mem=1024,1024 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   l2-payload::

    testpmd> flow create 0 ingress pattern eth type is 0x0807 / raw relative is 1 pattern is ab / end actions queue index 1 / end

   ipv4-other::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 4095 / ipv4 proto is 255 ttl is 40 / raw relative is 1 offset is 2 pattern is ab / raw relative is 1 offset is 10 pattern is abcdefghij / raw relative is 1 offset is 0 pattern is abcd / end actions queue index 2 / end

   ipv4-udp::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 2.2.2.4 dst is 2.2.2.5 / udp src is 22 dst is 23 / raw relative is 1 offset is 2 pattern is fhds / end actions queue index 3 / end

   ipv4-tcp::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 2.2.2.4 dst is 2.2.2.5 tos is 4 ttl is 3 / tcp src is 32 dst is 33 / raw relative is 1 offset is 2 pattern is hijk / end actions queue index 4 / end

   ipv4-sctp::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 2.2.2.4 dst is 2.2.2.5 / sctp src is 42 / raw relative is 1 offset is 2 pattern is abcdefghijklmnop / end actions queue index 5 / end

   ipv6-tcp::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 1 / ipv6 src is 2001::1 dst is 2001::2 tc is 3 hop is 30 / tcp src is 32 dst is 33 / raw relative is 1 offset is 0 pattern is hijk / raw relative is 1 offset is 8 pattern is abcdefgh / end actions queue index 6 / end

   spec-mask(not supportted now, 6wind will update lately)
   restart testpmd, create new rules::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 2.2.2.4 dst is 2.2.2.5 / tcp src is 32 dst is 33 / raw relative is 1 offset is 2 pattern spec \x61\x62\x63\x64 pattern mask \x00\x00\xff\x01 / end actions queue index 7 / end
 
3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:55", type=0x0807)/Raw(load="\x61\x62\x63\x64")
    pkt2 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=4095)/IP(src="192.168.0.1", dst="192.168.0.2", proto=255, ttl=40)/Raw(load="xxabxxxxxxxxxxabcdefghijabcdefg")
    pkt3 = Ether(dst="00:11:22:33:44:55")/IP(src="2.2.2.4", dst="2.2.2.5")/UDP(sport=22,dport=23)/Raw(load="fhfhdsdsfwef")
    pkt4 = Ether(dst="00:11:22:33:44:55")/IP(src="2.2.2.4", dst="2.2.2.5", tos=4, ttl=3)/TCP(sport=32,dport=33)/Raw(load="fhhijk")
    pkt5 = Ether(dst="00:11:22:33:44:55")/IP(src="2.2.2.4", dst="2.2.2.5")/SCTP(sport=42,dport=43,tag=1)/Raw(load="xxabcdefghijklmnopqrst")
    pkt6 = Ether(dst="00:11:22:33:44:55")/IP(src="2.2.2.4", dst="2.2.2.5")/SCTP(sport=42,dport=43,tag=1)/Raw(load="xxabxxxabcddxxabcdefghijklmn")
    pkt7 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=1)/IPv6(src="2001::1", dst="2001::2", tc=3, hlim=30)/TCP(sport=32,dport=33)/Raw(load="hijkabcdefghabcdefghijklmn")

   pkt8-pkt10 are not supported now::

    pkt8 = Ether(dst="00:11:22:33:44:55")/IP(src="2.2.2.4", dst="2.2.2.5")/TCP(sport=32,dport=33)/Raw(load="\x68\x69\x61\x62\x63\x64")
    pkt9 = Ether(dst="00:11:22:33:44:55")/IP(src="2.2.2.4", dst="2.2.2.5")/TCP(sport=32,dport=33)/Raw(load="\x68\x69\x68\x69\x63\x74")
    pkt10 = Ether(dst="00:11:22:33:44:55")/IP(src="2.2.2.4", dst="2.2.2.5")/TCP(sport=32,dport=33)/Raw(load="\x68\x69\x61\x62\x63\x65")

   check pkt1 to pkt5 are received by queue 1 to queue 5, pkt6 to queue 0,
   pkt7 to queue6. pkt8 to queue7, pkt8 and pkt9 to queue 0.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0


Test case: Fortville fdir for ipv4
==================================

   Prerequisites:
   
   add two vfs on dpdk pf, then bind the vfs to vfio-pci::

    echo 2 >/sys/bus/pci/devices/0000:05:00.0/max_vfs
    ./usertools/dpdk-devbind.py -b vfio-pci 05:02.0 05:02.1

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -w 05:00.0 --file-prefix=pf --socket-mem=1024,1024 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e0000 -n 4 -w 05:02.0 --file-prefix=vf0 --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e00000 -n 4 -w 05:02.1 --file-prefix=vf1 --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   ipv4-other::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 proto is 3 / end actions queue index 1 / end

   ipv4-udp::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 ttl is 3 / udp src is 22 dst is 23 / end actions queue index 2 / end

   ipv4-tcp::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 tos is 3 / tcp src is 32 dst is 33 / end actions queue index 3 / end

   ipv4-sctp::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 1 / ipv4 src is 192.168.0.1 dst is 192.168.0.2 tos is 3 ttl is 3 / sctp src is 44 dst is 45 tag is 1 / end actions queue index 4 / end

   ipv4-other-vf0::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 proto is 3 / vf id is 0 / end actions queue index 1 / end

   ipv4-sctp-vf1::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 2 / ipv4 src is 192.168.0.1 dst is 192.168.0.2 tos is 4 ttl is 4 / sctp src is 46 dst is 47 tag is 1 / vf id is 1 / end actions queue index 2 / end

   ipv4-sctp drop::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.5 dst is 192.168.0.6 tos is 3 ttl is 3 / sctp src is 44 dst is 45 tag is 1 / end actions drop / end

   ipv4-sctp passthru-flag::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 3 / ipv4 src is 192.168.0.1 dst is 192.168.0.2 tos is 4 ttl is 4 / sctp src is 44 dst is 45 tag is 1 / end actions passthru / flag / end

   ipv4-udp queue-flag::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 ttl is 4 / udp src is 22 dst is 23 / end actions queue index 5 / flag / end

   ipv4-tcp queue-mark::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 tos is 4 / tcp src is 32 dst is 33 / end actions queue index 6 / mark id 3 / end

   ipv4-other passthru-mark::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.3 dst is 192.168.0.4 proto is 3 / end actions passthru / mark id 4 / end

3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2", proto=3)/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2", ttl=3)/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2", tos=3)/TCP(sport=32,dport=33)/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=1)/IP(src="192.168.0.1", dst="192.168.0.2", tos=3, ttl=3)/SCTP(sport=44,dport=45,tag=1)/SCTPChunkData(data="X" * 20)
    pkt5 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=2)/IP(src="192.168.0.1", dst="192.168.0.2", tos=4, ttl=4)/SCTP(sport=46,dport=47,tag=1)/Raw('x' * 20)
    pkt6 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.5", dst="192.168.0.6", tos=3, ttl=3)/SCTP(sport=44,dport=45,tag=1)/SCTPChunkData(data="X" * 20)
    pkt7 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=3)/IP(src="192.168.0.1", dst="192.168.0.2", tos=4, ttl=4)/SCTP(sport=44,dport=45,tag=1)/Raw('x' * 20)
    pkt8 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2", ttl=4)/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt9 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2", tos=4)/TCP(sport=32,dport=33)/Raw('x' * 20)
    pkt10 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.3", dst="192.168.0.4", proto=3)/Raw('x' * 20)

   verify packet 
   pkt1 to queue 1 and vf0 queue 1, pkt2 to queue 2, pkt3 to queue 3,
   pkt4 to queue 4, pkt5 to vf1 queue 2, pkt6 can't be received by pf.
   if not "--disable-rss",
   pkt7 to queue 0, FDIR matched hash 0 ID 0, pkt8 to queue 5,
   FDIR matched hash 0 ID 0, pkt9 to queue 6, FDIR matched ID 3,
   pkt10 queue determined by rss rule, FDIR matched ID 4.
   if "--disable-rss"
   pkt7-9 has same result with above, pkt10 to queue 0, FDIR matched ID 4.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0


Test case: Fortville fdir for ipv6
==================================

   Prerequisites:

   add two vfs on dpdk pf, then bind the vfs to vfio-pci::

    echo 2 >/sys/bus/pci/devices/0000:05:00.0/max_vfs
    ./usertools/dpdk-devbind.py -b vfio-pci 05:02.0 05:02.1

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -w 05:00.0 --file-prefix=pf --socket-mem=1024,1024 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e0000 -n 4 -w 05:02.0 --file-prefix=vf0 --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e00000 -n 4 -w 05:02.1 --file-prefix=vf1 --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   ipv6-other::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 1 / ipv6 src is 2001::1 dst is 2001::2 tc is 1 proto is 5 hop is 10 / end actions queue index 1 / end

   ipv6-udp::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 2 / ipv6 src is 2001::1 dst is 2001::2 tc is 2 hop is 20 / udp src is 22 dst is 23 / end actions queue index 2 / end

   ipv6-tcp::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 3 / ipv6 src is 2001::1 dst is 2001::2 tc is 3 hop is 30 / tcp src is 32 dst is 33 / end actions queue index 3 / end

   ipv6-sctp::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 4 / ipv6 src is 2001::1 dst is 2001::2 tc is 4 hop is 40 / sctp src is 44 dst is 45 tag is 1 / end actions queue index 4 / end

   ipv6-other-vf0::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 5 / ipv6 src is 2001::3 dst is 2001::4 tc is 5 proto is 5 hop is 50 / vf id is 0 / end actions queue index 1 / end

   ipv6-tcp-vf1::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 4095 / ipv6 src is 2001::3 dst is 2001::4 tc is 6 hop is 60 / tcp src is 32 dst is 33 / vf id is 1 / end actions queue index 3 / end

   ipv6-sctp-drop::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 7 / ipv6 src is 2001::1 dst is 2001::2 tc is 7 hop is 70 / sctp src is 44 dst is 45 tag is 1 / end actions drop / end

   ipv6-tcp-vf1-drop::

    testpmd> flow create 0 ingress pattern eth / vlan tci is 8 / ipv6 src is 2001::3 dst is 2001::4 tc is 8 hop is 80 / tcp src is 32 dst is 33 / vf id is 1 / end actions drop / end

3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=1)/IPv6(src="2001::1", dst="2001::2", tc=1, nh=5, hlim=10)/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=2)/IPv6(src="2001::1", dst="2001::2", tc=2, hlim=20)/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=3)/IPv6(src="2001::1", dst="2001::2", tc=3, hlim=30)/TCP(sport=32,dport=33)/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=4)/IPv6(src="2001::1", dst="2001::2", tc=4, nh=132, hlim=40)/SCTP(sport=44,dport=45,tag=1)/SCTPChunkData(data="X" * 20)
    pkt5 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=5)/IPv6(src="2001::3", dst="2001::4", tc=5, nh=5, hlim=50)/Raw('x' * 20)
    pkt6 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=4095)/IPv6(src="2001::3", dst="2001::4", tc=6, hlim=60)/TCP(sport=32,dport=33)/Raw('x' * 20)
    pkt7 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=7)/IPv6(src="2001::1", dst="2001::2", tc=7, nh=132, hlim=70)/SCTP(sport=44,dport=45,tag=1)/SCTPChunkData(data="X" * 20)
    pkt8 = Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=8)/IPv6(src="2001::3", dst="2001::4", tc=8, hlim=80)/TCP(sport=32,dport=33)/Raw('x' * 20)

   verify packet
   pkt1 to queue 1 and vf queue 1, pkt2 to queue 2, pkt3 to queue 3,
   pkt4 to queue 4, pkt5 to vf0 queue 1, pkt6 to vf1 queue 3,
   pkt7 can't be received by pf, pkt8 can't be received by vf1.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0


Test case: Fortville fdir wrong parameters
==========================================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -w 05:00.0 --file-prefix=pf --socket-mem=1024,1024 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   Exceeds maximum payload limit::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 2.2.2.4 dst is 2.2.2.5 / sctp src is 42 / raw relative is 1 offset is 2 pattern is abcdefghijklmnopq / end actions queue index 5 / end

   it shows "Caught error type 9 (specific pattern item): cause: 0x7fd87ff60160
   exceeds maximum payload limit".

2) can't set mac_addr when setting fdir filter::

    testpmd> flow create 0 ingress pattern eth dst is 00:11:22:33:44:55 / vlan tci is 4095 / ipv6 src is 2001::3 dst is 2001::4 tc is 6 hop is 60 / tcp src is 32 dst is 33 / end actions queue index 3 / end

   it shows "Caught error type 9 (specific pattern item): cause: 0x7f463ff60100
   Invalid MAC_addr mask".

3) can't change the configuration of the same packet type::
    testpmd> flow create 0 ingress pattern eth / vlan tci is 3 / ipv4 src is 192.168.0.1 dst is 192.168.0.2 tos is 4 ttl is 4 / sctp src is 44 dst is 45 tag is 1 / end actions passthru / flag / end
    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 tos is 4 ttl is 4 / sctp src is 34 dst is 35 tag is 1 / end actions passthru / flag / end

   it shows "Caught error type 9 (specific pattern item): cause: 0x7feabff60120
   Conflict with the first rule's input set".

4) invalid queue ID::

    testpmd> flow create 0 ingress pattern eth / ipv6 src is 2001::3 dst is 2001::4 tc is 6 hop is 60 / tcp src is 32 dst is 33 / end actions queue index 16 / end

   it shows "Caught error type 11 (specific action): cause: 0x7ffc7bb9a338,
   Invalid queue ID for FDIR".

   If create a rule on vf that has invalid queue ID::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 proto is 3 / vf id is 0 / end actions queue index 4 / end

   it shows "Caught error type 11 (specific action): cause: 0x7ffc7bb9a338,
   Invalid queue ID for FDIR".


Note:

/// not support IP fragment ///


Test case: Fortville tunnel vxlan
=================================

   Prerequisites:

   add a vf on dpdk pf, then bind the vf to vfio-pci::

    echo 1 >/sys/bus/pci/devices/0000:05:00.0/max_vfs
    ./usertools/dpdk-devbind.py -b vfio-pci 05:02.0

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -w 05:00.0 --file-prefix=pf --socket-mem=1024,1024 -- -i --rxq=16 --txq=16 --txqflags=0x0 --disable-rss
    testpmd> rx_vxlan_port add 4789 0
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> set promisc all off
    testpmd> start
    the pf's mac address is 00:00:00:00:01:00

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e0000 -n 4 -w 05:02.0 --file-prefix=vf --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --txqflags=0x0 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> set promisc all off
    testpmd> start

   the vf's mac address is D2:8C:1A:50:2A:78

2. create filter rules

   inner mac + actions pf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / udp / vxlan / eth dst is 00:11:22:33:44:55 / end actions pf / queue index 1 / end

   vni + inner mac + actions pf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / udp / vxlan vni is 2 / eth dst is 00:11:22:33:44:55 / end actions pf / queue index 2 / end

   inner mac + inner vlan +actions pf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / udp / vxlan / eth dst is 00:11:22:33:44:55 / vlan tci is 10 / end actions pf / queue index 3 / end

   vni + inner mac + inner vlan + actions pf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / udp / vxlan vni is 4 / eth dst is 00:11:22:33:44:55 / vlan tci is 20 / end actions pf / queue index 4 / end

   inner mac + outer mac + vni + actions pf::

    testpmd> flow create 0 ingress pattern eth dst is 00:11:22:33:44:66 / ipv4 / udp / vxlan vni is 5 /  eth dst is 00:11:22:33:44:55 / end actions pf / queue index 5 / end

   vni + inner mac + inner vlan + actions vf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / udp / vxlan vni is 6 / eth dst is 00:11:22:33:44:55 / vlan tci is 30 / end actions vf id 0 / queue index 1 / end

   inner mac + outer mac + vni + actions vf::

    testpmd> flow create 0 ingress pattern eth dst is 00:11:22:33:44:66 / ipv4 / udp / vxlan vni is 7 /  eth dst is 00:11:22:33:44:55 / end actions vf id 0 / queue index 3 / end

3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan()/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan(vni=2)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt31 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan()/Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=10)/IP()/TCP()/Raw('x' * 20)
    pkt32 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan()/Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=11)/IP()/TCP()/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan(vni=4)/Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=20)/IP()/TCP()/Raw('x' * 20)
    pkt51 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan(vni=5)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt52 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan(vni=4)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt53 = Ether(dst="00:00:00:00:01:00")/IP()/UDP()/Vxlan(vni=5)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt54 = Ether(dst="00:11:22:33:44:77")/IP()/UDP()/Vxlan(vni=5)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt55 = Ether(dst="00:00:00:00:01:00")/IP()/UDP()/Vxlan(vni=5)/Ether(dst="00:11:22:33:44:77")/IP()/TCP()/Raw('x' * 20)
    pkt56 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan(vni=5)/Ether(dst="00:11:22:33:44:77")/IP()/TCP()/Raw('x' * 20)
    pkt61 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan(vni=6)/Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=30)/IP()/TCP()/Raw('x' * 20)
    pkt62 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan(vni=6)/Ether(dst="00:11:22:33:44:77")/Dot1Q(vlan=30)/IP()/TCP()/Raw('x' * 20)
    pkt63 = Ether(dst="D2:8C:1A:50:2A:78")/IP()/UDP()/Vxlan(vni=6)/Ether(dst="00:11:22:33:44:77")/Dot1Q(vlan=30)/IP()/TCP()/Raw('x' * 20)
    pkt64 = Ether(dst="00:00:00:00:01:00")/IP()/UDP()/Vxlan(vni=6)/Ether(dst="00:11:22:33:44:77")/Dot1Q(vlan=30)/IP()/TCP()/Raw('x' * 20)
    pkt71 = Ether(dst="00:11:22:33:44:66")/IP()/UDP()/Vxlan(vni=7)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt72 = Ether(dst="D2:8C:1A:50:2A:78")/IP()/UDP()/Vxlan(vni=7)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt73 = Ether(dst="D2:8C:1A:50:2A:78")/IP()/UDP()/Vxlan(vni=7)/Ether(dst="00:11:22:33:44:77")/IP()/TCP()/Raw('x' * 20)
    pkt74 = Ether(dst="00:00:00:00:01:00")/IP()/UDP()/Vxlan(vni=7)/Ether(dst="00:11:22:33:44:77")/IP()/TCP()/Raw('x' * 20)

   verify pkt1 received by pf queue 1, pkt2 to pf queue 2,
   pkt31 to pf queue 3, pkt32 to pf queue 1, pkt4 to pf queue 4,
   pkt51 to pf queue 5, pkt52 to pf queue 1, pkt53 to pf queue 1,
   pkt54 to pf queue 1, pkt55 to pf queue 0, pf can't receive pkt56.
   pkt61 to vf queue 1 and pf queue 1, pf and vf can't receive pkt62,
   pkt63 to vf queue 0, pkt64 to pf queue 0, vf can't receive pkt64,
   pkt71 to vf queue 3 and pf queue 1, pkt72 to pf queue 1, vf can't receive
   pkt72, pkt73 to vf queue 0, pkt74 to pf queue 0, vf can't receive pkt74.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0

   verify pkt51 to pf queue 5, pkt53 and pkt55 to pf queue 0,
   pf can't receive pkt52,pkt54 and pkt56. pkt71 to vf queue 3,
   pkt72 and pkt73 to vf queue 0, pkt74 to pf queue 0, vf can't receive pkt74.
   Then::

    testpmd> flow flush 0
    testpmd> flow list 0


Test case: Fortville tunnel nvgre
=================================

   Prerequisites:

   add two vfs on dpdk pf, then bind the vfs to vfio-pci::

    echo 2 >/sys/bus/pci/devices/0000:05:00.0/max_vfs
    ./usertools/dpdk-devbind.py -b vfio-pci 05:02.0 05:02.1

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -w 05:00.0 --file-prefix=pf --socket-mem=1024,1024 -- -i --rxq=16 --txq=16 --txqflags=0x0
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> set promisc all off
    testpmd> start

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e0000 -n 4 -w 05:02.0 --file-prefix=vf0 --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --txqflags=0x0
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> set promisc all off
    testpmd> start

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e00000 -n 4 -w 05:02.1 --file-prefix=vf1 --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --txqflags=0x0
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> set promisc all off
    testpmd> start

   the pf's mac address is 00:00:00:00:01:00
   the vf0's mac address is 54:52:00:00:00:01
   the vf1's mac address is 54:52:00:00:00:02

2. create filter rules

   inner mac + actions pf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / nvgre / eth dst is 00:11:22:33:44:55 / end actions pf / queue index 1 / end

   tni + inner mac + actions pf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / nvgre tni is 2 / eth dst is 00:11:22:33:44:55 / end actions pf / queue index 2 / end

   inner mac + inner vlan + actions pf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / nvgre / eth dst is 00:11:22:33:44:55 / vlan tci is 30 / end actions pf / queue index 3 / end

   tni + inner mac + inner vlan + actions pf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / nvgre tni is 0x112244 / eth dst is 00:11:22:33:44:55 / vlan tci is 40 / end actions pf / queue index 4 / end

   inner mac + outer mac + tni + actions pf::

    testpmd> flow create 0 ingress pattern eth dst is 00:11:22:33:44:66 / ipv4 / nvgre tni is 0x112255 /  eth dst is 00:11:22:33:44:55 / end actions pf / queue index 5 / end

   tni + inner mac + inner vlan + actions vf::

    testpmd> flow create 0 ingress pattern eth / ipv4 / nvgre tni is 0x112266 / eth dst is 00:11:22:33:44:55 / vlan tci is 60 / end actions vf id 0 / queue index 1 / end

   inner mac + outer mac + tni + actions vf::

    testpmd> flow create 0 ingress pattern eth dst is 00:11:22:33:44:66 / ipv4 / nvgre tni is 0x112277 /  eth dst is 00:11:22:33:44:55 / end actions vf id 1 / queue index 3 / end

3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE()/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE(TNI=2)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt31 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE()/Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=30)/IP()/TCP()/Raw('x' * 20)
    pkt32 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE()/Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=31)/IP()/TCP()/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE(TNI=0x112244)/Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=40)/IP()/TCP()/Raw('x' * 20)
    pkt51 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE(TNI=0x112255)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt52 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE(TNI=0x112256)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt53 = Ether(dst="00:00:00:00:01:00")/IP()/NVGRE(TNI=0x112255)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt54 = Ether(dst="00:11:22:33:44:77")/IP()/NVGRE(TNI=0x112255)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt55 = Ether(dst="00:00:00:00:01:00")/IP()/NVGRE(TNI=0x112255)/Ether(dst="00:11:22:33:44:77")/IP()/TCP()/Raw('x' * 20)
    pkt56 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE(TNI=0x112255)/Ether(dst="00:11:22:33:44:77")/IP()/TCP()/Raw('x' * 20)
    pkt61 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE(TNI=0x112266)/Ether(dst="00:11:22:33:44:55")/Dot1Q(vlan=60)/IP()/TCP()/Raw('x' * 20)
    pkt62 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE(TNI=0x112266)/Ether(dst="00:11:22:33:44:77")/Dot1Q(vlan=60)/IP()/TCP()/Raw('x' * 20)
    pkt63 = Ether(dst="54:52:00:00:00:01")/IP()/NVGRE(TNI=0x112266)/Ether(dst="00:11:22:33:44:77")/Dot1Q(vlan=60)/IP()/TCP()/Raw('x' * 20)
    pkt64 = Ether(dst="00:00:00:00:01:00")/IP()/NVGRE(TNI=0x112266)/Ether(dst="00:11:22:33:44:77")/Dot1Q(vlan=60)/IP()/TCP()/Raw('x' * 20)
    pkt71 = Ether(dst="00:11:22:33:44:66")/IP()/NVGRE(TNI=0x112277)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt72 = Ether(dst="54:52:00:00:00:02")/IP()/NVGRE(TNI=0x112277)/Ether(dst="00:11:22:33:44:55")/IP()/TCP()/Raw('x' * 20)
    pkt73 = Ether(dst="54:52:00:00:00:02")/IP()/NVGRE(TNI=0x112277)/Ether(dst="00:11:22:33:44:77")/IP()/TCP()/Raw('x' * 20)
    pkt74 = Ether(dst="00:00:00:00:01:00")/IP()/NVGRE(TNI=0x112277)/Ether(dst="00:11:22:33:44:77")/IP()/TCP()/Raw('x' * 20)

   verify pkt1 received by pf queue 1, pkt2 to pf queue 2,
   pkt31 to pf queue 3, pkt32 to pf queue 1, pkt4 to pf queue 4,
   pkt51 to pf queue 5, pkt52 to pf queue 1, pkt53 to pf queue 1,
   pkt54 to pf queue 1, pkt55 to pf queue 0, pf can't receive pkt56.
   pkt61 to vf0 queue 1 and pf queue 1, pf and vf0 can't receive pkt62,
   pkt63 to vf0 queue 0, pkt64 to pf queue 0, vf0 can't receive pkt64,
   pkt71 to vf1 queue 3 and pf queue 1, pkt72 to pf queue 1, vf1 can't receive
   pkt72, pkt73 to vf1 queue 0, pkt74 to pf queue 0, vf1 can't receive pkt74.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0

   verify pkt51 to pf queue 5, pkt53 and pkt55 to pf queue 0,
   pf can't receive pkt52,pkt54 and pkt56. pkt71 to vf1 queue 3,
   pkt72 and pkt73 to vf1 queue 0, pkt74 to pf queue 0, vf1 can't receive pkt74.
   Then::
    
    testpmd> flow flush 0
    testpmd> flow list 0


Test case: IXGBE SYN
====================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   ipv4::

    testpmd> flow create 0 ingress pattern eth / ipv4 / tcp flags spec 0x02 flags mask 0x02 / end actions queue index 3 / end

   ipv6::

    testpmd> flow destroy 0 rule 0
    testpmd> flow create 0 ingress pattern eth / ipv6 / tcp flags spec 0x02 flags mask 0x02 / end actions queue index 4 / end

   send packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(dport=80,flags="S")/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(dport=80,flags="PA")/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/TCP(dport=80,flags="S")/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/TCP(dport=80,flags="PA")/Raw('x' * 20)

   ipv4 verify pkt1 to queue 3, pkt2 to queue 0, pkt3 to queue 3, pkt4 to queue 0
   ipv6 verify pkt1 to queue 4, pkt2 to queue 0, pkt3 to queue 4, pkt4 to queue 0
   notes: the out packet default is Flags [S], so if the flags is omitted in sent
   pkt, the pkt will be into queue 3 or queue 4.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0


Test case: IXGBE n-tuple(supported by x540 and 82599)
=====================================================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   ipv4-other::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / end actions queue index 1 / end

   ipv4-udp::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.2 dst is 192.168.0.3 proto is 17 / udp src is 22 dst is 23 / end actions queue index 2 / end

   ipv4-tcp::

    testpmd> flow create 0 ingress pattern ipv4 src is 192.168.0.2 dst is 192.168.0.3 proto is 6 / tcp src is 32 dst is 33 / end actions queue index 3 / end

   ipv4-sctp::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.2 dst is 192.168.0.3 proto is 132 / sctp src is 44 dst is 45 / end actions queue index 4 / end

3. send packets::

    pkt11 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/Raw('x' * 20)
    pkt12 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.2", dst="192.168.0.3")/Raw('x' * 20)
    pkt21 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.2", dst="192.168.0.3")/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt22 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.2", dst="192.168.0.3")/UDP(sport=22,dport=24)/Raw('x' * 20)
    pkt31 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.2", dst="192.168.0.3")/TCP(sport=32,dport=33)/Raw('x' * 20)
    pkt32 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.2", dst="192.168.0.3")/TCP(sport=34,dport=33)/Raw('x' * 20)
    pkt41 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.2", dst="192.168.0.3")/SCTP(sport=44,dport=45)/Raw('x' * 20)
    pkt42 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.2", dst="192.168.0.3")/SCTP(sport=44,dport=46)/Raw('x' * 20)
    pkt5 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/SCTP(sport=44,dport=45)/Raw('x' * 20)
    pkt6 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/TCP(sport=32,dport=33)/Raw('x' * 20)

   verify pkt11 to queue 1, pkt12 to queue 0,
   pkt21 to queue 2, pkt22 to queue 0,
   pkt31 to queue 3, pkt32 to queue 0,
   pkt41 to queue 4, pkt42 to queue 0,
   pkt5 to queue 1, pkt6 to queue 0,

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0


Test case: IXGBE ethertype
==========================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules::

    testpmd> flow validate 0 ingress pattern eth type is 0x0806 / end actions queue index 3 / end
    testpmd> flow validate 0 ingress pattern eth type is 0x86DD / end actions queue index 5 / end
    testpmd> flow create 0 ingress pattern eth type is 0x0806 / end actions queue index 3 / end
    testpmd> flow create 0 ingress pattern eth type is 0x88cc / end actions queue index 4 / end

   the ixgbe don't support the 0x88DD eth type packet. so the second command
   failed.

3. send packets::

    pkt1 = Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(pdst="192.168.1.1")
    pkt2 = Ether(dst="00:11:22:33:44:55", type=0x88CC)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55", type=0x86DD)/Raw('x' * 20)

   verify pkt1 to queue 3, and pkt2 to queue 4, pkt3 to queue 0.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0

   verify pkt1 to queue 0, and pkt2 to queue 4.
   Then::

    testpmd> flow list 0
    testpmd> flow flush 0

   verify pkt1 to queue 0, and pkt2 to queue 0.
   Then::

    testpmd> flow list 0


Test case: IXGBE L2-tunnel(supported by x552 and x550)
======================================================

   Prerequisites:

   add two vfs on dpdk pf, then bind the vfs to vfio-pci::

    echo 2 >/sys/bus/pci/devices/0000:05:00.0/max_vfs
    ./usertools/dpdk-devbind.py -b vfio-pci 05:02.0 05:02.1

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -w 05:00.0 --file-prefix=pf --socket-mem=1024,1024 -- -i --rxq=16 --txq=16 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e0000 -n 4 -w 05:02.0 --file-prefix=vf0 --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1e00000 -n 4 -w 05:02.1 --file-prefix=vf1 --socket-mem=1024,1024 -- -i --rxq=4 --txq=4 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

   Enabling ability of parsing E-tag packet, set on pf::

    testpmd> port config 0 l2-tunnel E-tag enable

   Enable E-tag packet forwarding, set on pf::

    testpmd> E-tag set forwarding on port 0

2. create filter rules::

    testpmd> flow create 0 ingress pattern e_tag grp_ecid_b is 0x1309 / end actions queue index 0 / end
    testpmd> flow create 0 ingress pattern e_tag grp_ecid_b is 0x1308 / end actions queue index 1 / end
    testpmd> flow create 0 ingress pattern e_tag grp_ecid_b is 0x1307 / end actions queue index 2 / end

3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/Dot1BR(GRP=0x1, ECIDbase=0x309)/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55")/Dot1BR(GRP=0x1, ECIDbase=0x308)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55")/Dot1BR(GRP=0x1, ECIDbase=0x307)/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:55")/Dot1BR(GRP=0x2, ECIDbase=0x309)/Raw('x' * 20)

   verify pkt1 to vf0 queue0, pkt2 to vf1 queue0, pkt3 to pf queue0,
   pkt4 can't received by pf and vfs.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0


Test case: IXGBE fdir for ipv4
==============================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=perfect
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   ipv4-other
   (only support by 82599 and x540, this rule matches the n-tuple)::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / end actions queue index 1 / end

   ipv4-udp::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.3 dst is 192.168.0.4 / udp src is 22 dst is 23 / end actions queue index 2 / end

   ipv4-tcp::

    testpmd> flow create 0 ingress pattern ipv4 src is 192.168.0.3 dst is 192.168.0.4 / tcp src is 32 dst is 33 / end actions queue index 3 / end

   ipv4-sctp
   (x550/x552, 82599 can support this format, because it matches n-tuple)::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.3 dst is 192.168.0.4 / sctp src is 44 dst is 45 / end actions queue index 4 / end

   ipv4-sctp(82599/x540)::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.3 dst is 192.168.0.4 / sctp / end actions queue index 4 / end

   ipv4-sctp-drop(x550/x552)::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.3 dst is 192.168.0.4 / sctp src is 46 dst is 47 / end actions drop / end

   ipv4-sctp-drop(82599/x540)::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.5 dst is 192.168.0.6 / sctp / end actions drop / end

notes: 82599 don't support the sctp port match drop, x550 and x552 support it.

   ipv4-udp-flexbytes::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / udp src is 24 dst is 25 / raw relative is 0 search is 0 offset is 44 limit is 0 pattern is 86 / end actions queue index 5 / end

   ipv4-tcp-flexbytes::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.3 dst is 192.168.0.4 / tcp src is 22 dst is 23 / raw relative spec 0 relative mask 1 search spec 0 search mask 1 offset spec 54 offset mask 0xffffffff limit spec 0 limit mask 0xffff pattern is ab pattern is cd / end actions queue index 6 / end

notes: the second pattern will overlap the first pattern.
the rule 6 and 7 should be created after the testpmd reset,
because the flexbytes rule is global bit masks.

   invalid queue id::
 
    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / udp src is 32 dst is 33 / end actions queue index 16 / end

notes: the rule can't be created successfully because the queue id
exceeds the max queue id.

3. send packets::

    pkt1 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.1", dst="192.168.0.2")/Raw('x' * 20)
    pkt2 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.4")/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt3 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.4")/TCP(sport=32,dport=33)/Raw('x' * 20)

   for x552/x550::

    pkt41 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.4")/SCTP(sport=44,dport=45)/Raw('x' * 20)
    pkt42 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.4")/SCTP(sport=42,dport=43)/Raw('x' * 20)

   for 82599/x540::

    pkt41 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.4")/SCTP()/Raw('x' * 20)
    pkt42 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.5")/SCTP()/Raw('x' * 20)

   for x552/x550::

    pkt5 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.4")/SCTP(sport=46,dport=47)/Raw('x' * 20)

   for 82599/x540::

    pkt5 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.5", dst="192.168.0.6")/SCTP()/Raw('x' * 20)
    pkt6 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.1", dst="192.168.0.2")/UDP(sport=24,dport=25)/Raw(load="xx86ddef")
    pkt7 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.4")/TCP(sport=22,dport=23)/Raw(load="abcdxxx")
    pkt8 = Ether(dst="A0:36:9F:7B:C5:A9")/IP(src="192.168.0.3", dst="192.168.0.4")/TCP(sport=22,dport=23)/Raw(load="cdcdxxx")

   verify pkt1 to pkt3 can be received by queue 1 to queue 3 correctly.
   pkt41 to queue 4, pkt42 to queue 0, pkt5 couldn't be received.
   pkt6 to queue 5, pkt7 to queue 0, pkt8 to queue 6.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0

Test case: IXGBE fdir for signature(ipv4/ipv6)
==============================================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=signature
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   ipv6-other
   (82599 support this rule,x552 and x550 don't support this rule)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 1 / ipv6 src is 2001::1 dst is 2001::2 / end actions queue index 1 / end

   ipv6-udp::

    testpmd> flow create 0 ingress pattern fuzzy thresh spec 2 thresh last 5 thresh mask 0xffffffff / ipv6 src is 2001::1 dst is 2001::2 / udp src is 22 dst is 23 / end actions queue index 2 / end

   ipv6-tcp::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 3 / ipv6 src is 2001::1 dst is 2001::2 / tcp src is 32 dst is 33 / end actions queue index 3 / end

   ipv6-sctp
   (x552 and x550)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 4 / ipv6 src is 2001::1 dst is 2001::2 / sctp src is 44 dst is 45 / end actions queue index 4 / end

   (82599 and x540)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 4 / ipv6 src is 2001::1 dst is 2001::2 / sctp / end actions queue index 4 / end

   ipv6-other-flexbytes
   (just for 82599/x540)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 6 / ipv6 src is 2001::1 dst is 2001::2 / raw relative is 0 search is 0 offset is 56 limit is 0 pattern is 86 / end actions queue index 5 / end

notes: this rule can be created successfully on 82599/x540, but can't be
created successfully on x552/x550, because it's an ipv4-other rule.
but the offset<=62, the mac header is 14bytes, the ipv6 header is 40 bytes,
the shortest L4 header (udp header) is 8bytes, the total header is 62 bytes,
there is no payload can be set offset. so we don't test the ipv6 flexbytes
on x550/x552.
according to hardware limitation, signature mode does not support drop action,
while IPv6 rely on signature mode, so it is expected result that a IPv6 flow
with drop action can't be created

   ipv4-other
   (82599 support this rule,x552 and x550 don't support this rule)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 1 / eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / end actions queue index 6 / end

   ipv4-udp::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 2 / eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / udp src is 22 dst is 23 / end actions queue index 7 / end

   ipv4-tcp::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 3 / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / tcp src is 32 dst is 33 / end actions queue index 8 / end

   ipv4-sctp(x550/x552)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 4 / eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / sctp src is 44 dst is 45 / end actions queue index 9 / end

   ipv4-sctp(82599/x540)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 5 / eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / sctp / end actions queue index 9 / end

notes: if set the ipv4-sctp rule with sctp ports on 82599, it will fail
to create the rule.

   ipv4-sctp-flexbytes(x550/x552)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 6 / eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / sctp src is 24 dst is 25 / raw relative is 0 search is 0 offset is 48 limit is 0 pattern is ab / end actions queue index 10 / end

   ipv4-sctp-flexbytes(82599/x540)::

    testpmd> flow create 0 ingress pattern fuzzy thresh is 6 / eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 / sctp / raw relative is 0 search is 0 offset is 48 limit is 0 pattern is ab / end actions queue index 10 / end

notes: you need to reset testpmd before create this rule,
because it's conflict with the rule 9.

3. send packets

   ipv6 packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/TCP(sport=32,dport=33)/Raw(load="xxxxabcd")

   for x552/x550::

    pkt4 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2",nh=132)/SCTP(sport=44,dport=45,tag=1)/SCTPChunkData(data="cdxxxx")
    pkt5 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2",nh=132)/SCTP(sport=46,dport=47,tag=1)/SCTPChunkData(data="cdxxxx")

   for 82599/x540::

    pkt41 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2",nh=132)/SCTP(sport=44,dport=45,tag=1)/SCTPChunkData(data="cdxxxx")
    pkt42 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2",nh=132)/SCTP()/SCTPChunkData(data="cdxxxx")
    pkt51 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2",nh=132)/SCTP(sport=46,dport=47,tag=1)/SCTPChunkData(data="cdxxxx")
    pkt52 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::3", dst="2001::4",nh=132)/SCTP(sport=46,dport=47,tag=1)/SCTPChunkData(data="cdxxxx") 
    pkt6 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/Raw(load="xx86abcd")
    pkt7 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/Raw(load="xxx86abcd")

   ipv4 packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=32,dport=33)/Raw('x' * 20)

   for x552/x550::

    pkt41 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/SCTP(sport=44,dport=45)/Raw('x' * 20)
    pkt42 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/SCTP(sport=42,dport=43)/Raw('x' * 20)

   for 82599/x540::

    pkt41 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/SCTP()/Raw('x' * 20)
    pkt42 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.3")/SCTP()/Raw('x' * 20)
    pkt51 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/SCTP(sport=24,dport=25)/Raw(load="xxabcdef")
    pkt52 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/SCTP(sport=24,dport=25)/Raw(load="xxaccdef")

   verify ipv6 packets:
   for x552/x550:
   pkt1 to queue 0, pkt2 to queue 2, pkt3 to queue 3.
   pkt4 to queue 4, pkt5 to queue 0.

   for 82599/x540:
   packet pkt1 to pkt3 can be received by queue 1 to queue 3 correctly.
   pkt41 and pkt42 to queue 4, pkt51 to queue 4, pkt52 to queue 0. 
   pkt6 to queue 5, pkt7 to queue 0.

   verify ipv4 packets:
   for x552/x550:
   pk1 to queue 0, pkt2 to queue 7, pkt3 to queue 8.
   pkt41 to queue 9, pkt42 to queue 0,
   pkt51 to queue 10, pkt52 to queue 0.

   for 82599/x540:
   pkt1 to pkt3 can be received by queue 6 to queue 8 correctly.
   pkt41 to queue 9, pkt42 to queue 0,
   pkt51 to queue 10, pkt52 to queue 0.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0

Test case: IXGBE fdir for mac/vlan(support by x540, x552, x550)
===============================================================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=perfect-mac-vlan
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start
    testpmd> vlan set strip off 0
    testpmd> vlan set filter off 0

2. create filter rules::

    testpmd> flow create 0 ingress pattern eth dst is A0:36:9F:7B:C5:A9 / vlan tpid is 0x8100 tci is 1 / end actions queue index 9 / end
    testpmd> flow create 0 ingress pattern eth dst is A0:36:9F:7B:C5:A9 / vlan tpid is 0x8100 tci is 4095 / end actions queue index 10 / end

3. send packets::

    pkt1 = Ether(dst="A0:36:9F:7B:C5:A9")/Dot1Q(vlan=1)/IP()/TCP()/Raw('x' * 20)
    pkt2 = Ether(dst="A0:36:9F:7B:C5:A9")/Dot1Q(vlan=4095)/IP()/UDP()/Raw('x' * 20)

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0

Test case: IXGBE fdir for tunnel (vxlan and nvgre)(support by x540, x552, x550)
===============================================================================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=16 --txq=16 --disable-rss --pkt-filter-mode=perfect-tunnel
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   vxlan::

    testpmd> flow create 0 ingress pattern eth / ipv4 / udp / vxlan vni is 8 / eth dst is A0:36:9F:7B:C5:A9 / vlan tci is 2 tpid is 0x8100 / end actions queue index 1 / end
    testpmd> flow create 0 ingress pattern eth / ipv6 / udp / vxlan vni is 9 / eth dst is A0:36:9F:7B:C5:A9 / vlan tci is 4095 tpid is 0x8100 / end actions queue index 2 / end

   nvgre::

    testpmd> flow create 0 ingress pattern eth / ipv4 / nvgre tni is 0x112244 / eth dst is A0:36:9F:7B:C5:A9 / vlan tci is 20 / end actions queue index 3 / end
    testpmd> flow create 0 ingress pattern eth / ipv6 / nvgre tni is 0x112233 / eth dst is A0:36:9F:7B:C5:A9 / vlan tci is 21 / end actions queue index 4 / end

3. send packets

   vxlan::

    pkt1=Ether(dst="A0:36:9F:7B:C5:A9")/IP()/UDP()/Vxlan(vni=8)/Ether(dst="A0:36:9F:7B:C5:A9")/Dot1Q(vlan=2)/IP()/TCP()/Raw('x' * 20)
    pkt2=Ether(dst="A0:36:9F:7B:C5:A9")/IPv6()/UDP()/Vxlan(vni=9)/Ether(dst="A0:36:9F:7B:C5:A9")/Dot1Q(vlan=4095)/IP()/TCP()/Raw('x' * 20)

   nvgre::

    pkt3 = Ether(dst="A0:36:9F:7B:C5:A9")/IP()/NVGRE(TNI=0x112244)/Ether(dst="A0:36:9F:7B:C5:A9")/Dot1Q(vlan=20)/IP()/TCP()/Raw('x' * 20)
    pkt4 = Ether(dst="A0:36:9F:7B:C5:A9")/IPv6()/NVGRE(TNI=0x112233)/Ether(dst="A0:36:9F:7B:C5:A9")/Dot1Q(vlan=21)/IP()/TCP()/Raw('x' * 20)

   verify pkt1 to pkt4 are into queue 1 to queue 4.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0

Test case: igb SYN
==================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=8 --txq=8 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   ipv4::

    testpmd> flow create 0 ingress pattern eth / ipv4 / tcp flags spec 0x02 flags mask 0x02 / end actions queue index 3 / end

   ipv6::

    testpmd> flow destroy 0 rule 0
    testpmd> flow create 0 ingress pattern eth / ipv6 / tcp flags spec 0x02 flags mask 0x02 / end actions queue index 4 / end

3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(dport=80,flags="S")/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/TCP(dport=80,flags="S")/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(dport=80,flags="PA")/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/TCP(dport=80,flags="PA")/Raw('x' * 20)

   ipv4 verify pkt1 to queue 3, pkt2 to queue 0, pkt3 to queue 0
   ipv6 verify pkt2 to queue 4, pkt1 to queue 0, pkt4 to queue 0

notes: the out packet default is Flags [S], so if the flags is omitted in
sent pkt, the pkt will be into queue 3 or queue 4.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0

Test case: igb n-tuple(82576)
=============================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=8 --txq=8 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules::

    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 proto is 17 / udp src is 22 dst is 23 / end actions queue index 1 / end
    testpmd> flow create 0 ingress pattern eth / ipv4 src is 192.168.0.1 dst is 192.168.0.2 proto is 6 / tcp src is 22 dst is 23 / end actions queue index 2 / end

3. send packets::

    pkt1 = Ether(dst="%s")/IP(src="192.168.0.1", dst="192.168.0.2")/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt2 = Ether(dst="%s")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=32,dport=33)/Raw('x' * 20)

   verify pkt1 to queue 1, pkt2 to queue 2, pkt3 to queue 3.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0

Test case: igb n-tuple(i350 or 82580)
=====================================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=8 --txq=8 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules::

    testpmd> flow create 0 ingress pattern eth / ipv4 proto is 17 / udp dst is 23 / end actions queue index 1 / end
    testpmd> flow create 0 ingress pattern eth / ipv4 proto is 6 / tcp dst is 33 / end actions queue index 2 / end

3. send packets::

    pkt1 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/UDP(sport=22,dport=23)/Raw('x' * 20)
    pkt2 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/UDP(sport=22,dport=24)/Raw('x' * 20)
    pkt3 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=32,dport=33)/Raw('x' * 20)
    pkt4 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=32,dport=34)/Raw('x' * 20)

   verify pkt1 to queue 1, pkt2 to queue 0.
   pkt3 to queue 2, pkt4 to queue 0.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0

Test case: igb ethertype
========================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=8 --txq=8
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules::

    testpmd> flow validate 0 ingress pattern eth type is 0x0806 / end actions queue index 3 / end
    testpmd> flow validate 0 ingress pattern eth type is 0x86DD / end actions queue index 5 / end
    testpmd> flow create 0 ingress pattern eth type is 0x0806 / end actions queue index 3 / end
    testpmd> flow create 0 ingress pattern eth type is 0x88cc / end actions queue index 4 / end
    testpmd> flow create 0 ingress pattern eth type is 0x88cc / end actions queue index 8 / end

   the ixgbe don't support the 0x88DD eth type packet. so the second command
   failed. the queue id exceeds the max queue id, so the last command failed.

3. send packets::

    pkt1 = Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(pdst="192.168.1.1")
    pkt2 = Ether(dst="00:11:22:33:44:55", type=0x88CC)/Raw('x' * 20)

   verify pkt1 to queue 3, and pkt2 to queue 4.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    verify pkt1 to queue 0, and pkt2 to queue 4.
    testpmd> flow list 0
    testpmd> flow flush 0

   verify pkt1 to queue 0, and pkt2 to queue 0
   Then::

    testpmd> flow list 0

Test case: igb flexbytes
========================

1. Launch the app ``testpmd`` with the following arguments::

    ./x86_64-native-linuxapp-gcc/app/testpmd -c 1ffff -n 4 -- -i --rxq=8 --txq=8 --disable-rss
    testpmd> set fwd rxonly
    testpmd> set verbose 1
    testpmd> start

2. create filter rules

   l2 packet::

    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 14 pattern is fhds / end actions queue index 1 / end

   l2 packet relative is 1
   (the first relative must be 0, so this rule won't work)::

    testpmd> flow create 0 ingress pattern raw relative is 1 offset is 2 pattern is fhds / end actions queue index 2 / end

   ipv4 packet::
 
    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 34 pattern is ab / end actions queue index 3 / end

   ipv6 packet::

    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 58 pattern is efgh / end actions queue index 4 / end

   3 fields relative is 0::

    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 38 pattern is ab / raw relative is 0 offset is 34 pattern is cd / raw relative is 0 offset is 42 pattern is efgh / end actions queue index 5 / end

   4 fields relative is 0 and 1::

    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 48 pattern is ab / raw relative is 1 offset is 0 pattern is cd / raw relative is 0 offset is 44 pattern is efgh / raw relative is 1 offset is 10 pattern is hijklmnopq / end actions queue index 6 / end

   3 fields offset conflict::

    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 64 pattern is ab / raw relative is 1 offset is 4 pattern is cdefgh / raw relative is 0 offset is 68 pattern is klmn / end actions queue index 7 / end

   1 field 128bytes
   
   flush the rules::

    testpmd> flow flush 0

   then create the rule::

    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 128 pattern is ab / end actions queue index 1 / end
    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 126 pattern is abcd / end actions queue index 1 / end
    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 126 pattern is ab / end actions queue index 1 / end

   the first two rules failed to create, only the last flow rule is created successfully.

   2 field 128bytes::

    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 68 pattern is ab / raw relative is 1 offset is 58 pattern is cd / end actions queue index 2 / end
    testpmd> flow create 0 ingress pattern raw relative is 0 offset is 68 pattern is ab / raw relative is 1 offset is 56 pattern is cd / end actions queue index 2 / end

   the first rule failed to create, only the last flow rule is created successfully.

3. send packets::

    pkt11 = Ether(dst="00:11:22:33:44:55")/Raw(load="fhdsab")
    pkt12 = Ether(dst="00:11:22:33:44:55")/Raw(load="afhdsb")
    pkt2 = Ether(dst="00:11:22:33:44:55")/Raw(load="abfhds")
    pkt3 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/Raw(load="abcdef")
    pkt41 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/Raw(load="xxxxefgh")
    pkt42 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2")/TCP(sport=32,dport=33)/Raw(load="abcdefgh")
    pkt5 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/Raw(load="cdxxabxxefghxxxx")
    pkt6 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2", tos=4, ttl=3)/UDP(sport=32,dport=33)/Raw(load="xxefghabcdxxxxxxhijklmnopqxxxx")
    pkt71 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=22,dport=23)/Raw(load="xxxxxxxxxxabxxklmnefgh")
    pkt72 = Ether(dst="00:11:22:33:44:55")/IPv6(src="2001::1", dst="2001::2", tc=3, hlim=30)/Raw(load="xxxxxxxxxxabxxklmnefgh")
    pkt73 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=22,dport=23)/Raw(load="xxxxxxxxxxabxxklcdefgh")
    pkt81 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=22,dport=23)/Raw(load="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxab")
    pkt82 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=22,dport=23)/Raw(load="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxcb")
    pkt91 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=22,dport=23)/Raw(load="xxxxxxxxxxxxxxabxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxcd")
    pkt92 = Ether(dst="00:11:22:33:44:55")/IP(src="192.168.0.1", dst="192.168.0.2")/TCP(sport=22,dport=23)/Raw(load="xxxxxxxxxxxxxxabxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxce")

   verify pkt11 to queue 1, pkt12 to queue 0.
   pkt2 to queue 0.
   pkt3 to queue 3.
   pkt41 to queue 4, pkt42 to queue 0, // tcp header has 20 bytes.
   pkt5 to queue 5.
   pkt6 to queue 6.
   pkt71 to queue 7, pkt72 to queue 7, pkt73 to queue 0.
   pkt81 to queue 1, pkt82 to queue 0.
   pkt91 to queue 2, pkt92 to queue 0.

4. verify rules can be listed and destroyed::

    testpmd> flow list 0
    testpmd> flow destroy 0 rule 0
    testpmd> flow list 0
    testpmd> flow flush 0
    testpmd> flow list 0
