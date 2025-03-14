# pfctest


The purpose of this script is to provide an easy way to test priority-based flow control, as defined in the IEEE 802.1Qbb standard.
This allows Network Engineers and other technical staff a way to easily test their PFC settings on their network and the impact
that various PFC quanta values can have.  


# Author
Jeremy Georges 

# Description

pfctest creates pfc packets based on a quanta value per traffic class.
The purpose of this tool is it help Network Engineers and other technical staff to test their PFC implementation on their
network. It is not intended to be used to DoS a network by forcing hosts to pause to eternity. Therefore, use
at your own risk and preferably in a lab environment.


The Ethernet Frame format for PFC packets is the following:

                     -------------------------
     Destination MAC |   01:80:C2:00:00:01   |
                     -------------------------
     Source MAC      |      Station MAC      |
                     -------------------------
     Ethertype       |         0x8808        |
                     -------------------------
     OpCode          |         0x0101        |
                     -------------------------
     Class Enable V  | 0x00 E7...E0          |   - Class-enable vector, 8 bits for each class 
                     -------------------------
     Time Class 0    |       0x0000          |
                     -------------------------
     Time Class 1    |       0x0000          |
                     -------------------------
     ...     
                     -------------------------
     Time Class 7    |       0x0000          |
                     -------------------------


Note: Time in quanta where each quantum represents time it takes to transmit 512 bits at the current network speed. For example, Fast Ethernet
takes 10ns per bit, Gb Ethernet is 1ns and 10Gb is 0.1ns per bit time. So if quanta is set to max of 65535 for a 10Gb link PFC class,
then 0.1(512)*65535 = 3.3ms pause time.


Each block above from Ethertype down is 16bits (2 octets)

Sending a quanta of 0 for a specific class tells a receiver that it can 'unpause' explictly for that class. 

```
header pfc_frame_t {
    fields {
        mac_dst : 48;
        mac_src : 48;
        eth_type : 16;
        opcode : 16;
        class_enable : 8;
        pause_quanta : 128;  // 16 bits per priority (total 8 priorities)
    }
}

action send_pfc_frame() {
    pfc_frame_t pfc;
    pfc.mac_dst = 0x0180C2000001;
    pfc.mac_src = 0x001122334455;
    pfc.eth_type = 0x8808;
    pfc.opcode = 0x0101;
    pfc.class_enable = 0x08; // Enable priority 3
    pfc.pause_quanta = 0x0040; // Pause for 64 quanta

    emit(pfc);
}

```


```
Complete Example: Multi-Priority PFC Frame

from scapy.all import Ether, sendp

# Define PFC Frame
pfc_frame = Ether(
    dst="01:80:C2:00:00:01",
    src="00:11:22:33:44:55",
    type=0x8808
) / b"\x01\x01"  # Opcode: PFC

# Enable priorities 3 and 4 (0x18) and set pause times
class_enable_vector = b"\x18"  # Priorities 3 and 4 enabled
pause_quanta = b"\x00\x40" + b"\x00\x80" + b"\x00\x00" * 6  # Different pause times

# Construct and send the frame
pfc_frame = pfc_frame / class_enable_vector / pause_quanta
sendp(pfc_frame, iface="eth0")
```


# Usage

pfctest.py requires a few arguments. The egress interface must be specified and a PFC class. The Quanta value can be 
from 0 - 65535. 

Additionally, an iteration value can be specified which is really the number of packets the script will send out. The default 
is only one packet.


     Usage: pfctest.py [options] arg1 arg2
     
     Options:
       -h, --help            show this help message and exit
       -V, --version         The version
       -d Interface, --device=Interface
                             The Interface to egress packets
       --p0                  Priority Flow Control Enable Class 0
       --p1                  Priority Flow Control Enable Class 1
       --p2                  Priority Flow Control Enable Class 2
       --p3                  Priority Flow Control Enable Class 3
       --p4                  Priority Flow Control Enable Class 4
       --p5                  Priority Flow Control Enable Class 5
       --p6                  Priority Flow Control Enable Class 6
       --p7                  Priority Flow Control Enable Class 7
       --q0=Quanta           Time in Quanta for Class 0
       --q1=Quanta           Time in Quanta for Class 1
       --q2=Quanta           Time in Quanta for Class 2
       --q3=Quanta           Time in Quanta for Class 3
       --q4=Quanta           Time in Quanta for Class 4
       --q5=Quanta           Time in Quanta for Class 5
       --q6=Quanta           Time in Quanta for Class 6
       --q7=Quanta           Time in Quanta for Class 7
       -i number, --iteration=number
                             Number of times to iterate




Additionally, please note that this script only supports Python 2.6/2.7 and Linux.

```
python3 pfctest.py -d enp23s0f1   --p0  --q0=3  -i 2
72
Generating 2 Packet(s)
```

#  byteosaurus_hex.py 
```
python3 byteosaurus_hex.py 

==================================================
Scapy based packet generator
==================================================

1 -- ICMP
2 -- ARP
3 -- IGMP
4 -- Multicast
5 -- VXLAN
6 -- Pause Frame
7 -- Priority Flow Control
8 -- MPLS
9 -- Load PCAP File
10 -- Exit

Enter your choice (1-10): 7
Enter the number of flows > 2

Building flow number [ 1 ]:

Random 802.1Qbb PFC Frame? (y/n) > y
Count (c for continous) > 3
Source Interface > enp23s0f1
2025-03-06 07:31:32,640: INFO: 802.1Qbb PFC Frame built
###[ Ethernet ]### 
  dst       = 01:80:c2:00:00:01
  src       = 2b:d1:22:5e:b3:99
  type      = 0x8808
###[ Raw ]### 
     load      = "\x01\x01\x00\x8fw\x01\x8e\xf8\xd9n\x03m\x00\x00\x00\x00\x00\x00\xde'"
###[ Padding ]### 
        load      = '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'


Building flow number [ 2 ]:

Random 802.1Qbb PFC Frame? (y/n) > y
Count (c for continous) > 2
Source Interface > enp23s0f1
2025-03-06 07:31:54,047: INFO: 802.1Qbb PFC Frame built
###[ Ethernet ]### 
  dst       = 01:80:c2:00:00:01
  src       = c9:e2:3b:10:44:cb
  type      = 0x8808
###[ Raw ]### 
     load      = '\x01\x01\x00}\xce\xd1\x00\x00\x89\xb3M)\x9b\xdb\xd7%\xd7\x92\x00\x00'
###[ Padding ]### 
        load      = '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

2025-03-06 07:31:54,047: INFO: Sending out all flows
2025-03-06 07:31:54,116: INFO: Done sending all flows
2025-03-06 07:31:54,138: INFO: Module completed

```
