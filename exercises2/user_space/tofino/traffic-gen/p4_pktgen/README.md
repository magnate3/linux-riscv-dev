# WARN: This repo is not organized yet, and do not ensure anyting.


# Tested SDE

* SDE 9.7.1

# p4_pktgen

Simple Example of p4_pktgen:
1. Timer - One Shot
2. Timer - Periodic
3. Port Down
4. Recirculate Pattern Matching
5. Deparser Triggering - TF2 only
6. PFC - TF2 only

## Naming

Start with "tf2" : It means that the features are only supported by tf2.
Start with "tf" : It means that the features are supported by tf1 and tf2.

## Describtion

One Shot : When timer meet the dealine, pktgne will generate one packet to the specified port.

Periodic : pktgen will generate the packet periodically by timer.

Port Down : When the pktgen detecting the port goes down, then generate packe to the speicified port.

Recirculate Pattern Matching : If pktgen detect that the packet recirculate with specific byte pattern, then pktgen will generate a packet to the speicified port.

Note: pktgen check 32-bit front of the packet for pattern matching.

Deparser Triggr : Using pktgen extern to trigger pktgen. This feature is only triggring pktgen, all of the setting of pktgen defined by control plane.

PFC : When PPG and Queue are meet the threshold, the pktgen will generate the PFC packet.

## Switch Port Setup

for pktgen feature testing.
bfshell -f port_up.txt

for pktgen performance testing.
bfshell -f port_up_perf.txt


## Test Result

https://www.notion.so/p4switchprofile-ecmplag/PKT-Gen-84050d00da3c40129418e16254fc5f2c
