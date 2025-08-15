
## Control Plane
The `control plane` directory includes the control plane implementation of RIFO, which is used to configure the Tofino switch for running experiments. A packet is required to initialize the recirculation of the `rifo_worker` packet for egress queue occupancy.

## DPDK
The `dpdk` directory contains the DPDK code for generating and receiving packets by the sender and receiver. The client and server code must be compiled and executed separately. A Python script with default configurations is provided to replicate the experiments presented in the paper.  

**Note:** Ensure that your servers have sufficient CPU cores to meet the default settings before running the script.

## Source Code
The `src` directory contains the P4_16 implementation of RIFO for Tofino 1 switches, based on SDE-9.11.0.
