# SingeQueue_WFQ

## 1. Introduction:

This is the repo of "TCP-friendly Packet Scheduling for Approximate Weighted Fair Queueing with a Single Queue".

## 2. Content of this repo:

- Simulation_code/:  Packet level simulation code for SQ and SQ-T, which is built on top of [NetBench](https://github.com/ndal-eth/netbench).

- Testbed_code/:
  - SQ_Switch/:  P4 code and controller code for barefoot-tofino switch of SQ algorithm.
  - SQ-T_Switch/:   P4 code and controller code for barefoot-tofino switch of SQ-T algorithm.
  - UDPHosts/:  DPDK host code for the udp experiments.
  - TCPHosts/:   Script for tcp experiments.

## 3. Environment:
- Testbed:

  - Tofino Switch with SDK >= 9.7.2	

  - Hosts with DPDK version >= 17.11.10

  - Python 3.5 or higher

  - iperf3
- Simulation:
  - check it on [here](https://github.com/ndal-eth/netbench).
