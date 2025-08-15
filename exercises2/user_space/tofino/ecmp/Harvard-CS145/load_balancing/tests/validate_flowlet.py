#!/usr/bin/python3

import os
import sys
import time
import math
import json

# This testing script checks whether the dataplane distributes flowlets into different paths.
# We conducted 4 tests, each sending a single flow, to measure the standard deviation of 
# packet count across four paths at the same level.
# Without flowlet, the distribution should be observed as [1, 2, 8888, 4], etc.
# With flowlet, the distribution should be observed as [2222, 2121, 2122, 2211], etc.

# parse topology
f = open("topology.json")
topo_json = json.load(f)
f.close()
topo = 0
links = topo_json['links']
for l in links:
    if (l['node1'] == 'a1' and l['node2'] == 'c1'):
        if l['port1'] == 1 or l['port1'] == 2:
            topo = 1
    if (l['node1'] == 'c1' and l['node2'] == 'a1'):
        if l['port2'] == 1 or l['port2'] == 2:
            topo = 1

print("Test ECMP for flowlet")

# Run the test for 4 times
for i in range(4):
    print("Test case {}".format(i+1))

    # Running the script to send traffic
    # The script randomly chooses two different hosts in different pods, and run send traffic in a flowlet manner
    # The flowlet gap is around 0.5sec
    # The script also uses TCPDUMP to record the traffic for the 4 links from the two aggregated switches to the four core switches
    # The two aggregated switches could be in either pod for the two hosts
    os.system("sudo bash tests/iperf_flowlet.sh {}".format(topo))

    # Get the number of packets for the 4 links
    files = os.listdir("tcpdump_logs")
    d = []
    for file in files:
        with open("tcpdump_logs/"+ file, "r") as f:
            contents = f.read()
            d.append(len(contents))

    print("# of packets on the four links")
    print(d)
    
    # If the flowlet switching works, the deviation of the 4 numbers should be small
    avg = 0.0
    for item in d:
        avg += item
    avg /= len(d)
    dev = 0.0
    for item in d:
        dev += (item - avg) ** 2
    dev /= len(d)
    dev = math.sqrt(dev)
    dev = dev / avg
    print("stddev of four links")
    print(dev)
    
    if abs(dev) > 0.3:
        print("Test fail")
        exit()

print("Test pass")
