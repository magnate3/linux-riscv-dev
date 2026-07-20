#!/usr/bin/python3

import os
import sys
import time
import math

print("Test ECMP")
print("Running iperf")

os.system("sudo bash tests/iperf_send.sh")
# time.sleep(30)

fail = False
for pod in range(1, 5):
    s1 = (pod - 1) * 2 + 1
    s2 = (pod - 1) * 2 + 2
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    with open("tcpdump_logs/log{}_1.output".format(s1), "r") as f:
        contents = f.read()
        d1 = len(contents)
    with open("tcpdump_logs/log{}_2.output".format(s1), "r") as f:
        contents = f.read()
        d2 = len(contents)
    with open("tcpdump_logs/log{}_1.output".format(s2), "r") as f:
        contents = f.read()
        d3 = len(contents)
    with open("tcpdump_logs/log{}_2.output".format(s2), "r") as f:
        contents = f.read()
        d4 = len(contents)
    avg = (d1 + d2 + d3 + d4) / 4.0
    if d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0:
        fail = True
    dev = ((d1 - avg) * (d1 - avg) + (d2 - avg) * (d2 - avg) + (d3 - avg) * (d3 - avg) + (d4 - avg) * (d4 - avg)) / 4.0
    dev = math.sqrt(dev)
    dev = dev / avg
    print(dev)
    if abs(dev) > 0.25:
        fail = True

if not fail:
    print("Test pass")
else:
    print("Test fail")