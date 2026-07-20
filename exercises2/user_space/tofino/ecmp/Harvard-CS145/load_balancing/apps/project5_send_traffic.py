""" Usage
sudo python send_traffic_asym.py [duration_in_seconds] [random_seed]
- Example:
    sudo python send_traffic_asym.py 10 10000 (send 10s traffic)
    The random seed is the seed to generate random source and destination ports.
"""

import random
import time
from p4utils.utils.helper import load_topo
from subprocess import Popen, DEVNULL
import sys
import os

path = "log"
if not os.path.exists(path):
    os.makedirs(path)

if len(sys.argv) == 3:
    seed = int(sys.argv[2])
else:
    seed = 1237

random.seed(seed)
topo = load_topo("topology.json")


# -b control the bandwidth of the flow (2.5Mbps)
iperf_send = "mx {0} iperf3 -c {1} -l 1000 -t {2} --bind {3} -p {4} --cport {5} 2>&1 > log/iperf_client_{6}.log"
iperf_recv = "mx {0} iperf3 -s -p {1} --one-off 2>&1 > log/iperf_server_{2}.log"

# print("#############CLEAN UP DANGLING IPERFS...###########")
Popen("sudo killall iperf iperf3", shell=True, stdout=DEVNULL, stderr=DEVNULL)

used_ports = []
c_ports = []

duration = int(sys.argv[1])
print("send duration: " + str(duration) + " sec")

num_flows = 2
for x in range(num_flows):
    port = random.randint(1024, 65000)
    while port in used_ports:
        port = random.randint(1024, 65000)
    used_ports.append(port)

    port = random.randint(1024, 65000)
    while port in c_ports:
        port = random.randint(1024, 65000)
    c_ports.append(port)

print("random seed: " + str(seed))
print(
    "Src port and Dst port for flow 1: [" + str(used_ports[0]) + ", ",
    str(c_ports[0]) + "]",
)
print(
    "Src port and Dst port for flow 2: [" + str(used_ports[1]) + ", ",
    str(c_ports[1]) + "]",
)


Popen(iperf_recv.format("h13", used_ports[0], 0), shell=True)
Popen(iperf_recv.format("h16", used_ports[1], 1), shell=True)

time.sleep(2)

Popen(
    iperf_send.format(
        "h1",
        topo.get_host_ip("h13"),
        duration,
        topo.get_host_ip("h1"),
        used_ports[0],
        c_ports[0],
        0,
    ),
    shell=True,
)
Popen(
    iperf_send.format(
        "h3",
        topo.get_host_ip("h16"),
        duration,
        topo.get_host_ip("h3"),
        used_ports[1],
        c_ports[1],
        1,
    ),
    shell=True,
)

time.sleep(duration + 2)


print("############ STATS FOR FLOW1 ###########")
with open("log/iperf_client_0.log") as f:
    print(f.read())
print("############ STATS FOR FLOW2 ###########")
with open("log/iperf_client_1.log") as f:
    print(f.read())
print("############ For more information, check files under log/ ###########")
