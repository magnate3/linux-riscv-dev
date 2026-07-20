"""
Usage
sudo python send_traffic_asym.py [duration_in_seconds] [random_seed]
- Example: 
    sudo python send_traffic_asym.py 10 10000 (send 10s traffic)
"""
import random
import time
from p4utils.utils.helper import load_topo
from subprocess import Popen
import sys



if len(sys.argv) == 3:
    seed = int(sys.argv[2])
else:
    seed = 10004

random.seed(seed)
topo = load_topo("topology.json")


iperf_send = "mx {0} iperf3 -c {1} -b 600K -l 1000 -t {2} --bind {3} -p {4} --cport {5} 2>&1 > log/iperf_client_{6}.log"
iperf_recv = "mx {0} iperf3 -s -p {1} --one-off 2>&1 > log/iperf_server_{2}.log"

Popen("sudo killall iperf iperf3", shell=True)

used_ports = []
c_ports = []

duration = int(sys.argv[1]) if len(sys.argv) == 3 else 30


num_flows = 8
for x in range(num_flows):
    port = random.randint(1024, 65000)
    while port in used_ports:
        port = random.randint(1024, 65000)
    used_ports.append(port)

    port = random.randint(1024, 65000)
    while port in c_ports:
        port = random.randint(1024, 65000)
    c_ports.append(port)


print(used_ports)
print(c_ports)
print("num_flows : "+ str(num_flows))



for i,port in enumerate(used_ports):
	Popen(iperf_recv.format("h2", port, i), shell=True)

time.sleep(2)


for i,port in enumerate(used_ports):
	Popen(iperf_send.format("h1", topo.get_host_ip("h2"), duration, topo.get_host_ip("h1"), port, c_ports[i], i), shell=True)

print("Waiting for all flows finished")
time.sleep(duration+20)

sum_thpt = 0
for i in range(num_flows):
    with open("log/iperf_server_{}.log".format(i)) as f:
        tokens = f.readlines()[-1].split()
        sum_thpt += int(tokens[-3])

print("Total throughput: {} Kbps".format(sum_thpt))



