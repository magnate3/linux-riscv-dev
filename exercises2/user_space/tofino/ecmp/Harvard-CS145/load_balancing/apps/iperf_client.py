import socket
import threading
import time
import sys
import Queue
from utils import wait_util
from iperf_trace import IperfTrace


class IperfClient:
    def __init__(self, start_time, traffics, port_num=5001):
        self.start_time, self.traffics, self.port_num = start_time, traffics, port_num
        self.buf = "x" * 5000

    def work(self):
        th = []
        q = Queue.Queue()
        for traffic in self.traffics:
            t = threading.Thread(target=self.execute, args=(traffic, q))
            th.append(t)
            t.start()
        print(f"There are a total of {len(th)} threads")
        for t in th:
            t.join()
        print(
            "# The following numbers represent the throughput of each iperf requests "
            "from this iperf client. The unit is byte per second"
        )
        while not q.empty():
            traffic, bps = q.get()
            print(bps)
            sys.stdout.flush()

    def execute(self, traffic, q):
        byte = 0
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        addr = (traffic.dst_ip, traffic.port)
        print(addr)
        sock.connect(addr)

        end_time = self.start_time + traffic.time + traffic.duration
        wait_util(self.start_time + traffic.time)
        start = time.time()
        now = start
        while now < end_time:
            byte += sock.send(self.buf)
            now = time.time()
        ret = float(byte) / (now - start) * 8
        q.put((traffic, ret))
        return ret


def read_traffic_file(host_name, traffic_file):
    traffics = []
    cur_iperf_port = 5001
    with open(traffic_file, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            tokens = line.strip().split(" ")  # host_name, time, func, dst_ip, duration
            if tokens[2] != "2":  # filter out non-iperf traffic
                continue
            if tokens[0] == host_name:
                traffics.append(
                    IperfTrace(
                        float(tokens[1]), tokens[3], cur_iperf_port, float(tokens[4])
                    )
                )
            cur_iperf_port += 1
    return traffics


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python iperf_client.py <start_time> <host_name> <traffic_file>")
        sys.exit(1)
    start_time = float(sys.argv[1])  # we indicate a global start time
    host_name = sys.argv[2]
    traffic_file = sys.argv[3]

    # use host_name to find the traffic
    traffics = read_traffic_file(host_name, traffic_file)
    client = IperfClient(start_time, traffics)
    client.work()
