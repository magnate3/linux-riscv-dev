import memcache
import sys
from utils import measure_time, wait_util
from memcached_trace import MemcachedTrace


class Client:
    def __init__(self, start_time, actions, server_ip):
        self.start_time = start_time
        self.actions = actions
        self.mc = memcache.Memcache([(ip, 11211) for ip in server_ip])

    def work(self):
        for action in self.actions:
            latency = self.execute(action)
            print(f"{latency * 1e6:.0f}")  # us

    def execute(self, action):
        wait_util(self.start_time + action.time)
        if action.func == 0:
            return measure_time(lambda: self.mc.set(action.key, action.value))
        else:
            return measure_time(lambda: self.mc.get(action.key))


def read_traffic_file(host_name, traffic_file):
    actions = []
    with open(traffic_file, "r") as file:
        lines = file.readlines()
        tokens = lines[0].strip().split(" ")
        assert len(tokens) % 2 == 0
        server_ip = []
        idx = 1
        while idx < len(tokens):
            server_ip.append(tokens[idx])
            idx += 2
        for line in lines[1:]:  # rest of the file are actions
            tokens = line.strip().split(" ")  # host_name, time, func, key[, value]
            if tokens[0] != host_name:
                continue
            if tokens[2] == "0":  # set: time, func, key, value
                actions.append(
                    MemcachedTrace(
                        float(tokens[1]) / 1000000.0, 0, tokens[3], tokens[4]
                    )
                )
            elif tokens[2] == "1":  # get: time, func, key
                actions.append(
                    MemcachedTrace(float(tokens[1]) / 1000000.0, 1, tokens[3])
                )
    return server_ip, actions


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "usage: python memcached_client.py <start_time> <host_name> <traffic_file>"
        )
        sys.exit(1)
    start_time = float(sys.argv[1])  # we indicate a global start time
    host_name = sys.argv[2]
    traffic_file = sys.argv[3]

    # use host_name to find the actions
    server_ip, actions = read_traffic_file(host_name, traffic_file)
    client = Client(start_time, actions, server_ip)
    client.work()
