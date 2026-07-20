# Inspired by https://github.com/vpenso/ganglia-sensors/blob/master/lib/python_modules/infiniband.py#/

import logging
import re
import sys
import json
import time
import subprocess


METRIC_NAMES = ["PortXmitData","PortRcvData"]
metrics = {}

def decode_str_list(line_list):
  return [x.decode("utf-8") for x in line_list]

def get_cmd_out(cmd):
    return decode_str_list(subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.readlines())

def ibstat_ports():
    lid7port = []
    ibstat = get_cmd_out("ibstat")
    for index,line in enumerate(ibstat):
        line = line.strip()
        match = re.match("Port [0-9]\:",line)
        if match:
            number = line.split(' ')[1].replace(':','')
            state = ibstat[index+1].split(':')[1].strip()
            an = re.match("Active",state)
            if an:
                lid = ibstat[index+4].split(':')[1].strip()
                lid7port.append((lid, number))
    return lid7port

# Return a key-value pair, eventually empty if the line didn't match
def parse_counter_line(line, keys):
    if re.match("^[a-zA-z0-9]*\:\.\.\.*[0-9]*$",line):
        line = line.split(':')
        key = line[0]
        if key in keys:
            value = line[1].replace('.','').strip()
            return (key, int(value))
    return ("",0)

# Parse the complete input from perfquery for lines matching counters,
# and return all counters and their values as dictionary
def parse_counters(counters, keys):
    counts = {}
    for line in counters:
        key, value = parse_counter_line(line, keys)
        # Omit empty return values...
        if key:
          logging.debug("[parse_counters] Found counter: %s=%s", key, value)
          counts[key] = value
    return counts

# Call perfquery for extended traffic counters, and reset the counters
def traffic_counter(lid, port = 1):
    command = ["/usr/sbin/perfquery", "-x", "-r", lid, port]
    logging.debug("[traffic_counters] Execute command: %s", " ".join(command))
    counters = get_cmd_out(command)
    return parse_counters(counters, METRIC_NAMES)

def init_metric():
    metrics["last_update"] = time.time()

def update_metric():
    global metrics

    # NOTE: time_since_last_update is not calculated precisely
    time_since_last_update = time.time() - metrics["last_update"]
    logging.debug("[update_metrics] Update metrics after %ss", time_since_last_update)

    for lid, port in ibstat_ports():
        metric2counts = traffic_counter(lid, port)
        metrics[lid] = {port: metric2counts}
        for metric in METRIC_NAMES:
            # Data port counters indicate octets divided by 4 rather than just octets.
            #
            # It's consistent with what the IB spec says (IBA 1.2 vol 1 p.948) as to
            # how these quantities are counted. They are defined to be octets divided
            # by 4 so the choice is to display them the same as the actual quantity
            # (which is why they are named Data rather than Octets) or to multiply by
            # 4 for Octets. The former choice was made.
            #
            # For simplification the values are multiplied by 4 to represent octets/bytes
            num_bytes = metric2counts[metric] * 4
            metrics[lid][port][metric.replace("Data", "Bytes")] = num_bytes
            metrics[lid][port][metric.replace("Data", "GB/s")] = num_bytes / (time_since_last_update * 1024*1024*1024)


    metrics["last_update"] = time.time()

if __name__ == '__main__':

    logging.root.setLevel(logging.INFO)
    update_interval = 10 if len(sys.argv) == 1 else sys.argv[1] # default is 10s

    init_metric()

    while True:
        update_metric()
        print("Note: This is a **Rough** traffic monitor for Infiniband, the bw below may be bigger than real bw")
        print(json.dumps(metrics, indent=2, sort_keys=True))

        time.sleep(update_interval)
