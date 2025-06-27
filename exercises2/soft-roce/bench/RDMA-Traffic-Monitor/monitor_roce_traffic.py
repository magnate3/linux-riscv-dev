import logging 
import subprocess 
import time 
import json 
import sys 
import re 

ALL_METRIC_NAMES = [ 
    "rx_vport_unicast_packets", "rx_vport_unicast_bytes", 
    "tx_vport_unicast_packets", "tx_vport_unicast_bytes", 
    "rx_vport_multicast_packets", "rx_vport_multicast_bytes", 
    "tx_vport_multicast_packets", "tx_vport_multicast_bytes", 
    "rx_vport_broadcast_packets", "rx_vport_broadcast_bytes", 
    "tx_vport_broadcast_packets", "tx_vport_broadcast_bytes", 
    "rx_vport_rdma_unicast_packets", "rx_vport_rdma_unicast_bytes", 
    "tx_vport_rdma_unicast_packets", "tx_vport_rdma_unicast_bytes", 
    "rx_vport_rdma_multicast_packets", "rx_vport_rdma_multicast_bytes", 
    "tx_vport_rdma_multicast_packets", "tx_vport_rdma_multicast_bytes" 
] 

METRIC_NAMES = [ 
    "rx_vport_unicast_bytes", 
    "tx_vport_unicast_bytes", 
    "rx_vport_rdma_unicast_bytes", 
    "tx_vport_rdma_unicast_bytes", 
] 

metrics = {}

def decode_str_list(line_list): 
    return [x.decode("utf-8") for x in line_list]

def get_cmd_out(cmd): 
    return decode_str_list(subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.readlines())


def get_ethtool_stats(interface):
    cmd = ["ethtool", "-S", interface]
    output = get_cmd_out(cmd)
    stats = {}
    for line in output:
        match = re.match(r'\s*(\S+): (\d+)', line)
        if match:
            key = match.group(1)
            value = int(match.group(2))
            stats[key] = value
    return stats

def update_metric(last_update):
    global metrics

    time_since_last_update = time.time() - metrics.get("last update time", time.time())
    logging.debug("[update_metrics] Update metrics after %ss", time_since_last_update)

    devices = ["eth0"]
    for device in devices:
        stats = get_ethtool_stats(device)

        
        if device not in last_update:
            last_update[device] = {}

        if device not in metrics:
            metrics[device] = {}

        for metric in METRIC_NAMES:
            metric_tag = metric.replace("_vport", "").replace("_", " ")
            
            if metric in stats:
                current_value = stats[metric]
                last_value = last_update[device].get(metric_tag, current_value)
                increment = current_value - last_value
                metrics[device][metric_tag] = increment

                if "bytes" in metric:
                    metrics[device][metric_tag.replace("bytes", "MB/s")] = increment / (time_since_last_update * 1024 * 1024)

                last_update[device][metric_tag] = current_value

    metrics["last update time"] = time.time()
    return metrics

logging.root.setLevel(logging.INFO)
update_interval = 5
is_first_loop = True

while True: 
    if not is_first_loop: 
        updated_metrics = update_metric(metrics) 
        print("Traffic monitor for RoCE interfaces.") 
        filtered_metrics = {}
        for device in metrics:
            if isinstance(metrics[device], dict):
                filtered_metrics[device] = {metric: value for metric, value in metrics[device].items() if "MB/s" in metric}
        print(json.dumps(filtered_metrics, indent=2, sort_keys=True))
    else: 
        updated_metrics = update_metric(metrics) 
        print("Begin, please wait")
        
    is_first_loop = False 
    time.sleep(update_interval)
