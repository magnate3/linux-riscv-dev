#!/usr/bin/python3

import random as rdm
import sys
import json
from p4utils.utils.helper import load_topo


def get_ip_from_hostname(hostname, topo):
    intfs = topo.get_interfaces(hostname)
    intf = intfs[0]
    return topo.node_interface_ip(hostname, intf)


alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def generate_random_string():
    string_len = rdm.randint(5, 8)
    res = ""
    for i in range(string_len):
        alphabet_idx = rdm.randint(0, len(alphabet) - 1)
        res += alphabet[alphabet_idx]
    return res


mc_key_list = []


class Trace:
    host = ""
    start_time = 0
    trace_type = 0
    ip_address = ""
    length = 0
    flowlet_size = 0
    flowlet_gap = 0
    mc_key = ""
    mc_value = ""

    def generate_string(self):
        res_str = self.host + " " + str(self.start_time) + " " + str(self.trace_type)
        if self.trace_type == 0:
            res_str += " " + self.mc_key
        elif self.trace_type == 1:
            res_str += " " + self.mc_key + " " + self.mc_value
        elif self.trace_type == 2:
            res_str += (
                " "
                + self.ip_address
                + " "
                + str(self.length)
                + " "
                + str(self.flowlet_size)
                + " "
                + str(self.flowlet_gap)
            )
        return res_str


class Distribution:
    dist_type = 0
    value1 = 0
    value2 = 0

    def generate_value(self):
        if self.dist_type == 0:
            return self.value1
        else:
            return rdm.randint(self.value1, self.value2)


def ConstantDistribution(value):
    dist = Distribution()
    dist.dist_type = 0
    dist.value1 = value
    return dist


def UniformDistribution(min_val, max_val):
    if min_val < 0 or max_val < 0:
        print("Uniform distribution cannot have negative values")
        exit()
    if min_val >= max_val:
        print("Uniform distribution: max value must be larger than min value")
        exit()
    dist = Distribution()
    dist.dist_type = 1
    dist.value1 = min_val
    dist.value2 = max_val
    return dist


class FlowGroup:
    start_time = 0
    length = 0
    src_host_list = []
    dst_host_list = []
    flow_size_dist = None
    flow_gap_dist = None
    flowlet_size_dist = None
    flowlet_gap_dist = None


class Config:
    flow_group_list = []
    mc_host_list = []
    mc_gap_distribution = None
    length = 0
    output = None


def parse_distribution(jsonDist):
    if jsonDist["type"] == "uniform":
        return UniformDistribution(jsonDist["min"], jsonDist["max"])
    elif jsonDist["type"] == "constant":
        return ConstantDistribution(jsonDist["value"])
    else:
        print("Invalid distribution type detected!")
        exit()


def parse_json(json_config):
    cfg = Config()
    jsonFlowGroupList = json_config["flow_groups"]
    for i in range(len(jsonFlowGroupList)):
        jsonFlowGroup = jsonFlowGroupList[i]
        flow_group = FlowGroup()
        flow_group.start_time = jsonFlowGroup["start_time"]
        flow_group.length = jsonFlowGroup["length"]
        flow_group.src_host_list = jsonFlowGroup["src_host_list"]
        flow_group.dst_host_list = jsonFlowGroup["dst_host_list"]
        flow_group.flow_size_dist = parse_distribution(
            jsonFlowGroup["flow_size_distribution"]
        )
        flow_group.flow_gap_dist = parse_distribution(
            jsonFlowGroup["flow_gap_distribution"]
        )
        if (
            flow_group.flow_gap_dist.dist_type == 0
            and flow_group.flow_gap_dist.value1 == 0
        ):
            print("Flow gap distribution cannot be a constant zero")
            exit()
        flow_group.flowlet_size_dist = parse_distribution(
            jsonFlowGroup["flowlet_size_distribution"]
        )
        flow_group.flowlet_gap_dist = parse_distribution(
            jsonFlowGroup["flowlet_gap_distribution"]
        )
        cfg.flow_group_list.append(flow_group)
    cfg.mc_host_list = json_config["mc_host_list"]
    cfg.mc_gap_distribution = parse_distribution(json_config["mc_gap_distribution"])
    cfg.length = json_config["length"]
    cfg.output = json_config["output"]
    return cfg


def generate_flow_group(cfgFlowGroup, topo):
    trace_list = []
    start_time = cfgFlowGroup.start_time
    end_time = cfgFlowGroup.length + start_time
    current_time = start_time
    num_src_host = len(cfgFlowGroup.src_host_list)
    num_dst_host = len(cfgFlowGroup.dst_host_list)
    while current_time < end_time:
        src_host = cfgFlowGroup.src_host_list[rdm.randint(0, num_src_host - 1)]
        dst_host = cfgFlowGroup.dst_host_list[rdm.randint(0, num_dst_host - 1)]
        while dst_host == src_host:
            dst_host = cfgFlowGroup.dst_host_list[rdm.randint(0, num_dst_host - 1)]
        length = cfgFlowGroup.flow_size_dist.generate_value()
        trace = Trace()
        trace.host = src_host
        trace.start_time = current_time
        trace.trace_type = 2
        trace.ip_address = get_ip_from_hostname(dst_host, topo)
        trace.length = length
        trace.flowlet_size = cfgFlowGroup.flowlet_size_dist.generate_value()
        trace.flowlet_gap = cfgFlowGroup.flowlet_gap_dist.generate_value()
        trace_list.append(trace)
        current_time += cfgFlowGroup.flow_gap_dist.generate_value()
    return trace_list


class MemcachedRequest:
    src_host = None
    start_time = 0
    request_type = 0
    key = None
    value = None

    def generate_string(self):
        res = (
            self.src_host
            + " "
            + str(self.start_time)
            + " "
            + str(self.request_type)
            + " "
            + self.key
        )
        if self.request_type == 0:
            res += " " + self.value
        return res


def generate_mc_requests(cfgMcList, cfgMcDist, length):
    # Skip memcached requests if there are no mchosts
    if len(cfgMcList) == 0:
        return []
    mc_gap_distribution = cfgMcDist
    res = []
    current_time = 0
    host_idx = rdm.randint(0, len(cfgMcList) - 1)
    request = MemcachedRequest()
    request.src_host = cfgMcList[host_idx]
    request.request_type = 0
    request.key = generate_random_string()
    request.value = generate_random_string()
    mc_key_list.append(request.key)
    res.append(request)
    current_time += mc_gap_distribution.generate_value()
    while current_time < length:
        host_idx = rdm.randint(0, len(cfgMcList) - 1)
        request = MemcachedRequest()
        request.src_host = cfgMcList[host_idx]
        request.start_time = current_time
        request.request_type = rdm.randint(0, 1)
        if request.request_type == 0:
            toss = rdm.randint(0, 1)
            if toss == 0:
                key_idx = rdm.randint(0, len(mc_key_list) - 1)
                request.key = mc_key_list[key_idx]
                request.value = generate_random_string()
            else:
                request.key = generate_random_string()
                request.value = generate_random_string()
                mc_key_list.append(request.key)
        else:
            key_idx = rdm.randint(0, len(mc_key_list) - 1)
            request.key = mc_key_list[key_idx]
        res.append(request)
        current_time += mc_gap_distribution.generate_value()
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: ./generate_trace.py [config file]")
        exit()

    config_filename = sys.argv[1]
    json_config = None
    with open(config_filename, "r") as f:
        json_config = json.load(f)

    if json_config is None:
        print("Cannot open the config file!")
        exit()

    topo = load_topo("topology.json")

    cfg = parse_json(json_config)
    trace_list = []
    for i in range(len(cfg.flow_group_list)):
        trace_list.extend(generate_flow_group(cfg.flow_group_list[i], topo))

    mc_request_list = generate_mc_requests(
        cfg.mc_host_list, cfg.mc_gap_distribution, cfg.length
    )

    with open(cfg.output, "w") as f:
        mc_ip_str = ""
        for host in json_config["mc_host_list"]:
            mc_ip_str += host + " " + get_ip_from_hostname(host, topo) + " "
        f.write(mc_ip_str + "\n")
        trace_idx = 0
        mc_request_idx = 0
        while trace_idx < len(trace_list) or mc_request_idx < len(mc_request_list):
            if trace_idx == len(trace_list):
                f.write(mc_request_list[mc_request_idx].generate_string() + "\n")
                mc_request_idx += 1
            elif mc_request_idx == len(mc_request_list):
                f.write(trace_list[trace_idx].generate_string() + "\n")
                trace_idx += 1
            else:
                if (
                    mc_request_list[mc_request_idx].start_time
                    < trace_list[trace_idx].start_time
                ):
                    f.write(mc_request_list[mc_request_idx].generate_string() + "\n")
                    mc_request_idx += 1
                else:
                    f.write(trace_list[trace_idx].generate_string() + "\n")
                    trace_idx += 1
