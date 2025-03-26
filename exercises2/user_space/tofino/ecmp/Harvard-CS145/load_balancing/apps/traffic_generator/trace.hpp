#pragma once

#include <arpa/inet.h>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

class TraceItem {
public:
  std::string src_host_;
  double start_time_;
  size_t trace_type_;
  std::string dst_host_or_mc_key_;
  union {
    double flow_duration_;
    size_t mc_value_;
  };
  double flowlet_duration_;
  double flowlet_gap_;
};

class Trace {
  std::string trace_filename_;
  std::ifstream trace_f_;
  std::vector<TraceItem> trace_item_list_;
  std::set<std::string> host_list_;
  size_t cur_item_index_;

  int validateIpAddress(const std::string &ipAddress) {
    struct sockaddr_in sa;
    int result = inet_pton(AF_INET, ipAddress.c_str(), &(sa.sin_addr));
    return result != 0;
  }

public:
  Trace(std::string trace_filename, std::vector<std::string> &host_list) {
    trace_filename_ = trace_filename;
    trace_f_ = std::ifstream(trace_filename);
    for (auto host : host_list) {
      host_list_.insert(host);
    }
    read_trace_from_file();
    sort_trace_item_list();
    cur_item_index_ = 0;
  }

  int get_next_trace_item(std::string src_host, TraceItem *trace_item) {
    if (trace_item == nullptr) {
      return -1;
    }
    if (cur_item_index_ >= trace_item_list_.size()) {
      return -1;
    }
    while (cur_item_index_ < trace_item_list_.size()) {
      if (trace_item_list_[cur_item_index_].src_host_ == src_host &&
          trace_item_list_[cur_item_index_].trace_type_ == 2) {
        *trace_item = trace_item_list_[cur_item_index_];
        cur_item_index_++;
        return 0;
      }
      cur_item_index_++;
    }
    return -1;
  }

  int read_trace_from_file() {
    std::string line;
    std::getline(trace_f_, line);
    while (std::getline(trace_f_, line)) {
      TraceItem item;
      int ret = parse_trace_str(line, item);
      if (ret == 0) {
        trace_item_list_.push_back(item);
      }
    }
    return 0;
  }

  int print_traces() {
    for (auto item : trace_item_list_) {
      std::cout << item.src_host_ << " " << item.start_time_ << " "
                << item.trace_type_ << " " << item.dst_host_or_mc_key_ << " "
                << item.flow_duration_ << std::endl;
    }
    return 0;
  }

  int parse_trace_str(std::string trace_str, TraceItem &item) {
    std::string delimiter = " ";
    size_t idx = trace_str.find(delimiter, 0);
    std::string src_host = trace_str.substr(0, idx);
    if (host_list_.find(src_host) == host_list_.end()) {
      std::cerr << "Invalid hostname " << src_host << ", line: " << trace_str
                << std::endl;
      return -1;
    }
    item.src_host_ = src_host;

    size_t start_idx = idx + delimiter.length();
    idx = trace_str.find(delimiter, start_idx);
    std::string start_time_str = trace_str.substr(start_idx, idx - start_idx);
    double start_time = atof(start_time_str.c_str());
    if (start_time < 0) {
      std::cerr << "Invalid trace start time " << start_time << std::endl;
      return -1;
    }
    item.start_time_ = start_time;

    start_idx = idx + delimiter.length();
    idx = trace_str.find(delimiter, start_idx);
    std::string trace_type_str = trace_str.substr(start_idx, idx - start_idx);
    size_t trace_type = atoi(trace_type_str.c_str());
    if (trace_type > 2) {
      std::cerr << "Invalid trace type " << trace_type << std::endl;
      return -1;
    }
    item.trace_type_ = trace_type;

    start_idx = idx + delimiter.length();
    idx = trace_str.find(delimiter, start_idx);
    std::string token_str = trace_str.substr(start_idx, idx - start_idx);
    if (trace_type == 2) {
      int ret = validateIpAddress(token_str);
      if (ret != 1) {
        std::cerr << "Invalid IP address " << token_str << std::endl;
        return -1;
      }
    }
    item.dst_host_or_mc_key_ = token_str;

    start_idx = idx + delimiter.length();
    idx = trace_str.find(delimiter, start_idx);
    std::string number_str = trace_str.substr(start_idx, idx - start_idx);
    if (trace_type == 2) {
      double flow_duration = atof(number_str.c_str());
      if (flow_duration < 0) {
        std::cerr << "Invalid flow duration time " << flow_duration
                  << std::endl;
        return -1;
      }
      item.flow_duration_ = flow_duration;
    } else if (trace_type == 0) {
      size_t value = atoi(number_str.c_str());
      item.mc_value_ = value;
    }

    if (trace_type == 2) {
      start_idx = idx + delimiter.length();
      idx = trace_str.find(delimiter, start_idx);
      std::string flowlet_duration_str =
          trace_str.substr(start_idx, idx - start_idx);
      double flowlet_duration = atof(flowlet_duration_str.c_str());
      if (flowlet_duration < 0) {
        std::cerr << "Invalid flowlet duration time " << flowlet_duration
                  << std::endl;
        return -1;
      }
      item.flowlet_duration_ = flowlet_duration;

      start_idx = idx + delimiter.length();
      idx = trace_str.find(delimiter, start_idx);
      std::string flowlet_gap_str =
          trace_str.substr(start_idx, idx - start_idx);
      size_t flowlet_gap = atoi(flowlet_gap_str.c_str());
      if (flowlet_gap < 0) {
        std::cerr << "Invalid flowlet gap " << flowlet_gap << std::endl;
        return -1;
      }
      item.flowlet_gap_ = flowlet_gap;
    }

    return 0;
  }

  void sort_trace_item_list() {
    std::sort(trace_item_list_.begin(), trace_item_list_.end(),
              [](const TraceItem &a, const TraceItem &b) {
                return a.start_time_ < b.start_time_;
              });
  }
};
