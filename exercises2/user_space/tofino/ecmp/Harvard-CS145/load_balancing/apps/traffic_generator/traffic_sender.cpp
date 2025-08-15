#include <chrono>
#include <cstdio>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "trace.hpp"

#include <arpa/inet.h>
#include <ctime>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using json = nlohmann::json;

DEFINE_string(host, "", "The host name to run the traffic sender");
DEFINE_string(tracefile, "apps/trace/trace.txt",
              "The trace filename to replay");
DEFINE_string(topofile, "topology.json", "The topology database JSON file");
DEFINE_string(protocol, "udp", "The protocol to use");
DEFINE_uint64(start_time, 0, "The start time of the traffic");
DEFINE_string(logdir, "logs", "The log file to record the traffic");
DEFINE_bool(verbose, false, "Print verbose messages");
DEFINE_uint64(bandwidth, 1000000, "The bandwidth of the sender");
DEFINE_uint64(port, 0, "Assign specific port");

std::vector<std::thread *> all_client_threads;
std::vector<double> all_client_thread_throughputs;

int GetHostListFromTopoDB(json &topo_db_json,
                          std::vector<std::string> &host_list) {
  host_list.clear();
  for (json::iterator it = topo_db_json["nodes"].begin();
       it != topo_db_json["nodes"].end(); ++it) {
    if ((*it)["isHost"] == true) {
      host_list.push_back((*it)["id"]);
    }
  }
  return 0;
}

static inline double GetTimeUs() {
  struct timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  return tv.tv_sec * 1000000 + tv.tv_nsec / 1000.0;
}

int WaitUntil(double time) {
  double cur_time = GetTimeUs();
  if (time > cur_time) {
    std::this_thread::sleep_for(
        std::chrono::microseconds((int)(time - cur_time)));
  }
  return 0;
}

int IsTimePassed(double time) {
  double cur_time = GetTimeUs();
  if (cur_time >= time) {
    return 1;
  }
  return 0;
}

void threadTcpConnection(size_t tid, std::string host, std::string dst_ip_addr,
                         double flow_start_time, double flow_duration,
                         double flowlet_duration, double flowlet_gap) {
  std::cout << "Connecting to " << host << " at " << dst_ip_addr << std::endl;
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    std::cerr << "Error: failed to create socket" << std::endl;
    return;
  }

  if (FLAGS_port != 0) {
    struct sockaddr_in client_addr;
    bzero(&client_addr, sizeof(client_addr));
    client_addr.sin_family = AF_INET;
    client_addr.sin_port = htons(FLAGS_port);
    if (bind(sockfd, (struct sockaddr *)&client_addr,
             sizeof(struct sockaddr_in)) == 0)
      printf("Binded Correctly\n");
    else
      printf("Unable to bind\n");
  }

  struct sockaddr_in serv_addr;
  bzero(&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(5001);
  inet_pton(AF_INET, dst_ip_addr.c_str(), &serv_addr.sin_addr);

  if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    std::cerr << "Error: failed to connect to " << dst_ip_addr << std::endl;
    return;
  }

  char buf[8192];
  for (int i = 0; i < 8192; i++) {
    buf[i] = 'a';
  }

  const size_t rate_report_period = 5000000;
  size_t cur_report_num = 1;
  size_t cur_bytes_sent = 0;
  size_t total_bytes_sent = 0;
  if (FLAGS_verbose) {
    std::cout << "Wait for " << (flow_start_time - GetTimeUs()) / 1000000
              << " seconds" << std::endl;
  }
  WaitUntil(flow_start_time);
  if (FLAGS_verbose) {
    std::cout << "threadTcpConnection: start sending" << std::endl;
  }
  double cur_time = GetTimeUs();
  std::cout << "Current time: " << cur_time
            << ", flow start time: " << flow_start_time << std::endl;
  int state = 1;
  double next_state_transition_time = flowlet_duration < 1e-8
                                          ? flow_start_time + flow_duration
                                          : flow_start_time + flowlet_duration;
  while (cur_time < flow_start_time + flow_duration) {
    if (state == 1) {
      int bytes_sent = send(sockfd, buf, 1400, 0);
      if (bytes_sent >= 0) {
        cur_bytes_sent += bytes_sent;
        total_bytes_sent += bytes_sent;
        std::this_thread::sleep_for(std::chrono::microseconds(
            1000000UL * bytes_sent * 8 / FLAGS_bandwidth));
      }
      if (IsTimePassed(next_state_transition_time)) {
        state = 0;
        next_state_transition_time += flowlet_gap;
      }
    } else if (state == 0) {
      if (IsTimePassed(next_state_transition_time)) {
        state = 1;
        next_state_transition_time += flowlet_duration;
      }
    }
    cur_time = GetTimeUs();
    if (FLAGS_verbose &&
        IsTimePassed(flow_start_time + cur_report_num * rate_report_period)) {
      std::cout << "Sent "
                << cur_bytes_sent * 8 / 1000 * 1000000 / rate_report_period
                << "Kbps"
                << " " << (cur_time - flow_start_time) / 1000000.0 << std::endl;
      cur_report_num++;
      cur_bytes_sent = 0;
    }
  }

  close(sockfd);

  cur_time = GetTimeUs();
  std::cout << "Total bytes sent: " << total_bytes_sent / 1000 << "Kbytes"
            << std::endl;
  std::cout << "Total time: " << (cur_time - flow_start_time) / 1000000
            << " seconds" << std::endl;
  std::cout << "Average throughput: "
            << total_bytes_sent * 8 / 1000 /
                   ((cur_time - flow_start_time) / 1000000)
            << "Kbps" << std::endl;
  all_client_thread_throughputs[tid] =
      total_bytes_sent * 8 / 1000 / ((cur_time - flow_start_time) / 1000000);
}

void LaunchTcpClient(std::string host, Trace &trace) {
  TraceItem item;
  size_t cur_tid = 0;
  while (trace.get_next_trace_item(host, &item) == 0) {
    WaitUntil(item.start_time_ + FLAGS_start_time * 1000000 - 1000000);
    std::cout << "Sending traffic to " << item.dst_host_or_mc_key_ << std::endl;
    std::thread *t = new std::thread(
        threadTcpConnection, cur_tid, host, item.dst_host_or_mc_key_,
        item.start_time_ + FLAGS_start_time * 1000000, item.flow_duration_,
        item.flowlet_duration_, item.flowlet_gap_);
    all_client_threads.push_back(t);
    all_client_thread_throughputs.push_back(0);
    cur_tid++;
  }
}

void threadUdpConnection(size_t tid, std::string host, std::string dst_ip_addr,
                         double flow_start_time, double flow_duration,
                         double flowlet_duration, double flowlet_gap) {
  std::cout << "threadUdpConnection: " << host << " " << dst_ip_addr << " "
            << flow_start_time << " " << flow_duration << std::endl;
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    std::cerr << "Error: failed to create socket" << std::endl;
    return;
  }

  struct sockaddr_in serv_addr;
  bzero(&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(5001);
  inet_pton(AF_INET, dst_ip_addr.c_str(), &serv_addr.sin_addr);

  char buf[8192];
  for (int i = 0; i < 8192; i++) {
    buf[i] = 'a';
  }

  size_t cur_report_num = 1;
  size_t cur_bytes_sent = 0;
  size_t cur_bytes_received = 0;
  size_t total_bytes_sent = 0;
  size_t total_bytes_received = 0;
  if (FLAGS_verbose) {
    std::cout << "Wait for " << (flow_start_time - GetTimeUs()) / 1000000
              << " seconds" << std::endl;
  }
  WaitUntil(flow_start_time);
  if (FLAGS_verbose) {
    std::cout << "threadUdpConnection: start sending" << std::endl;
  }
  double cur_time = GetTimeUs();
  std::cout << "Current time: " << cur_time
            << ", flow start time: " << flow_start_time << std::endl;
  int state = 1;
  unsigned int len;
  double next_state_transition_time = flowlet_duration < 1e-8
                                          ? flow_start_time + flow_duration
                                          : flow_start_time + flowlet_duration;
  while (cur_time < flow_start_time + flow_duration) {
    if (state == 1) {
      size_t bytes_sent =
          sendto(sockfd, buf, 1400, MSG_CONFIRM, (struct sockaddr *)&serv_addr,
                 sizeof(serv_addr));
      cur_bytes_sent += bytes_sent;
      total_bytes_sent += bytes_sent;
      // size_t bytes_received = recvfrom(sockfd, buf, sizeof(buf), MSG_WAITALL,
      // (struct sockaddr*)&serv_addr, &len); cur_bytes_received +=
      // bytes_received; total_bytes_received += bytes_received;
      if (IsTimePassed(next_state_transition_time)) {
        state = 0;
        next_state_transition_time += flowlet_gap;
      }
    } else if (state == 0) {
      if (IsTimePassed(next_state_transition_time)) {
        state = 1;
        next_state_transition_time += flowlet_duration;
      }
    }
    cur_time = GetTimeUs();
    if (FLAGS_verbose &&
        IsTimePassed(flow_start_time + cur_report_num * 1000000)) {
      std::cout << "Sent " << cur_bytes_sent * 8 / 1000 << "Kbps" << std::endl;
      cur_report_num++;
      cur_bytes_sent = 0;
    }
  }

  close(sockfd);

  cur_time = GetTimeUs();
  std::cout << "Total bytes sent: " << total_bytes_sent / 1000 << "Kbytes"
            << std::endl;
  std::cout << "Total time: " << (cur_time - flow_start_time) / 1000000
            << " seconds" << std::endl;
  std::cout << "Average throughput: "
            << total_bytes_sent * 8 / 1000 /
                   ((cur_time - flow_start_time) / 1000000)
            << "Kbps" << std::endl;
  all_client_thread_throughputs[tid] =
      total_bytes_sent * 8 / 1000 / ((cur_time - flow_start_time) / 1000000);
}

void LaunchUdpClient(std::string host, Trace &trace) {
  TraceItem item;
  size_t cur_tid = 0;
  while (trace.get_next_trace_item(host, &item) == 0) {
    WaitUntil(item.start_time_ + FLAGS_start_time * 1000000 - 1000000);
    std::cout << "Sending traffic to " << item.dst_host_or_mc_key_ << std::endl;
    std::thread *t = new std::thread(
        threadUdpConnection, cur_tid, host, item.dst_host_or_mc_key_,
        item.start_time_ + FLAGS_start_time * 1000000, item.flow_duration_,
        item.flowlet_duration_, item.flowlet_gap_);
    all_client_threads.push_back(t);
    all_client_thread_throughputs.push_back(0);
    cur_tid++;
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  printf("Trace file: %s\n", FLAGS_tracefile.c_str());
  printf("Topology database file: %s\n", FLAGS_topofile.c_str());

  std::ifstream topo_db_f(FLAGS_topofile.c_str());
  json topo_db_json = json::parse(topo_db_f);

  std::vector<std::string> host_list;
  GetHostListFromTopoDB(topo_db_json, host_list);
  if (find(host_list.begin(), host_list.end(), FLAGS_host) == host_list.end()) {
    std::cerr << "Host " << FLAGS_host << " not found in topology database"
              << std::endl;
    return 1;
  }

  if (FLAGS_start_time == 0) {
    std::cerr << "Start time is not specified" << std::endl;
    return 1;
  }

  Trace trace(FLAGS_tracefile, host_list);
  trace.print_traces();

  // Listen to the port and receive the traffic
  if (FLAGS_protocol == "tcp") {
    LaunchTcpClient(FLAGS_host, trace);
  } else if (FLAGS_protocol == "udp") {
    LaunchUdpClient(FLAGS_host, trace);
  } else {
    std::cerr << "Unknown protocol: " << FLAGS_protocol << std::endl;
    return 1;
  }

  for (auto &t : all_client_threads) {
    t->join();
  }

  std::cout << "All threads finished" << std::endl;

  std::ofstream throughput_file(FLAGS_logdir + "/" + FLAGS_host + "_iperf.log");
  for (size_t i = 0; i < all_client_thread_throughputs.size(); i++) {
    throughput_file << all_client_thread_throughputs[i] << std::endl;
  }
  throughput_file.close();

  return 0;
}
