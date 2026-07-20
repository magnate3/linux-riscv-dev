#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "trace.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using json = nlohmann::json;

DEFINE_string(host, "", "The host name to run the traffic receiver");
DEFINE_string(topofile, "topology.json", "The topology database JSON file");
DEFINE_uint64(start_time, 0, "The start time of the traffic");
DEFINE_string(protocol, "udp", "The protocol to use");
DEFINE_string(logdir, "logs", "The log file to record the traffic");
DEFINE_bool(verbose, false, "Print verbose messages");

static std::vector<size_t> udp_bytes_received_by_window;

void killHandler(int signum) {
  std::cout << "Writing throughputs to log file" << std::endl;
  std::ofstream throughput_file(FLAGS_logdir + "/" + FLAGS_host +
                                "_iperf_server.log");
  for (size_t i = 0; i < udp_bytes_received_by_window.size(); i++) {
    throughput_file << udp_bytes_received_by_window[i] << std::endl;
  }
  throughput_file.close();
}

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

void threadReceiveTraffic(int connFd) {
  printf("Receiving traffic...\n");
  const size_t rate_report_period = 5000000;
  char buf[65536];
  double start_time = GetTimeUs();
  size_t total_bytes_received = 0;
  size_t cur_bytes_received = 0;
  size_t cur_report_num = 0;
  while (true) {
    int n = recv(connFd, buf, 1400, 0);
    total_bytes_received += n;
    cur_bytes_received += n;
    if (n == 0) {
      printf("Connection closed\n");
      break;
    }
    if (n < 0) {
      perror("recv error");
      break;
    }
    if (FLAGS_verbose &&
        IsTimePassed(start_time + (cur_report_num + 1) * rate_report_period)) {
      std::cout << "Received "
                << cur_bytes_received * 8 / 1000 * 1000000 / rate_report_period
                << "Kbps" << std::endl;
      cur_bytes_received = 0;
      cur_report_num++;
    }
  }
  double end_time = GetTimeUs();
  std::cout << "Total received "
            << total_bytes_received * 8 / 1000 * 1000000 /
                   (end_time - start_time)
            << "Kbps" << std::endl;
}

void LaunchTcpServer(std::string host_ip_address) {
  addrinfo hints, *res, *p;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  int status = getaddrinfo(host_ip_address.c_str(), "5001", &hints, &res);
  if (status != 0) {
    std::cerr << "getaddrinfo: " << gai_strerror(status) << std::endl;
    return;
  }

  int sockfd = -1;
  for (p = res; p != NULL; p = p->ai_next) {
    sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (sockfd == -1) {
      continue;
    }
    if (bind(sockfd, p->ai_addr, p->ai_addrlen) == 0) {
      break;
    }
    close(sockfd);
  }

  if (p == NULL) {
    std::cerr << "failed to bind" << std::endl;
    return;
  }

  freeaddrinfo(res);
  listen(sockfd, 5);
  while (true) {
    int client_fd = accept(sockfd, NULL, NULL);
    if (client_fd == -1) {
      std::cerr << "accept: " << strerror(errno) << std::endl;
      continue;
    }
    std::thread t(threadReceiveTraffic, client_fd);
    t.detach();
  }
}

size_t LaunchUdpServer(std::string host_ip_address) {
  const size_t rate_report_period = 5000000;

  addrinfo hints, *res, *p;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_flags = AI_PASSIVE;

  std::cout << "host_ip_address: " << host_ip_address << std::endl;

  int status = getaddrinfo(host_ip_address.c_str(), "5001", &hints, &res);
  if (status != 0) {
    std::cerr << "getaddrinfo: " << gai_strerror(status) << std::endl;
    return 0;
  }

  int sockfd = -1;
  for (p = res; p != NULL; p = p->ai_next) {
    sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (sockfd == -1) {
      continue;
    }
    if (bind(sockfd, p->ai_addr, p->ai_addrlen) == 0) {
      break;
    }
    close(sockfd);
  }

  if (p == NULL) {
    std::cerr << "failed to bind" << std::endl;
    return 0;
  }

  freeaddrinfo(res);
  // size_t start_time = GetTimeUs();
  size_t cur_bytes_received = 0;
  size_t total_bytes_received = 0;
  size_t cur_report_num = 0;
  struct sockaddr_in cliaddr;
  unsigned int len = sizeof(cliaddr);
  WaitUntil(FLAGS_start_time * 1000000);
  while (true) {
    char buf[65536];
    int n = recvfrom(sockfd, buf, sizeof(buf), MSG_WAITALL,
                     (struct sockaddr *)&cliaddr, &len);
    cur_bytes_received += n;
    total_bytes_received += n;
    if (n == 0) {
      break;
    }
    if (n < 0) {
      perror("recv");
      break;
    }
    // sendto(sockfd, buf, n, MSG_CONFIRM, (const struct sockaddr *) &cliaddr,
    // len);
    if (FLAGS_verbose &&
        IsTimePassed(FLAGS_start_time * 1000000 +
                     (cur_report_num + 1) * rate_report_period)) {
      std::cout << "Received "
                << cur_bytes_received * 8 / 1000 * 1000000 / rate_report_period
                << "Kbps" << std::endl;
      udp_bytes_received_by_window.push_back(cur_bytes_received * 8 / 1000 *
                                             1000000 / rate_report_period);
      cur_bytes_received = 0;
      cur_report_num++;
    }
  }

  return cur_report_num;
}

int main(int argc, char **argv) {
  signal(SIGTERM, killHandler);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
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

  auto &node_list_json = topo_db_json["nodes"];
  std::string host_ip_address = node_list_json[0]["ip"];
  for (auto &node_json : node_list_json) {
    if (node_json["id"] == FLAGS_host) {
      host_ip_address = node_json["ip"];
      break;
    }
  }
  host_ip_address = host_ip_address.substr(0, host_ip_address.find("/"));

  udp_bytes_received_by_window.reserve(1024);

  // Listen to the port and receive the traffic
  if (FLAGS_protocol == "udp") {
    size_t report_num = LaunchUdpServer(host_ip_address);
  } else if (FLAGS_protocol == "tcp") {
    LaunchTcpServer(host_ip_address);
  } else {
    std::cerr << "Invalid protocol: " << FLAGS_protocol << std::endl;
    return -1;
  }

  return 0;
}
