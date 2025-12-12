#pragma once

#include <nccl.h>
#include <memory>
#include <string>
#include <vector>

class TCPSocket {
 public:
  TCPSocket();
  ~TCPSocket();

  TCPSocket(const TCPSocket&) = delete;
  TCPSocket& operator=(const TCPSocket&) = delete;

  bool listen(int port, int backlog = 10);
  std::unique_ptr<TCPSocket> accept();

  bool connect(const std::string& host, int port);

  bool send(const void* data, size_t size);
  bool recv(void* buffer, size_t size);

  bool isConnected() const;
  void close();

 private:
  int socketFD;
  bool connected;
};

class NCCLIdBroadcaster {
 public:
  static bool broadcastNCCLId(ncclUniqueId& ncclId, int nodeRank, int worldSize, const std::string& masterIP, int port);
};