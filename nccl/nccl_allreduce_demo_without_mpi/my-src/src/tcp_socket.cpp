#include "tcp_socket.h"
#include "utils.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
typedef int socklen_t;
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define SOCKET_ERROR (-1)
#  define INVALID_SOCKET (-1)
typedef int SOCKET;
#endif

TCPSocket::TCPSocket() : socketFD(INVALID_SOCKET), connected(false) {
#ifdef _WIN32
  static bool wsaInitialized = false;
  if (!wsaInitialized) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
      Logger::log("WSAStartup failed");
      return;
    }
    wsaInitialized = true;
  }
#endif

  socketFD = socket(AF_INET, SOCK_STREAM, 0);
  if (socketFD == INVALID_SOCKET) {
    Logger::log("Socket creation failed");
  }

  int flag = 1;
  if (setsockopt(socketFD, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) == SOCKET_ERROR) {
    Logger::log("Set TCP_NODELAY failed");
  }
}

TCPSocket::~TCPSocket() { close(); }

bool TCPSocket::listen(int port, int backlog) {
  if (socketFD == INVALID_SOCKET) {
    Logger::log("Invalid socket in listen()");
    return false;
  }

  int opt = 1;
  if (setsockopt(socketFD, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt)) == SOCKET_ERROR) {
    Logger::log("setsockopt(SO_REUSEADDR) failed");
  }

  struct sockaddr_in serverAddr;
  memset(&serverAddr, 0, sizeof(serverAddr));
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  serverAddr.sin_port = htons(port);

  if (bind(socketFD, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
    Logger::log("Bind failed on port", port);
    return false;
  }

  if (::listen(socketFD, backlog) == SOCKET_ERROR) {
    Logger::log("Listen failed on port", port);
    return false;
  }

  Logger::log("Server listening on port", port);
  return true;
}

std::unique_ptr<TCPSocket> TCPSocket::accept() {
  if (socketFD == INVALID_SOCKET) {
    Logger::log("Invalid socket in accept()");
    return nullptr;
  }

  struct sockaddr_in clientAddr;
  socklen_t clientAddrLen = sizeof(clientAddr);
  SOCKET clientSocketFD = ::accept(socketFD, (struct sockaddr*)&clientAddr, &clientAddrLen);

  if (clientSocketFD == INVALID_SOCKET) {
    Logger::log("Accept failed");
    return nullptr;
  }

  std::unique_ptr<TCPSocket> clientSocket(new TCPSocket());
  clientSocket->socketFD = clientSocketFD;
  clientSocket->connected = true;

  char clientIP[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &(clientAddr.sin_addr), clientIP, INET_ADDRSTRLEN);
  Logger::log("Accepted connection from", clientIP, ":", ntohs(clientAddr.sin_port));

  return clientSocket;
}

bool TCPSocket::connect(const std::string& host, int port) {
  if (socketFD == INVALID_SOCKET) {
    Logger::log("Invalid socket in connect()");
    return false;
  }

  struct sockaddr_in serverAddr;
  memset(&serverAddr, 0, sizeof(serverAddr));
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(port);

  if (inet_pton(AF_INET, host.c_str(), &serverAddr.sin_addr) <= 0) {
    struct addrinfo hints, *result = nullptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int status = getaddrinfo(host.c_str(), nullptr, &hints, &result);
    if (status != 0) {
      Logger::log("Invalid address / Address not supported:", host);
      return false;
    }

    struct sockaddr_in* addr = (struct sockaddr_in*)result->ai_addr;
    serverAddr.sin_addr = addr->sin_addr;
    freeaddrinfo(result);
  }

  const int maxRetries = 10;
  for (int retry = 0; retry < maxRetries; retry++) {
    if (::connect(socketFD, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) != SOCKET_ERROR) {
      connected = true;
      Logger::log("Connected to", host, ":", port);
      return true;
    }

    Logger::log("Connection attempt", retry + 1, "/", maxRetries, "to", host, ":", port, "failed, retrying...");

    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  Logger::log("Failed to connect to", host, ":", port, "after", maxRetries, "retries");
  return false;
}

bool TCPSocket::send(const void* data, size_t size) {
  if (!isConnected()) {
    Logger::log("Not connected in send()");
    return false;
  }

  const char* buffer = static_cast<const char*>(data);
  size_t totalSent = 0;

  while (totalSent < size) {
    int sent = ::send(socketFD, buffer + totalSent, size - totalSent, 0);
    if (sent == SOCKET_ERROR) {
      Logger::log("Send failed");
      return false;
    }

    totalSent += sent;
  }

  return true;
}

bool TCPSocket::recv(void* buffer, size_t size) {
  if (!isConnected()) {
    Logger::log("Not connected in recv()");
    return false;
  }

  char* buf = static_cast<char*>(buffer);
  size_t totalReceived = 0;

  while (totalReceived < size) {
    int received = ::recv(socketFD, buf + totalReceived, size - totalReceived, 0);
    if (received <= 0) {
      if (received == 0) {
        Logger::log("Connection closed by peer");
      } else {
        Logger::log("Recv failed");
      }
      return false;
    }

    totalReceived += received;
  }

  return true;
}

bool TCPSocket::isConnected() const { return connected && socketFD != INVALID_SOCKET; }

void TCPSocket::close() {
  if (socketFD != INVALID_SOCKET) {
#ifdef _WIN32
    closesocket(socketFD);
#else
    ::close(socketFD);
#endif
    socketFD = INVALID_SOCKET;
  }
  connected = false;
}

bool NCCLIdBroadcaster::broadcastNCCLId(ncclUniqueId& ncclId, int nodeRank, int worldSize, const std::string& masterIP,
                                        int port) {
  if (worldSize <= 0 || nodeRank < 0 || nodeRank >= worldSize) {
    Logger::log("Invalid rank or world size");
    return false;
  }

  if (worldSize == 1) {
    return true;
  }

  if (nodeRank == 0) {
    TCPSocket serverSocket;
    if (!serverSocket.listen(port)) {
      Logger::log("Master failed to listen on port", port);
      return false;
    }

    std::vector<std::unique_ptr<TCPSocket>> clientSockets;

    for (int i = 1; i < worldSize; i++) {
      auto clientSocket = serverSocket.accept();
      if (!clientSocket) {
        Logger::log("Failed to accept connection from worker", i);
        return false;
      }

      if (!clientSocket->send(&ncclId, sizeof(ncclUniqueId))) {
        Logger::log("Failed to send NCCL ID to worker", i);
        return false;
      }

      clientSockets.push_back(std::move(clientSocket));
    }

    Logger::log("Master successfully broadcast NCCL ID to all workers");
  } else {
    TCPSocket clientSocket;
    if (!clientSocket.connect(masterIP, port)) {
      Logger::log("Worker", nodeRank, "failed to connect to master at", masterIP, ":", port);
      return false;
    }

    if (!clientSocket.recv(&ncclId, sizeof(ncclUniqueId))) {
      Logger::log("Worker", nodeRank, "failed to receive NCCL ID from master");
      return false;
    }

    Logger::log("Worker", nodeRank, "successfully received NCCL ID from master");
  }

  return true;
}