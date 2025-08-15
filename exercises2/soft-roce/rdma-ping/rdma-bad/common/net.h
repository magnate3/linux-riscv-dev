#pragma once

#define _GNU_SOURCE

#include "common.h"
#include "utils.h"

#include <arpa/inet.h>
#include <errno.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/types.h>
#include <net/if.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

int setup_socket (void);

/**
 * Convert a string IP address to a uint32_t.
 * @param ip the string IP address
 * @param ip_addr the output buffer to write the IP address to
 * @return 0 on success, -1 on failure
 */
int convert_ip (const char *ip, uint32_t *ip_addr);

/**
 * Construct the base structure of the packet inside buf by writing the ethernet and ip headers
 * to the given buffer.
 *
 * @param buf the buffer to write the headers to
 * @param src_mac the source mac address
 * @param dest_mac the destination mac address
 * @param src_ip the source ip address
 * @param dest_ip the destination ip address
 * @return 0 on success, -1 on failure
 */
int build_base_packet (char *buf, const uint8_t *src_mac, const uint8_t *dest_mac,
                       const uint32_t src_ip, const uint32_t dest_ip);

/**
 * Retrieve the payload from the given buffer.
 * This function assumes that the buffer contains a valid pingpong packet.
 * The returned pointer points to the payload inside the buffer.
 *
 * @param buf the buffer containing the packet
 * @return a pointer to the payload inside the buffer
 */
struct pingpong_payload *packet_payload (const char *buf);

/**
 * Build a sockaddr_ll structure with the given ifindex and destination mac address.
 * This structure is used to send packets to the remote node.
 *
 * @param ifindex the interface index
 * @param dest_mac the destination mac address
 * @return the constructed sockaddr_ll structure
 */
struct sockaddr_ll build_sockaddr (int ifindex, const unsigned char *dest_mac);

/**
 * Send a pingpong packet to the remote node.
 *
 * @param sock the socket to use to send the packet
 * @param buf the buffer to be sent
 * @param ifindex the interface index to send the packet from
 * @param dest_mac the destination mac address
 * @return 0 on success, -1 on failure
 */
int send_pingpong_packet (int sock, const char *buf, struct sockaddr_ll *sock_addr);

typedef int (*send_packet_t) (char *, uint64_t, struct sockaddr_ll *, void *);

/**
 * Start a thread to send the packets every `interval` microseconds.
 * The thread will send `iters` packets and then exit.
 *
 * @param iters the number of packets to send
 * @param interval the interval between packets in microseconds
 * @param base_packet the base packet to send
 * @param sock_addr the sockaddr_ll structure to use to send the packets
 * @param send_packet the function to use to send the packets, called by the sending thread.
 * @return 0 on success, -1 on failure
 */
int start_sending_packets (uint64_t iters, uint64_t interval, char *base_packet, struct sockaddr_ll *sock_addr, send_packet_t send_packet, void *aux);

#if !SERVER
pthread_t get_sender_thread (void);
#endif

/**
 * Retrieve the local interface MAC address.
 *
 * @param ifindex the index of the interface
 * @param out_mac the output buffer to write the MAC address to
 * @return 0 on success, -1 on failure
 */
int retrieve_local_mac (int ifindex, uint8_t *out_mac);

/**
 * Retrieve the local interface IP address.
 *
 * @param ifindex the index of the interface
 * @param out_addr the output buffer to write the IP address to
 * @return 0 on success, -1 on failure
 */
int retrieve_local_ip (int ifindex, uint32_t *out_addr);

/**
 * Make the client and server exchange IP and MAC addresses.
 * The client sends a UDP message containing its own addresses; the server replies with its own addresses.
 * Doing so avoids any kind of hard-coded addresses or configuration.
 *
 * The src_* and dest_* variables are meant to point to empty variables: src_* will be filled with automatically
 * gathered information about the local machine, while dest_* will be filled with the information received from the
 * remote machine.
 *
 * @param ifindex the index of the interface
 * @param server_ip the IP address of the server, required to send the UDP packet
 * @param is_server whether this node is the server or not
 * @param src_mac buffer to write the source MAC address to
 * @param dest_mac buffer to write the destination MAC address to
 * @param src_ip buffer to write the source IP address to
 * @param dest_ip buffer to write the destination IP address to
 * @return 0 on success, -1 on failure
 */
int exchange_eth_ip_addresses (const int ifindex, const char *server_ip, bool is_server,
                               uint8_t *src_mac, uint8_t *dest_mac,
                               uint32_t *src_ip, uint32_t *dest_ip);

/**
 * Generic function to exchange any kind of information between two machines in a reciprocal way.
 * The `buffer` variable contains the data that the local machine will send to the remote machine.
 * The `out_buffer` variable will be filled with the data received from the remote machine.
 * Both the buffers must be greater of equal in size to `packet_size`.
 *
 * If the local machine is the client, it will send its own information to the server and then wait for the server's
 * response. If the local machine is the server, it will wait for the client's message and then reply with its own
 * information.
 *
 * @param server_ip the IP address of the server
 * @param is_server whether this node is the server or not
 * @param packet_size the size of the packet to send
 * @param buffer the buffer to send. Must be at least `packet_size` bytes long
 * @param out_buffer the buffer to write the received data to. Must be at least `packet_size` bytes long
 * @return
 */
int exchange_data (const char *server_ip, bool is_server, uint32_t packet_size, uint8_t *buffer, uint8_t *out_buffer);
