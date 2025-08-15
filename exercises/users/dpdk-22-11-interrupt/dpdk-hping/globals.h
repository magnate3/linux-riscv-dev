
#include <rte_byteorder.h>
#include <rte_log.h>
#include <rte_common.h>
#include <rte_config.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_ether.h>
#include <rte_udp.h>
#include <rte_icmp.h>
#include <rte_tcp.h>

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define TABLESIZE 400

#define S_SENT 0
#define S_RECV 1

static uint16_t l2_len = 0, l3_len = 0;

uint16_t src_id = -1;
uint16_t seq_nb = 0;

uint64_t data_size = 64;

bool opt_icmp_mode = true;
bool opt_only_eth = false;

int delaytable_index = 0;

struct delaytable_element
{
  int seq;
  int src;
  time_t sec;
  time_t usec;
  int status;
};

volatile struct delaytable_element delaytable[TABLESIZE];

time_t get_usec(void)
{
  struct timeval tmptv;

  gettimeofday(&tmptv, NULL);
  return tmptv.tv_usec;
}

float
    rtt_min = 0,
    rtt_max = 0,
    rtt_avg = 0;

int avg_counter = 0;

static inline uint16_t
ck_sum(const unaligned_uint16_t *hdr, int hdr_len)
{
  uint32_t sum = 0;

  while (hdr_len > 1)
  {
    sum += *hdr++;
    if (sum & 0x80000000)
      sum = (sum & 0xFFFF) + (sum >> 16);
    hdr_len -= 2;
  }

  while (sum >> 16)
    sum = (sum & 0xFFFF) + (sum >> 16);

  return ~sum;
}