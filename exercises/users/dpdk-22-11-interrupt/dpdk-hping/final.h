
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
#include <signal.h>

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#define TABLESIZE 400

#define S_SENT 0
#define S_RECV 1

#define uint32_t_to_char(ip, a, b, c, d) \
  do                                     \
  {                                      \
    *a = (uint8_t)(ip >> 24 & 0xff);     \
    *b = (uint8_t)(ip >> 16 & 0xff);     \
    *c = (uint8_t)(ip >> 8 & 0xff);      \
    *d = (uint8_t)(ip & 0xff);           \
  } while (0)

static uint16_t l2_len = 0, l3_len = 0;

static uint32_t client_ip_addr = RTE_IPV4(172, 16, 166, 131);
static uint32_t server_ip_addr = RTE_IPV4(172, 16, 166, 132);

uint16_t total_packets = 1;
uint16_t packets_sent = 0;
uint16_t packets_recv = 0;

/* server mode */
static bool server_mode = false;
bool correct_packet = true;

bool is_not_limited = true;
bool opt_icmp_mode = true;
bool opt_only_eth = false;
bool is_timed_out = false;

uint16_t src_id = -1;
uint16_t seq_nb = 0;

uint64_t data_size = 64;
unsigned int time_out_value = 2;

int delaytable_index = 0;

struct delaytable_element
{
  int seq;
  int src;
  time_t sec;
  time_t usec;
  double tsc_value;
  int status;
};

float
    rtt_min = 0,
    rtt_max = 0,
    rtt_avg = 0,
    rtt_min_tsc = 0,
    rtt_max_tsc = 0,
    rtt_avg_tsc = 0
    ;

int avg_counter = 0;

volatile struct delaytable_element delaytable[TABLESIZE];

time_t get_usec(void)
{
  struct timeval tmptv;

  gettimeofday(&tmptv, NULL);
  return tmptv.tv_usec;
}

static inline uint16_t
ck_sum(const unaligned_uint16_t *header, int header_len)
{
  uint32_t sum = 0;

  while (header_len > 1)
  {
    sum += *header++;
    if (sum & 0x80000000)
      sum = (sum & 0xFFFF) + (sum >> 16);
    header_len -= 2;
  }

  while (sum >> 16)
    sum = (sum & 0xFFFF) + (sum >> 16);

  return ~sum;
}

void get_minavgmax(float delay_ms)
{
  if (rtt_min == 0 || delay_ms < rtt_min)
    rtt_min = delay_ms;
  if (rtt_max == 0 || delay_ms > rtt_max)
    rtt_max = delay_ms;
  rtt_avg = (rtt_avg * (avg_counter - 1) / avg_counter) + (delay_ms / avg_counter);
}

void get_minavgmax_tsc(float delay_ms)
{
  if (rtt_min_tsc == 0 || delay_ms < rtt_min_tsc)
    rtt_min_tsc = delay_ms;
  if (rtt_max_tsc == 0 || delay_ms > rtt_max_tsc)
    rtt_max_tsc = delay_ms;

  rtt_avg_tsc = (rtt_avg_tsc * (avg_counter - 1) / avg_counter) + (delay_ms / avg_counter);
}

int rtt(int *seqp, int recvport, float *delay_ms, double *rtt_tsc, double *tsc_value)
{
  long delay_sec = 0, delay_usec = 0;
  int i, tablepos = -1, status;

  const uint64_t tsc_hz = rte_get_tsc_hz();

  if (*seqp != 0)
  {
    for (i = 0; i < TABLESIZE; i++)
      if (delaytable[i].seq == *seqp)
      {
        tablepos = i;
        break;
      }
  }
  else
  {
    for (i = 0; i < TABLESIZE; i++)
      if (delaytable[i].src == recvport)
      {
        tablepos = i;
        break;
      }
    if (i != TABLESIZE)
      *seqp = delaytable[i].seq;
  }

  if (tablepos != -1)
  {
    status = delaytable[tablepos].status;
    delaytable[tablepos].status = S_RECV;

    delay_sec = time(NULL) - delaytable[tablepos].sec;
    delay_usec = get_usec() - delaytable[tablepos].usec;
    if (delay_sec == 0 && delay_usec < 0)
      delay_usec += 1000000;

    *delay_ms = (delay_sec * 1000) + ((float)delay_usec / 1000);

    avg_counter++;
    get_minavgmax(*delay_ms);

    double diff_tsc = *tsc_value - delaytable[i].tsc_value;

    *rtt_tsc = diff_tsc * US_PER_S / tsc_hz;

    get_minavgmax_tsc(*rtt_tsc);
  }
  else
  {
    *delay_ms = 0; /* not in table.. */
    status = 0;    /* we don't know if it's DUP */
  }

  /* SANITY CHECK */
  if (*delay_ms < 0)
  {
    printf("Error in Rtt calculation\n");
  }

  return status;
}

void delaytable_add(int seq, int src, time_t sec, time_t usec, int status)
{
  delaytable[delaytable_index % TABLESIZE].seq = seq;
  delaytable[delaytable_index % TABLESIZE].src = src;
  delaytable[delaytable_index % TABLESIZE].sec = sec;
  delaytable[delaytable_index % TABLESIZE].usec = usec;
  delaytable[delaytable_index % TABLESIZE].tsc_value = rte_rdtsc();
  delaytable[delaytable_index % TABLESIZE].status = status;
  delaytable_index++;
}

void parse_client(struct rte_mbuf *pkt)
{
  struct rte_ether_hdr *eth_hdr;
  struct rte_vlan_hdr *vlan_hdr;
  struct rte_ipv4_hdr *ip_hdr;
  struct rte_udp_hdr *udp_hdr;
  struct rte_tcp_hdr *tcp_hdr;
  struct rte_icmp_hdr *icmp_hdr;
  uint16_t eth_type, offset = 0;
  uint16_t next_proto;

  float ms_delay;

  double tsc_value = rte_rdtsc();

  double rtt_tsc = 0;

  eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
  eth_type = rte_cpu_to_be_16(eth_hdr->ether_type);

  if (eth_type == RTE_ETHER_TYPE_VLAN)
  {
    vlan_hdr = (struct rte_vlan_hdr *)((unsigned char *)(eth_hdr + 1));
    eth_type = rte_cpu_to_be_16(vlan_hdr->eth_proto);
    offset += sizeof(struct rte_vlan_hdr);
  }
  if (eth_type == RTE_ETHER_TYPE_IPV4)
  {
    ip_hdr = (struct rte_ipv4_hdr *)((unsigned char *)(eth_hdr + 1) + offset);
    // extract ip features
    // IP Check

    if (client_ip_addr != rte_be_to_cpu_32(ip_hdr->dst_addr))
    {
      correct_packet = false;
      return;
    }

    next_proto = ip_hdr->next_proto_id;

    switch (next_proto)
    {
    // case IPPROTO_UDP:
    // {
    //   udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
    //   // extract the info required port info, etc
    // }
    // case IPPROTO_TCP:
    // {
    //   tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);
    //   // extract the info required port info, etc
    // }
    case IPPROTO_ICMP:
    {
      icmp_hdr = (struct rte_icmp_hdr *)(ip_hdr + 1);
      int seq_num = rte_be_to_cpu_16(icmp_hdr->icmp_seq_nb);

      if (icmp_hdr->icmp_type == RTE_IP_ICMP_ECHO_REPLY)
      {
        icmp_hdr->icmp_type = RTE_IP_ICMP_ECHO_REQUEST;
        icmp_hdr->icmp_code = 0;
        icmp_hdr->icmp_cksum = rte_cpu_to_be_16(ck_sum((unaligned_uint16_t *)(icmp_hdr), sizeof(struct rte_icmp_hdr))); //

        int status = rtt(&seq_num, 0, &ms_delay, &rtt_tsc, &tsc_value);
      }

      char a, b, c, d;
      uint32_t_to_char(rte_bswap32(ip_hdr->src_addr), &a, &b, &c, &d);

      printf("len=%d ip=%3hhu.%3hhu.%3hhu.%3hhu ttl=%d id=%d icmp_seq=%u/%u rtt=%.1f ms rtt_tsc=%.2f us\n", data_size, a, b, c, d, (unsigned int)ip_hdr->time_to_live, (unsigned int)ip_hdr->packet_id, (unsigned int)seq_num, (unsigned int)rte_cpu_to_be_16(seq_num), ms_delay, rtt_tsc);

      seq_nb++;
      icmp_hdr->icmp_seq_nb = rte_cpu_to_be_16(seq_nb);
    }
    break;
    }
    rte_be32_t temp_ip;

    temp_ip = ip_hdr->dst_addr;
    ip_hdr->dst_addr = ip_hdr->src_addr;
    ip_hdr->src_addr = temp_ip;
  }
}

void parser_server(struct rte_mbuf *pkt)
{
  struct rte_ether_hdr *eth_hdr;
  struct rte_vlan_hdr *vlan_hdr;
  struct rte_ipv4_hdr *ip_hdr;
  struct rte_udp_hdr *udp_hdr;
  struct rte_tcp_hdr *tcp_hdr;
  struct rte_icmp_hdr *icmp_hdr;
  uint16_t eth_type, offset = 0;
  uint16_t next_proto;

  eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
  eth_type = eth_hdr->ether_type;

  if (eth_type == RTE_ETHER_TYPE_VLAN)
  {
    vlan_hdr = (struct rte_vlan_hdr *)((unsigned char *)(eth_hdr)); // eth + 1?
    eth_type = rte_cpu_to_be_16(vlan_hdr->eth_proto);
    offset += sizeof(struct rte_vlan_hdr);
  }
  if (eth_hdr->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4))
  {
    ip_hdr = (struct rte_ipv4_hdr *)((unsigned char *)(eth_hdr + 1) + offset); // eth + 1?

    char a, b, c, d;
    uint32_t_to_char(rte_bswap32(ip_hdr->src_addr), &a, &b, &c, &d);

    // printf("Packet received with ip: %u\n", );

    rte_be32_t temp_ip;
    temp_ip = ip_hdr->dst_addr;
    ip_hdr->dst_addr = ip_hdr->src_addr;
    ip_hdr->src_addr = temp_ip;

    next_proto = ip_hdr->next_proto_id;

    printf("%u\n", (unsigned int)next_proto);

    switch (next_proto)
    {
      // // case IPPROTO_UDP:
      // // {
      // //   udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
      // //   // extract the info required port info, etc
      // // }
      // // case IPPROTO_TCP:
      // // {
      // //   tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);
      // //   // extract the info required port info, etc
      // // }
    case IPPROTO_ICMP:
    {
      icmp_hdr = (struct rte_icmp_hdr *)(ip_hdr + 1);

      if (icmp_hdr->icmp_type == RTE_IP_ICMP_ECHO_REQUEST)
      {
        icmp_hdr->icmp_type = RTE_IP_ICMP_ECHO_REPLY;
        icmp_hdr->icmp_code = 0;
        icmp_hdr->icmp_cksum = rte_cpu_to_be_16(ck_sum((unaligned_uint16_t *)(icmp_hdr), sizeof(*icmp_hdr)));
      }
      int icmp_seq = rte_be_to_cpu_16(icmp_hdr->icmp_seq_nb);
      printf("with Icmp seq: %d\n", icmp_seq);
      break;
    }
    }
  }
}

uint16_t add_icmp(struct rte_mbuf *pkt)
{
  struct rte_icmp_hdr *icmp_hdr;

  icmp_hdr = (struct rte_icmp_hdr *)((unsigned char *)pkt + (l2_len + l3_len));

  // Consruct ICMP Request
  icmp_hdr->icmp_type = RTE_IP_ICMP_ECHO_REQUEST;
  icmp_hdr->icmp_code = 0;

  icmp_hdr->icmp_cksum = rte_cpu_to_be_16(ck_sum((unaligned_uint16_t *)(icmp_hdr), sizeof(*icmp_hdr)));
  icmp_hdr->icmp_ident = rte_cpu_to_be_16((uint16_t)getpid());
  icmp_hdr->icmp_seq_nb = rte_cpu_to_be_16(seq_nb);

  return (uint16_t)(sizeof(struct rte_icmp_hdr) + data_size);
}

#define IP_DEFTTL 64

uint32_t add_ip(struct rte_mbuf *pkt, uint32_t client_ip, uint32_t server_ip)
{
  struct rte_ipv4_hdr *ip_hdr;

  ip_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
  ip_hdr->version_ihl = RTE_IPV4_VHL_DEF;
  ip_hdr->type_of_service = 0;
  ip_hdr->fragment_offset = 0;
  ip_hdr->time_to_live = IP_DEFTTL;     // need to check for macro definition
  ip_hdr->next_proto_id = IPPROTO_ICMP; /// based on next proto
  ip_hdr->src_addr = rte_cpu_to_be_32(client_ip);
  ip_hdr->dst_addr = rte_cpu_to_be_32(server_ip);

  ip_hdr->packet_id = (src_id == -1) ? rte_cpu_to_be_32((unsigned short)rand()) : htons((unsigned short)src_id);

  uint16_t ip_packet_size = (uint16_t)data_size + sizeof(struct rte_ipv4_hdr);

  l3_len = sizeof(struct rte_ipv4_hdr);

  if (opt_icmp_mode)
  {
    ip_packet_size += add_icmp(pkt);
  }
  // else if (opt_tcp_mode)
  // {
  //   ip_packet_size += sizeof(struct rte_tcp_hdr);
  //   ip_hdr->total_length = rte_cpu_to_be_16(ip_packet_size);
  //   add_tcp();
  // }
  // else if (opt_udp_mode)
  // {
  //   ip_packet_size += sizeof(struct rte_udp_hdr);
  //   ip_hdr->total_length = rte_cpu_to_be_16(ip_packet_size);
  //   add_udp();
  // }
  // ip_hdr->hdr_checksum = ip_sum((unaligned_uint16_t *)ip_hdr,sizeof(*ipv4_hdr)); possible that this function might be correct

  ip_hdr->total_length = rte_cpu_to_be_16(ip_packet_size);
  ip_hdr->hdr_checksum = rte_ipv4_cksum(ip_hdr);
  return ip_packet_size;
}

void timeout_handler(int signal_id)
{
  is_timed_out = true;
}

void update_seq_nb(struct rte_mbuf *pkt)
{
  struct rte_icmp_hdr *icmp_hdr;
  icmp_hdr = (struct rte_icmp_hdr *)((unsigned char *)pkt + (l2_len + l3_len));
  seq_nb++;
  icmp_hdr->icmp_seq_nb = rte_cpu_to_be_16(seq_nb);
}

void print_statistics(int signal_id)
{
  unsigned int lossrate;

  if (packets_recv > 0)
    lossrate = 100 - ((packets_recv * 100) / packets_sent);
  else if (!packets_sent)
    lossrate = 0;
  else
    lossrate = 100;

  fprintf(stderr, "\n---- dpdk-hping statistic ----\n");

  fprintf(stderr, "%d packets tramitted, %d packets received, "
                  "%d%% packet loss\n",
          packets_sent, packets_recv, lossrate);

  fprintf(stderr, "round-trip min/avg/max = %.2f/%.2f/%.2f ms\n",
          rtt_min, rtt_avg, rtt_max);

  fprintf(stderr, "round-trip using tsc register min/avg/max = %.2f/%.2f/%.2f us\n",
          rtt_min_tsc, rtt_avg_tsc, rtt_max_tsc);

  /* manage exit code */

  if (packets_recv)
    exit(0);
  else
    exit(1);
};

/* Portable signal() from R.Stevens,
 * modified to reset the handler */
void (*Signal(int signo, void (*func)(int)))(int)
{
  struct sigaction act, oact;

  act.sa_handler = func;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0; /* So if set SA_RESETHAND is cleared */
  if (signo == SIGALRM)
  {
#ifdef SA_INTERRUPT
    act.sa_flags |= SA_INTERRUPT; /* SunOS 4.x */
#endif
  }
  else
  {
#ifdef SA_RESTART
    act.sa_flags |= SA_RESTART; /* SVR4, 4.4BSD, Linux */
#endif
  }
  if (sigaction(signo, &act, &oact) == -1)
    return SIG_ERR;
  return (oact.sa_handler);
}
