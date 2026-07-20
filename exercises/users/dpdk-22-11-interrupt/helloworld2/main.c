#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/queue.h>
#include <unistd.h>

#include <rte_debug.h>
#include <rte_byteorder.h>
#include <rte_common.h>
#include <rte_debug.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_launch.h>
#include <rte_lcore.h>
#include <rte_log.h>
#include <rte_mbuf.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_memzone.h>
#include <rte_per_lcore.h>
#include <rte_vect.h>

#define MEMPOOL_CACHE_SIZE 256
#define MAX_PKT_BURST 32
#define NUM_RX_Q 1
#define NUM_TX_Q 1

static struct rte_mempool *pktmbuf_pool;

#if 0
static struct rte_eth_conf port_conf = {
    .rxmode =
        {
            .mq_mode = RTE_ETH_MQ_RX_NONE,
            .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
            .split_hdr_size = 0,
            .header_split = 0,   /**< Header Split disabled */
            .hw_ip_checksum = 0, /**< IP checksum offload enabled */
            .hw_vlan_filter = 0,
            .hw_vlan_extend = 0,
            .hw_strip_crc = 0, /**< CRC stripped by hardware */
        },
    .rx_adv_conf =
        {
            .rss_conf =
                {
                    .rss_key = NULL,
                    //.rss_hf = ETH_RSS_IP,
                },
        },
    .txmode =
        {
            .mq_mode = ETH_MQ_TX_NONE,
        },
};
#else
static struct rte_eth_conf port_conf = {
        .rxmode = {
                .mq_mode        = RTE_ETH_MQ_RX_RSS,
                .offloads = RTE_ETH_RX_OFFLOAD_CHECKSUM,
        },
        //.rx_adv_conf = {
        //        .rss_conf = {
        //                .rss_key = NULL,
        //                .rss_hf = RTE_ETH_RSS_UDP,
        //        },
        //},
        .txmode = {
                .mq_mode = RTE_ETH_MQ_TX_NONE,
        }
};
#endif
static void hexdump(uint8_t buffer[], int len) {
#define HEXDUMP_LINE_LEN 16
  int i;
  char s[HEXDUMP_LINE_LEN + 1];
  bzero(s, HEXDUMP_LINE_LEN + 1);

  for (i = 0; i < len; i++) {
    if (!(i % HEXDUMP_LINE_LEN)) {
      if (s[0])
        printf("[%s]", s);
      printf("\n%05x: ", i);
      bzero(s, HEXDUMP_LINE_LEN);
    }
    s[i % HEXDUMP_LINE_LEN] = isprint(buffer[i]) ? buffer[i] : '.';
    printf("%02x ", buffer[i]);
  }
  while (i++ % HEXDUMP_LINE_LEN)
    printf("   ");

  printf("[%s]\n", s);
}

static void dump_rte_muf(struct rte_mbuf *mb) {
#define _(n)                                                                   \
  printf("%-18s %-8lu [0x%08lx]\n", #n ":", (long unsigned int)mb->n,          \
         (long unsigned int)mb->n)

  _(data_off);
  _(data_len);
  _(pkt_len);
  _(buf_len);
  _(nb_segs);
  _(ol_flags);
  _(packet_type);
  _(vlan_tci);
  _(vlan_tci_outer);
  _(hash.rss);
#undef _
  hexdump((uint8_t *)mb->buf_addr + mb->data_off, mb->data_len);
}

static void burst(uint8_t p, uint8_t q) {
  struct rte_mbuf *pkts[MAX_PKT_BURST];
  int nb_rx = 0;
  int i;
  nb_rx = rte_eth_rx_burst(p, q, pkts, MAX_PKT_BURST);
  if (!nb_rx)
    return;
  for (i = 0; i < nb_rx; i++) {
    printf("frame: %d/%d on port %u queue %u\n", i + 1, nb_rx, p, q);
    dump_rte_muf(pkts[i]);
    printf("\n==========================================================="
           "==============\n");
    rte_pktmbuf_free(pkts[i]);
  }
}

static int lcore_main(__attribute__((unused)) void *arg) {
  int p, q;
  unsigned lcore_id;

  lcore_id = rte_lcore_id();
  printf("Running on core %u\n", lcore_id);

  while (1) {
    if (port_conf.intr_conf.rxq) {
      int i, n;
      const int num = NUM_RX_Q;
      struct rte_epoll_event event[num];
      n = rte_epoll_wait(RTE_EPOLL_PER_THREAD, event, num, -1);
      printf("lcore %u is waked up from rx interrupt \n",lcore_id);
      for (i = 0; i < n; i++) {
        void *data = event[i].epdata.data;
        p = ((uintptr_t)data) >> CHAR_BIT;
        q = ((uintptr_t)data) & RTE_LEN2MASK(CHAR_BIT, uint8_t);
        printf("Interrupt received for port %u queue %u\n", p, q);
        rte_eth_dev_rx_intr_disable(p, q);
        burst(p, q);
        rte_eth_dev_rx_intr_enable(p, q);
      }
    } else {
      for (p = 0; p <rte_eth_dev_count_avail(); p++)
        for (q = 0; q < NUM_RX_Q; q++)
          burst(p, q);
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  int ret;
  int socketid = 0;
  struct rte_eth_txconf *txconf;
  struct rte_eth_dev_info dev_info;
  int p, q, opt;

  ret = rte_eal_init(argc, argv);
  if (ret < 0)
    rte_exit(EXIT_FAILURE, "Cannot init EAL\n");

  argc -= ret;
  argv += ret;

  while ((opt = getopt(argc, argv, "ij")) > 0) {
    switch (opt) {
    case 'i':
      port_conf.intr_conf.rxq = 1;
      break;
    case 'j':
      //port_conf.rxmode.jumbo_frame = 1;
      //port_conf.rxmode.enable_scatter = 1;
      //port_conf.rxmode.max_rx_pkt_len = 9216;
      break;
    default:
      exit(1);
    }
  }

  if (!rte_eth_dev_count_avail())
    rte_exit(EXIT_FAILURE, "We need at least one eth device\n");

  pktmbuf_pool = rte_pktmbuf_pool_create("mbuf_pool0", 4096, MEMPOOL_CACHE_SIZE,
                                         0, RTE_MBUF_DEFAULT_BUF_SIZE, 0);

  if (pktmbuf_pool == NULL)
    rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");

  for (p = 0; p <rte_eth_dev_count_avail(); p++) {
    ret = rte_eth_dev_configure(p, NUM_RX_Q, NUM_TX_Q, &port_conf);
    if (ret < 0)
      rte_exit(EXIT_FAILURE, "rte_eth_dev_configure: err=%d, port=%d\n", ret,
               p);

    /* Setup TX queue */
    rte_eth_dev_info_get(p, &dev_info);
    txconf = &dev_info.default_txconf;
    for (q = 0; q < NUM_TX_Q; q++) {
      ret = rte_eth_tx_queue_setup(p, 0, 512, socketid, txconf);
      if (ret < 0)
        rte_exit(EXIT_FAILURE,
                 "rte_eth_tx_queue_setup: err=%d, port=%d queue %d\n", ret, p,
                 q);
    }

    for (q = 0; q < NUM_RX_Q; q++) {
      ret = rte_eth_rx_queue_setup(p, q, 512, socketid, NULL, pktmbuf_pool);
      if (ret < 0)
        rte_exit(EXIT_FAILURE,
                 "rte_eth_rx_queue_setup: err=%d, port=%d queue=%d\n", ret, p,
                 q);
    }

    rte_eth_promiscuous_enable(p);

    ret = rte_eth_dev_start(p);
    if (ret < 0)
      rte_exit(EXIT_FAILURE, "rte_eth_dev_start: err=%d, port=%d\n", ret, p);

    if (port_conf.intr_conf.rxq)
      for (q = 0; q < NUM_RX_Q; q++) {
        uint32_t data = p << CHAR_BIT | q;
        ret = rte_eth_dev_rx_intr_ctl_q(p, q, RTE_EPOLL_PER_THREAD,
                                        RTE_INTR_EVENT_ADD,
                                        (void *)((uintptr_t)data));
        if (ret < 0)
          rte_exit(EXIT_FAILURE,
                   "rte_eth_dev_rx_intr_ctl_q: err=%d, port=%d queue=%d\n", ret,
                   p, q);
        rte_eth_dev_rx_intr_enable(p, q);
      }
  }

#if 0
	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		rte_eal_remote_launch(lcore_main, NULL, lcore_id);
	}
#endif
  lcore_main(NULL);

  rte_eal_mp_wait_lcore();
  return 0;
}
