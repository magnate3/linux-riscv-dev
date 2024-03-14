

# CONFIG_MACB_USE_HWSTAMP

```
root@ubuntu:~/linux-5.15.24# grep CONFIG_MACB_USE_HWSTAMP .config
CONFIG_MACB_USE_HWSTAMP=y
```


## enable  CONFIG_MACB_USE_HWSTAMP

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/ptpc.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/1588.png)


# hwtstamp_config

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/hwtstamp_config.png)


# module

```
drivers/net/ethernet/cadence/macb.ko
drivers/pps/pps_core.ko
drivers/ptp/ptp.ko
```

# gem_ptp_do_XXstamp

```
static inline int gem_ptp_do_txstamp(struct macb_queue *queue, struct sk_buff *skb, struct macb_dma_desc *desc)
{
        if (queue->bp->tstamp_config.tx_type == TSTAMP_DISABLED)
                return -ENOTSUPP;

        return gem_ptp_txstamp(queue, skb, desc);
}

static inline void gem_ptp_do_rxstamp(struct macb *bp, struct sk_buff *skb, struct macb_dma_desc *desc)
{
        if (bp->tstamp_config.rx_filter == TSTAMP_DISABLED)
                return;

        gem_ptp_rxstamp(bp, skb, desc);
}
int gem_get_hwtst(struct net_device *dev, struct ifreq *rq);
int gem_set_hwtst(struct net_device *dev, struct ifreq *ifr, int cmd);


```
## gem_tx_timestamp_flush
```
gem_tx_timestamp_flush

static void gem_tx_timestamp_flush(struct work_struct *work)
{
        struct macb_queue *queue =
                        container_of(work, struct macb_queue, tx_ts_task);
        unsigned long head, tail;
        struct gem_tx_ts *tx_ts;

        /* take current head */
        head = smp_load_acquire(&queue->tx_ts_head);
        tail = queue->tx_ts_tail;

        while (CIRC_CNT(head, tail, PTP_TS_BUFFER_SIZE)) {
                tx_ts = &queue->tx_timestamps[tail];
                gem_tstamp_tx(queue->bp, tx_ts->skb, &tx_ts->desc_ptp);
                /* cleanup */
                dev_kfree_skb_any(tx_ts->skb);
                /* remove old tail */
                smp_store_release(&queue->tx_ts_tail,
                                  (tail + 1) & (PTP_TS_BUFFER_SIZE - 1));
                tail = queue->tx_ts_tail;
        }
}


static void gem_tstamp_tx(struct macb *bp, struct sk_buff *skb,
                          struct macb_dma_desc_ptp *desc_ptp)
{
        struct skb_shared_hwtstamps shhwtstamps;
        struct timespec64 ts;

        gem_hw_timestamp(bp, desc_ptp->ts_1, desc_ptp->ts_2, &ts);
        memset(&shhwtstamps, 0, sizeof(shhwtstamps));
        shhwtstamps.hwtstamp = ktime_set(ts.tv_sec, ts.tv_nsec);
        skb_tstamp_tx(skb, &shhwtstamps);
}
```

# increasing tx_timestamp_timeout may correct this issue, but it is likely caused by a driver bug

```
tx_timestamp_timeout 20000
sk_tx_timeout = config_get_int(cfg, NULL, "tx_timestamp_timeout");
res = poll(&pfd, 1, sk_tx_timeout);
```

```
#ifdef CONFIG_MACB_USE_HWSTAMP
static unsigned int gem_get_tsu_rate(struct macb *bp)
{
	struct clk *tsu_clk;
	unsigned int tsu_rate;

	tsu_clk = devm_clk_get(&bp->pdev->dev, "tsu_clk");
	if (!IS_ERR(tsu_clk))
		tsu_rate = clk_get_rate(tsu_clk);
	/* try pclk instead */
	else if (!IS_ERR(bp->pclk)) {
		tsu_clk = bp->pclk;
		tsu_rate = clk_get_rate(tsu_clk);
	} else
		return -ENOTSUPP;
	return tsu_rate;
}
……

#endif
```
##  make CROSS_COMPILE=riscv64-linux-gnu-
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/make.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/state.png)




![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/module.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/macb.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/hw_ptp.png)