

# menuconfig 

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/cadence/menuconfig.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/macb/cadence/phylink.png)

# output   rxstamp/txstamp  log

**gem_rx->gem_ptp_do_rxstamp-> gem_ptp_rxstamp **

```

void gem_ptp_rxstamp(struct macb *bp, struct sk_buff *skb,
		     struct macb_dma_desc *desc)
{
	struct skb_shared_hwtstamps *shhwtstamps = skb_hwtstamps(skb);
	struct macb_dma_desc_ptp *desc_ptp;
	struct timespec64 ts;

	if (GEM_BFEXT(DMA_RXVALID, desc->addr)) {
		desc_ptp = macb_ptp_desc(bp, desc);
		/* Unlikely but check */
		if (!desc_ptp) {
			dev_warn_ratelimited(&bp->pdev->dev,
					     "Timestamp not supported in BD\n");
			return;
		}
		gem_hw_timestamp(bp, desc_ptp->ts_1, desc_ptp->ts_2, &ts);
		memset(shhwtstamps, 0, sizeof(struct skb_shared_hwtstamps));
		shhwtstamps->hwtstamp = ktime_set(ts.tv_sec, ts.tv_nsec);
	}
#if  TEST_PTP
	
	ktime_t t =  shhwtstamps->hwtstamp;
	printk("%s()  gem_ptp_rxstamp ktime_get=%llu\n", __func__, ktime_to_ns(t));
#endif
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
#if TEST_PTP

        ktime_t t =  shhwtstamps.hwtstamp;
        printk("%s()  gem_ptp_txstamp ktime_get=%llu\n", __func__, ktime_to_ns(t));
#endif
}
```