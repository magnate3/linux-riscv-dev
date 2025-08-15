

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.15.rt/drivers/net/ethernet/cadence/no_napi/noapi2.png)

# napi
##  napi_disable
```
#ifndef TEST_POLL_NO_USE_NAPI
	for (q = 0, queue = bp->queues; q < bp->num_queues; ++q, ++queue)
		napi_disable(&queue->napi);
#endif
```

## napi_enable
```
#ifndef TEST_POLL_NO_USE_NAPI
	for (q = 0, queue = bp->queues; q < bp->num_queues;
	     ++q, ++queue)
		napi_enable(&queue->napi);
#endif
```

##  napi_schedule

```
#ifdef TEST_POLL
		if (status & bp->rx_intr_mask) {
			/* There's no point taking any more interrupts
			 * until we have processed the buffers. The
			 * scheduling call may fail if the poll routine
			 * is already scheduled, so disable interrupts
			 * now.
			 */
			queue_writel(queue, IDR, bp->rx_intr_mask);
			if (bp->caps & MACB_CAPS_ISR_CLEAR_ON_WRITE)
				queue_writel(queue, ISR, MACB_BIT(RCOMP));

			//int budget = netdev_budget;//300
            		gem_rx(queue, NULL, 300);
		}
#else
		if (status & bp->rx_intr_mask) {
			/* There's no point taking any more interrupts
			 * until we have processed the buffers. The
			 * scheduling call may fail if the poll routine
			 * is already scheduled, so disable interrupts
			 * now.
			 */
			queue_writel(queue, IDR, bp->rx_intr_mask);
			if (bp->caps & MACB_CAPS_ISR_CLEAR_ON_WRITE)
				queue_writel(queue, ISR, MACB_BIT(RCOMP));

			if (napi_schedule_prep(&queue->napi)) {
				netdev_vdbg(bp->dev, "scheduling RX softirq\n");
				__napi_schedule(&queue->napi);
			}
		}
#endif
```

## netif_napi_add

```
#ifdef TEST_POLL_NO_USE_NAPI
	        dev_err(&pdev->dev, "not need to use napi \n");
			netif_napi_add(dev, &queue->napi, macb_poll, NAPI_POLL_WEIGHT);
#else 
		netif_napi_add(dev, &queue->napi, macb_poll, NAPI_POLL_WEIGHT);
#endif
```

# gem_rx(queue, &queue->napi, 300)

```
#ifdef TEST_POLL
                if (status & bp->rx_intr_mask) {
                        /* There's no point taking any more interrupts
                         * until we have processed the buffers. The
                         * scheduling call may fail if the poll routine
                         * is already scheduled, so disable interrupts
                         * now.
                         */
                        queue_writel(queue, IDR, bp->rx_intr_mask);
                        if (bp->caps & MACB_CAPS_ISR_CLEAR_ON_WRITE)
                                queue_writel(queue, ISR, MACB_BIT(RCOMP));

                        //int budget = netdev_budget;//300
                        //gem_rx(queue, NULL, 300);
                        gem_rx(queue, &queue->napi, 300);
                }
#else
                if (status & bp->rx_intr_mask) {
                        /* There's no point taking any more interrupts
                         * until we have processed the buffers. The
                         * scheduling call may fail if the poll routine
                         * is already scheduled, so disable interrupts
                         * now.
                         */
                        queue_writel(queue, IDR, bp->rx_intr_mask);
                        if (bp->caps & MACB_CAPS_ISR_CLEAR_ON_WRITE)
                                queue_writel(queue, ISR, MACB_BIT(RCOMP));

                        if (napi_schedule_prep(&queue->napi)) {
                                netdev_vdbg(bp->dev, "scheduling RX softirq\n");
                                __napi_schedule(&queue->napi);
                        }
                }
#endif
```
 