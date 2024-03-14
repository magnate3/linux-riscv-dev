 # purpose
  disable  recv and send interrupt to achieve poll recv
  
 # make
 ```
 make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi-  UIMAGE_LOADADDR=0x8000 uImage -j20
 ```
 # run
 
 ```
root@x86:/home/ubuntu# tunctl  -t qtap -u $(whoami) 
Set 'qtap' persistent and owned by uid 0
root@x86:/home/ubuntu# ip link set dev qtap up
root@x86:/home/ubuntu# ip addr add 169.254.1.1/16 dev qtap
 ```
 
 ```
 qemu-system-arm -M xilinx-zynq-a9 -serial /dev/null -serial mon:stdio -display none -kernel my.uImage.p -dtb Prebuilt_functional/my_devicetree.dtb --initrd Prebuilt_functional/ramdisk.img.gz  -net tap,ifname=qtap,script=no,downscript=no   -net nic,model=cadence_gem,macaddr=0e:b0:ba:5e:ba:12 
 ```
 
 ```
 zedboard-zynq7 login: root
root@zedboard-zynq7:~# ls
root@zedboard-zynq7:~# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
[   12.150342] random: fast init done
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether 0e:b0:ba:5e:ba:12 brd ff:ff:ff:ff:ff:ff
3: tunl0@NONE: <NOARP> mtu 1480 qdisc noop state DOWN group default qlen 1000
    link/ipip 0.0.0.0 brd 0.0.0.0
root@zedboard-zynq7:~# ip a add 169.254.1.2/16 dev eth0
root@zedboard-zynq7:~# ping  169.254.1.1
PING 169.254.1.1 (169.254.1.1): 56 data bytes
[   28.229665] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=0 ttl=64 time=9.054 ms
[   28.231379] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=1 ttl=64 time=18.616 ms
[   29.249623] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=2 ttl=64 time=6.358 ms
[   30.239755] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=3 ttl=64 time=25.371 ms
[   31.260026] macb: ************  netif_rx **********
[   31.349965] macb: ************  netif_rx **********
[   31.350600] NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!
[   31.351604] NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!
64 bytes from 169.254.1.1: seq=4 ttl=64 time=12.956 ms
[   32.249933] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=5 ttl=64 time=2.461 ms
[   33.239920] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=6 ttl=64 time=29.248 ms
[   34.269949] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=7 ttl=64 time=18.135 ms
[   35.260064] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=8 ttl=64 time=5.699 ms
[   36.249758] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=9 ttl=64 time=24.362 ms
[   37.269935] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=10 ttl=64 time=13.238 ms
[   38.259921] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=11 ttl=64 time=1.862 ms
QEMU: Terminated
root@x86:/home/ubuntu/QEMU_CPUFreq_Zynq# 

 ```
 ## xilinx-zynq-a9 ping
   ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/gem_poll/pic/poll.jpg)
   
## telnet in xilinx-zynq-a9  
  ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/gem_poll/pic/telnet_in_xlinix.png)

## telnet in host
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/gem_poll/pic/telnet.png)
   
   
# macb_interrupt case1

```
static irqreturn_t macb_interrupt(int irq, void *dev_id)
#ifdef TEST_POLL
        /* Receive complete */
        if (status & MACB_BIT(RCOMP))
        {
                pr_err("*********** recv complete call gem_rx ************");
                gem_rx(queue, NULL, 0);
        }
#endif
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
#ifndef TEST_POLL
                        if (napi_schedule_prep(&queue->napi)) {
                                netdev_vdbg(bp->dev, "scheduling RX softirq\n");
                                __napi_schedule(&queue->napi);
                        }
#else
                        if (napi_schedule_prep(&queue->napi)) {
                                pr_err("*********** napi_schedul raise softirq  ************");
                                netdev_vdbg(bp->dev, "scheduling RX softirq\n");
                                __napi_schedule(&queue->napi);
                        }
#endif
                }
```

```
root@zedboard-zynq7:~# ip addr add 169.254.1.2/16 dev eth0
root@zedboard-zynq7:~# ping 169.254.1.1
PING 169.254.1.1 (169.254.1.1): 56 data bytes
[   40.329233] macb: *********** recv complete call gem_rx ************
[   40.329646] macb: ************  netif_rx **********
[   40.330232] macb: *********** napi_schedul raise softirq  ************
64 bytes from 169.254.1.1: seq=0 ttl=64 time=20.607 ms
[   40.331923] macb: ************  netif_rx **********
[   41.318946] macb: *********** recv complete call gem_rx ************
[   41.319048] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=1 ttl=64 time=3.459 ms
[   41.319147] macb: *********** napi_schedul raise softirq  ************
[   42.339689] macb: *********** recv complete call gem_rx ************
[   42.340159] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=2 ttl=64 time=21.499 ms
[   42.340617] macb: *********** napi_schedul raise softirq  ************
[   43.329598] macb: *********** recv complete call gem_rx ************
[   43.329991] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=3 ttl=64 time=9.724 ms
[   43.330353] macb: *********** napi_schedul raise softirq  ************
[   43.359306] macb: *********** recv complete call gem_rx ************
[   43.359488] macb: ************  netif_rx **********
[   43.359664] macb: *********** napi_schedul raise softirq  ************
[   43.359914] NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!
[   43.360254] NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!
[   44.349536] macb: *********** recv complete call gem_rx ************
[   44.349627] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=4 ttl=64 time=28.320 ms
[   44.350010] macb: *********** napi_schedul raise softirq  ************
[   45.339578] macb: *********** recv complete call gem_rx ************
[   45.339979] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=5 ttl=64 time=17.289 ms
[   45.340380] macb: *********** napi_schedul raise softirq  ************
[   46.329412] macb: *********** recv complete call gem_rx ************
[   46.329873] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=6 ttl=64 time=5.428 ms
[   46.330301] macb: *********** napi_schedul raise softirq  ************
[   47.349621] macb: *********** recv complete call gem_rx ************
[   47.350060] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=7 ttl=64 time=23.850 ms
[   47.350481] macb: *********** napi_schedul raise softirq  ************
[   48.339589] macb: *********** recv complete call gem_rx ************
[   48.339993] macb: ************  netif_rx **********
64 bytes from 169.254.1.1: seq=8 ttl=64 time=12.269 ms
[   48.340385] macb: *********** napi_schedul raise softirq  ************
[   49.330890] macb: *********** recv complete call gem_rx ************
[   49.331288] macb: ************  netif_rx **********
……
64 bytes from 169.254.1.1: seq=21 ttl=64 time=4.370 ms
[  149.620145] macb: *********** napi_schedul raise softirq  ************
[  149.739624] macb: *********** recv complete call gem_rx ************
[  149.740112] macb: ************  netif_rx **********
[  149.740564] macb: *********** napi_schedul raise softirq  ************
[  149.740961] NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!
[  149.741237] NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!
64 bytes from 169.254.1.1: seq=9 ttl=64 time=2.688 ms
^C
--- 169.254.1.1 ping statistics ---
10 packets transmitted, 10 packets received, 0% packet loss
round-trip min/avg/max = 2.688/14.513/28.320 ms
```

# napi

## cmp napi and non napi

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/gem_poll/pic/napi.png)

```
static irqreturn_t sample_netdev_intr(int irq, void *dev)
{
    struct net_device *netdev = dev;
    struct nic *nic = netdev_priv(netdev);
 
    if (! nic->irq_pending())
        return IRQ_NONE;
 
    /* Ack interrupt(s) */
    nic->ack_irq();
 
    nic->disable_irq();  
 
    netif_rx_schedule(netdev);
 
    return IRQ_HANDLED;
}
 
 
static int sample_netdev_poll(struct net_device *netdev, int *budget)
{
    struct nic *nic = netdev_priv(netdev);
 
    unsigned int work_to_do = min(netdev->quota, *budget);
    unsigned int work_done = 0;
 
    nic->announce(&work_done, work_to_do);
 
    /* If no Rx announce was done, exit polling state. */
 
    if(work_done == 0) || !netif_running(netdev)) {
 
    netif_rx_complete(netdev);
    nic->enable_irq();  
 
    return 0;
    }
 
    *budget -= work_done;
    netdev->quota -= work_done;
 
    return 1;
}
```
# arp


```
drivers/net/ethernet/chelsio/cxgb/sge.c
  /* Hmmm, assuming to catch the gratious arp... and we'll use
                 * it to flush out stuck espi packets...
                 */
                if ((unlikely(!adapter->sge->espibug_skb[dev->if_port]))) {
                        if (skb->protocol == htons(ETH_P_ARP) &&
                            arp_hdr(skb)->ar_op == htons(ARPOP_REQUEST)) {
                                adapter->sge->espibug_skb[dev->if_port] = skb;
                                /* We want to re-use this skb later. We
                                 * simply bump the reference count and it
                                 * will not be freed...
                                 */
                                skb = skb_get(skb);
                        }
                }
```
  
# references
[e100 NAPI](https://blog.csdn.net/Rong_Toa/article/details/109401935)