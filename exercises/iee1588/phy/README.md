# dp83640 

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/phy/dp83640.png)

## CONFIG_NETWORK_PHY_TIMESTAMPING
 
 
These are devices that typically fulfill a Layer 1 role in the network stack,
hence they do not have a representation in terms of a network interface as DSA
switches do. However, PHYs may be able to detect and timestamp PTP packets, for
performance reasons: timestamps taken as close as possible to the wire have the
potential to yield a more stable and precise synchronization.

A PHY driver that supports PTP timestamping must create a ``struct
mii_timestamper`` and add a pointer to it in ``phydev->mii_ts``. The presence
of this pointer will be checked by the networking stack.

Since PHYs do not have network interface representations, the timestamping and
ethtool ioctl operations for them need to be mediated by their respective MAC
driver.  Therefore, as opposed to DSA switches, modifications need to be done
to each individual MAC driver for PHY timestamping support. This entails:

- Checking, in ``.ndo_eth_ioctl``, whether ``phy_has_hwtstamp(netdev->phydev)``
  is true or not. If it is, then the MAC driver should not process this request
  but instead pass it on to the PHY using ``phy_mii_ioctl()``.

- On RX, special intervention may or may not be needed, depending on the
  function used to deliver skb's up the network stack. In the case of plain
  ``netif_rx()`` and similar, MAC drivers must check whether
  ``skb_defer_rx_timestamp(skb)`` is necessary or not - and if it is, don't
  call ``netif_rx()`` at all.  If ``CONFIG_NETWORK_PHY_TIMESTAMPING`` is
  enabled, and ``skb->dev->phydev->mii_ts`` exists, its ``.rxtstamp()`` hook
  will be called now, to determine, using logic very similar to DSA, whether
  deferral for RX timestamping is necessary.  Again like DSA, it becomes the
  responsibility of the PHY driver to send the packet up the stack when the
  timestamp is available.

  For other skb receive functions, such as ``napi_gro_receive`` and
  ``netif_receive_skb``, the stack automatically checks whether
  ``skb_defer_rx_timestamp()`` is necessary, so this check is not needed inside
  the driver.
  
  

- On TX, again, special intervention might or might not be needed.  The
  function that calls the ``mii_ts->txtstamp()`` hook is named
  ``skb_clone_tx_timestamp()``. This function can either be called directly
  (case in which explicit MAC driver support is indeed needed), but the
  function also piggybacks from the ``skb_tx_timestamp()`` call, which many MAC
  drivers already perform for software timestamping purposes. Therefore, if a
  MAC supports software timestamping, it does not need to do anything further
  at this stage.
  
  
 


# skb_defer_rx_timestamp

```
  static int netif_receive_skb_internal(struct sk_buff *skb)
{
        int ret;

        net_timestamp_check(netdev_tstamp_prequeue, skb);

        if (skb_defer_rx_timestamp(skb))
                return NET_RX_SUCCESS;

        rcu_read_lock();
 
        ret = __netif_receive_skb(skb);
        rcu_read_unlock();
        return ret;
}
```

  netif_receive_skb_internal函数 函数首先检查接收的数据包是否为PTP协议帧，如果是则调用PHY驱动添加时间戳；
```
bool skb_defer_rx_timestamp(struct sk_buff *skb)
{
        struct mii_timestamper *mii_ts;
        unsigned int type;

        if (!skb->dev || !skb->dev->phydev || !skb->dev->phydev->mii_ts)
                return false;

        if (skb_headroom(skb) < ETH_HLEN)
                return false;

        __skb_push(skb, ETH_HLEN);

        type = ptp_classify_raw(skb);

        __skb_pull(skb, ETH_HLEN);

        if (type == PTP_CLASS_NONE)
                return false;

        mii_ts = skb->dev->phydev->mii_ts;
        if (likely(mii_ts->rxtstamp))
                return mii_ts->rxtstamp(mii_ts, skb, type);

        return false;
}
EXPORT_SYMBOL_GPL(skb_defer_rx_timestamp);
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/phy/rx_hw.png)


##  skb_tx_timestamp
  skb_tx_timestamp() - Driver hook for transmit timestamping
 
  Ethernet MAC Drivers should call this function in their hard_xmit()
  function immediately before giving the sk_buff to the MAC hardware.
  Specifically, one should make absolutely sure that this function is
  called before TX completion of this packet can trigger.  Otherwise
  the packet could potentially already be freed.
 
```
static inline void skb_tx_timestamp(struct sk_buff *skb)
{
        skb_clone_tx_timestamp(skb);
        if (skb_shinfo(skb)->tx_flags & SKBTX_SW_TSTAMP)
                skb_tstamp_tx(skb, NULL);
}



void skb_clone_tx_timestamp(struct sk_buff *skb)
{
        struct mii_timestamper *mii_ts;
        struct sk_buff *clone;
        unsigned int type;

        if (!skb->sk)
                return;

        type = classify(skb);
        if (type == PTP_CLASS_NONE)
                return;

        mii_ts = skb->dev->phydev->mii_ts;
        if (likely(mii_ts->txtstamp)) {
                clone = skb_clone_sk(skb);
                if (!clone)
                        return;
                mii_ts->txtstamp(mii_ts, clone, type);
        }
}
EXPORT_SYMBOL_GPL(skb_clone_tx_timestamp);

```

##  netdev_tx_t e1000_xmit_frame call  skb_tx_timestamp

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/phy/e1000e.png)

#  CONFIG_NETWORK_PHY_TIMESTAMPING 
```
 
ubuntu@ubuntux86:/boot$ grep  CONFIG_NETWORK_PHY_TIMESTAMPING  config-5.13.0-39-generic
CONFIG_NETWORK_PHY_TIMESTAMPING=y
ubuntu@ubuntux86:/boot$ 
```



# drivers/ptp/ptp_ines.c


## ines_ptp_probe_channel

```

static struct mii_timestamper *ines_ptp_probe_channel(struct device *device,
                                                      unsigned int index)
{
        struct device_node *node = device->of_node;
        struct ines_port *port;

        if (index > INES_N_PORTS - 1) {
                dev_err(device, "bad port index %u\n", index);
                return ERR_PTR(-EINVAL);
        }
        port = ines_find_port(node, index);
        if (!port) {
                dev_err(device, "missing port index %u\n", index);
                return ERR_PTR(-ENODEV);
        }
        port->mii_ts.rxtstamp = ines_rxtstamp;
        port->mii_ts.txtstamp = ines_txtstamp;
        port->mii_ts.hwtstamp = ines_hwtstamp;
        port->mii_ts.link_state = ines_link_state;
        port->mii_ts.ts_info = ines_ts_info;

        return &port->mii_ts;
}
```

```
static bool ines_rxtstamp(struct mii_timestamper *mii_ts,
                          struct sk_buff *skb, int type)
{
        struct ines_port *port = container_of(mii_ts, struct ines_port, mii_ts);
        struct skb_shared_hwtstamps *ssh;
        u64 ns;

        if (!port->rxts_enabled)
                return false;

        ns = ines_find_rxts(port, skb, type);
        if (!ns)
                return false;

        ssh = skb_hwtstamps(skb);
        ssh->hwtstamp = ns_to_ktime(ns);
        netif_rx(skb);

        return true;
}

```