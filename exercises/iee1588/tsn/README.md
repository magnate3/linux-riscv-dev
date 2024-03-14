
# TSN协议栈


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/stack.png)

#   LLDP_Multicast (01:80:c2:00:00:0e)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/lldp.png)


# tcpdump 

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/node.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/tcpdump.png)

## slave

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/slave.png)

## master

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/master.png)


#  !(status & E1000_RXDEXT_STATERR_TST) || !(er32(TSYNCRXCTL) & E1000_TSYNCRXCTL_VALID)

```
static void e1000e_rx_hwtstamp(struct e1000_adapter *adapter, u32 status,
                               struct sk_buff *skb)
{
        struct e1000_hw *hw = &adapter->hw;
        u64 rxstmp;

#if TEST_PTP
        pr_info("status & E1000_RXDEXT_STATERR_TST : %x, er32(TSYNCRXCTL) & E1000_TSYNCRXCTL_VALID): %x \n",(status & E1000_RXDEXT_STATERR_TST), (er32(TSYNCRXCTL) & E1000_TSYNCRXCTL_VALID));
#endif
        if (!(adapter->flags & FLAG_HAS_HW_TIMESTAMP) ||
            !(status & E1000_RXDEXT_STATERR_TST) ||
            !(er32(TSYNCRXCTL) & E1000_TSYNCRXCTL_VALID))
                return;

        /* The Rx time stamp registers contain the time stamp.  No other
         * received packet will be time stamped until the Rx time stamp
         * registers are read.  Because only one packet can be time stamped
         * at a time, the register values must belong to this packet and
         * therefore none of the other additional attributes need to be
         * compared.
         */
        rxstmp = (u64)er32(RXSTMPL);
        rxstmp |= (u64)er32(RXSTMPH) << 32;
        e1000e_systim_to_hwtstamp(adapter, skb_hwtstamps(skb), rxstmp);
#if TEST_PTP
        struct skb_shared_hwtstamps *hwts = skb_hwtstamps(skb);
        ktime_t t = hwts->hwtstamp;
        printk("%s()  ktime_get=%llu\n", __func__, ktime_to_ns(t));
#endif
        adapter->flags2 &= ~FLAG2_CHECK_RX_HWTSTAMP;
}
```

```
static void e1000_receive_skb(struct e1000_adapter *adapter,
                              struct net_device *netdev, struct sk_buff *skb,
                              u32 staterr, __le16 vlan)
{
        u16 tag = le16_to_cpu(vlan);

        e1000e_rx_hwtstamp(adapter, staterr, skb);

        skb->protocol = eth_type_trans(skb, netdev);

#if TEST_PTP
        struct ethhdr *eth =  (struct ethhdr *)skb_mac_header(skb);
        if(htons(ETH_P_1588) == eth->h_proto)
        {
             pr_info("1588 packet recv \n");
        }
#endif
        if (staterr & E1000_RXD_STAT_VP)
                __vlan_hwaccel_put_tag(skb, htons(ETH_P_8021Q), tag);

        napi_gro_receive(&adapter->napi, skb);
}
```

## iperf

```
iperf3 -c 192.168.6.1    -p 5201 -i 1  -l 6556
 iperf3  -s  -B 192.168.6.1 -4
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/topo.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/reg.png)


# ptp4l

```
 ptp4l  -i enp0s31f6 -m -f gPTP.cfg
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/ie1588.png)


# ptp_parse

```
#define PTP_MSGTYPE_SYNC        0x0
#define PTP_MSGTYPE_DELAY_REQ   0x1
#define PTP_MSGTYPE_PDELAY_REQ  0x2
#define PTP_MSGTYPE_PDELAY_RESP 0x3
/* Values for the messageType field */
#define SYNC                  0x0
#define DELAY_REQ             0x1
#define PDELAY_REQ            0x2
#define PDELAY_RESP           0x3
#define FOLLOW_UP             0x8
#define DELAY_RESP            0x9
#define PDELAY_RESP_FOLLOW_UP 0xA
#define ANNOUNCE              0xB
#define SIGNALING             0xC
#define MANAGEMENT            0xD

```

**include/linux/ptp_classify.h**

```
#include <linux/ip.h>
#include <linux/skbuff.h>

#define PTP_CLASS_NONE  0x00 /* not a PTP event message */
#define PTP_CLASS_V1    0x01 /* protocol version 1 */
#define PTP_CLASS_V2    0x02 /* protocol version 2 */
#define PTP_CLASS_VMASK 0x0f /* max protocol version is 15 */
#define PTP_CLASS_IPV4  0x10 /* event in an IPV4 UDP packet */
#define PTP_CLASS_IPV6  0x20 /* event in an IPV6 UDP packet */
#define PTP_CLASS_L2    0x40 /* event in a L2 packet */
#define PTP_CLASS_PMASK 0x70 /* mask for the packet type field */
#define PTP_CLASS_VLAN  0x80 /* event in a VLAN tagged packet */

#define PTP_CLASS_V1_IPV4 (PTP_CLASS_V1 | PTP_CLASS_IPV4)
#define PTP_CLASS_V1_IPV6 (PTP_CLASS_V1 | PTP_CLASS_IPV6) /* probably DNE */
#define PTP_CLASS_V2_IPV4 (PTP_CLASS_V2 | PTP_CLASS_IPV4)
#define PTP_CLASS_V2_IPV6 (PTP_CLASS_V2 | PTP_CLASS_IPV6)
#define PTP_CLASS_V2_L2   (PTP_CLASS_V2 | PTP_CLASS_L2)
#define PTP_CLASS_V2_VLAN (PTP_CLASS_V2 | PTP_CLASS_VLAN)
#define PTP_CLASS_L4      (PTP_CLASS_IPV4 | PTP_CLASS_IPV6)

#define PTP_MSGTYPE_SYNC        0x0
#define PTP_MSGTYPE_DELAY_REQ   0x1
#define PTP_MSGTYPE_PDELAY_REQ  0x2
#define PTP_MSGTYPE_PDELAY_RESP 0x3
```

```
static int enetc_ptp_parse(struct sk_buff *skb, u8 *udp,
                           u8 *msgtype, u8 *twostep,
                           u16 *correction_offset, u16 *body_offset)
{
        unsigned int ptp_class;
        struct ptp_header *hdr;
        unsigned int type;
        u8 *base;

        ptp_class = ptp_classify_raw(skb);
        if (ptp_class == PTP_CLASS_NONE)
                return -EINVAL;

        hdr = ptp_parse_header(skb, ptp_class);
        if (!hdr)
                return -EINVAL;

        type = ptp_class & PTP_CLASS_PMASK;
        if (type == PTP_CLASS_IPV4 || type == PTP_CLASS_IPV6)
                *udp = 1;
        else
                *udp = 0;

        *msgtype = ptp_get_msgtype(hdr, ptp_class);
        *twostep = hdr->flag_field[0] & 0x2;

        base = skb_mac_header(skb);
        *correction_offset = (u8 *)&hdr->correction - base;
        *body_offset = (u8 *)hdr + sizeof(struct ptp_header) - base;

        return 0;
}
```

```
static void e1000_receive_skb(struct e1000_adapter *adapter,
                              struct net_device *netdev, struct sk_buff *skb,
                              u32 staterr, __le16 vlan)
{
        u16 tag = le16_to_cpu(vlan);

        e1000e_rx_hwtstamp(adapter, staterr, skb);

        skb->protocol = eth_type_trans(skb, netdev);

#if TEST_PTP
        unsigned int ptp_class=PTP_CLASS_L2;
        struct ptp_header *hdr;
        unsigned int type;
        u8 msgtype=0xff;
        struct ethhdr *eth ;
        //ptp_class = ptp_classify_raw(skb);
        //if (ptp_class == PTP_CLASS_NONE)
        //{
        //     pr_info(" ptp class none\n");  
        //     goto out;
        //}
        hdr = ptp_parse_header(skb, ptp_class);
        if (NULL == hdr)
        {
             pr_info(" ptp header null\n");
             goto out;
        }
        msgtype = ptp_get_msgtype(hdr, ptp_class);
out:
        eth =  (struct ethhdr *)skb_mac_header(skb);
        if(htons(ETH_P_1588) == eth->h_proto)
        {
             pr_info("1588 packet recv and msg type %x \n", msgtype);
        }

#endif
        if (staterr & E1000_RXD_STAT_VP)
                __vlan_hwaccel_put_tag(skb, htons(ETH_P_8021Q), tag);

        napi_gro_receive(&adapter->napi, skb);
}
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/msg.png)
 
***SYNC、Delay_Req、Delay_Resp报文带有时间戳***

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/session.png)

# references

[PPSi/tools/dump-main.c](https://github.com/ngbrown/PPSi/blob/d4fcfa62ecb757679de68cacbbf5d30083b1aba4/tools/dump-main.c)