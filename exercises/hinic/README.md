
# hinic module
```
[1237411.707807] hinic 0000:05:00.0 enp5s0: set rx mode work
[root@centos7 hinic]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 hinic]# 
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hinic/pic/kernel.png)

# pci bind
```
[root@centos7 hinic]# insmod  hinic.ko
[root@centos7 hinic]# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
3: enp125s0f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b0:08:75:5f:b8:5b brd ff:ff:ff:ff:ff:ff
    inet 10.10.16.251/24 brd 10.10.16.255 scope global noprefixroute enp125s0f0
       valid_lft forever preferred_lft forever
    inet6 fe80::a82e:8486:712:201a/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
4: enp125s0f1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether b0:08:75:5f:b8:5c brd ff:ff:ff:ff:ff:ff
5: enp125s0f2: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether b0:08:75:5f:b8:5d brd ff:ff:ff:ff:ff:ff
6: enp125s0f3: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b0:08:75:5f:b8:5e brd ff:ff:ff:ff:ff:ff
8: virbr0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default qlen 1000
    link/ether 52:54:00:af:8e:96 brd ff:ff:ff:ff:ff:ff
    inet 192.168.122.1/24 brd 192.168.122.255 scope global virbr0
       valid_lft forever preferred_lft forever
9: virbr0-nic: <BROADCAST,MULTICAST> mtu 1500 qdisc pfifo_fast master virbr0 state DOWN group default qlen 1000
    link/ether 52:54:00:af:8e:96 brd ff:ff:ff:ff:ff:ff
10: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default 
    link/ether 02:42:d3:64:43:fb brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 scope global docker0
       valid_lft forever preferred_lft forever
[root@centos7 hinic]# echo -n  '0000:05:00.0' >  /sys/bus/pci/drivers/hinic/bind
-bash: echo: write error: No such device
[root@centos7 hinic]# echo -n  '0000:05:00.0' >  /sys/bus/pci/drivers/hinic/unbind
-bash: echo: write error: No such device
```
## should  give product id and vendor id
```
[root@centos7 hinic]# lspci -n |grep '05:00.0'
05:00.0 0200: 19e5:0200 (rev 45)
[root@centos7 hinic]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/hinic/new_id 
[root@centos7 hinic]# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
3: enp125s0f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b0:08:75:5f:b8:5b brd ff:ff:ff:ff:ff:ff
    inet 10.10.16.251/24 brd 10.10.16.255 scope global noprefixroute enp125s0f0
       valid_lft forever preferred_lft forever
    inet6 fe80::a82e:8486:712:201a/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
4: enp125s0f1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether b0:08:75:5f:b8:5c brd ff:ff:ff:ff:ff:ff
5: enp125s0f2: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether b0:08:75:5f:b8:5d brd ff:ff:ff:ff:ff:ff
6: enp125s0f3: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b0:08:75:5f:b8:5e brd ff:ff:ff:ff:ff:ff
8: virbr0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default qlen 1000
    link/ether 52:54:00:af:8e:96 brd ff:ff:ff:ff:ff:ff
    inet 192.168.122.1/24 brd 192.168.122.255 scope global virbr0
       valid_lft forever preferred_lft forever
9: virbr0-nic: <BROADCAST,MULTICAST> mtu 1500 qdisc pfifo_fast master virbr0 state DOWN group default qlen 1000
    link/ether 52:54:00:af:8e:96 brd ff:ff:ff:ff:ff:ff
10: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default 
    link/ether 02:42:d3:64:43:fb brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 scope global docker0
       valid_lft forever preferred_lft forever
42: enp5s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 44:a1:91:a4:9c:0b brd ff:ff:ff:ff:ff:ff
43: enp6s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 44:a1:91:a4:9c:0c brd ff:ff:ff:ff:ff:ff
[root@centos7 hinic]# echo -n  '0000:05:00.0' >  /sys/bus/pci/drivers/hinic/bind
-bash: echo: write error: No such device
[root@centos7 hinic]# echo -n  '0000:05:00.0' >  /sys/bus/pci/drivers/hinic/unbind
[root@centos7 hinic]# echo -n  '0000:05:00.0' >  /sys/bus/pci/drivers/hinic/bind
[root@centos7 hinic]# 
```

## ping
```
[root@centos7 hinic]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/hinic/new_id 
[root@centos7 hinic]# ip a add 192.168.10.251/24 dev enp6s0
[root@centos7 hinic]# 
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hinic/pic/ping.png)

### ceqe
```
static void ceq_irq_handler(struct hinic_eq *eq)
{
        struct hinic_ceqs *ceqs = ceq_to_ceqs(eq);
        u32 ceqe;
        int i;

        for (i = 0; i < eq->q_len; i++) {
                ceqe = *(GET_CURR_CEQ_ELEM(eq));

                /* Data in HW is in Big endian Format */
                ceqe = be32_to_cpu(ceqe);
                pr_info("***************** ceqe val is %8x ", ceqe);
                /* HW toggles the wrapped bit, when it adds eq element event */
                if (HINIC_EQ_ELEM_DESC_GET(ceqe, WRAPPED) == eq->wrapped)
                        break;

                ceq_event_handler(ceqs, ceqe);

                eq->cons_idx++;

                if (eq->cons_idx == eq->q_len) {
                        eq->cons_idx = 0;
                        eq->wrapped = !eq->wrapped;
                }
        }
}
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hinic/pic/ceqe.png)


```
static irqreturn_t rx_irq(int irq, void *data)
{
        struct hinic_rxq *rxq = (struct hinic_rxq *)data;
        struct hinic_rq *rq = rxq->rq;
        struct hinic_dev *nic_dev;

        /* Disable the interrupt until napi will be completed */
        disable_irq_nosync(rq->irq);

        nic_dev = netdev_priv(rxq->netdev);
        hinic_hwdev_msix_cnt_set(nic_dev->hwdev, rq->msix_entry);

        napi_schedule(&rxq->napi);
        return IRQ_HANDLED;
}
```

 