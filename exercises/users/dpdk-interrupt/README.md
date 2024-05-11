

# dpdk interrupt example

[VF One-shot Rx Interrupt Tests](https://dpdk-test-plans.readthedocs.io/en/latest/vf_interrupt_pmd_test_plan.html)   



# pingpong


```
[root@centos7 dpdk-pingpong]# modprobe vfio-pci
./usertools/dpdk-devbind.py -b  vfio-pci  0000:05:00.0
```


> ## RX Intr vector unset
```
Initilize port 0 done.
RX Intr vector unset
```

```
int
rte_eth_dev_rx_intr_ctl_q_get_fd(uint16_t port_id, uint16_t queue_id)
{
        struct rte_intr_handle *intr_handle;
        struct rte_eth_dev *dev;
        unsigned int efd_idx;
        uint32_t vec;
        int fd;

        RTE_ETH_VALID_PORTID_OR_ERR_RET(port_id, -1);

        dev = &rte_eth_devices[port_id];

        if (queue_id >= dev->data->nb_rx_queues) {
                RTE_ETHDEV_LOG(ERR, "Invalid RX queue_id=%u\n", queue_id);
                return -1;
        }

        if (!dev->intr_handle) {
                RTE_ETHDEV_LOG(ERR, "RX Intr handle unset\n");
                return -1;
        }

        intr_handle = dev->intr_handle;
        if (!intr_handle->intr_vec) {
                RTE_ETHDEV_LOG(ERR, "RX Intr vector unset\n");
                return -1;
        }

        vec = intr_handle->intr_vec[queue_id];
        efd_idx = (vec >= RTE_INTR_VEC_RXTX_OFFSET) ?
                (vec - RTE_INTR_VEC_RXTX_OFFSET) : vec;
        fd = intr_handle->efds[efd_idx];

        return fd;
}
```