

```
./build/pingpong -c0x3 -n 4
```


```
EAL:   probe driver: 8086:10fb rte_ixgbe_pmd
EAL:   Not managed by a supported kernel driver, skipped
PMD: ixgbe_dev_rx_queue_setup(): sw_ring=0x7fee585a34c0 sw_sc_ring=0x7fee585a1380 hw_ring=0x7fee585a5600 dma_addr=0x1004da5600
EAL: Error - exiting with code: 1
  Cause: rte_eth_tx_queue_setup:err=-22, port=0
```