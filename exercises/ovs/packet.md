


#  dp_packet_delete

```
  } else if (OVS_UNLIKELY(any_miss)) {
        DP_PACKET_BATCH_FOR_EACH (i, packet, packets_) {
            if (OVS_UNLIKELY(!rules[i])) {
                dp_packet_delete(packet);
                COVERAGE_INC(datapath_drop_lock_error);
                upcall_fail_cnt++;
            }
        }
    }
```