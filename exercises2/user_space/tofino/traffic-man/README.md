

```
bfrt.tf1.tm.port.sched_shaping
entry = bfrt.tf1.tm.port.cfg.get(dev_port, print_ents=False)
bfrt.tf1.tm.queue.sched_cfg.mod(pipe=pipe_num, pg_id=pg_id, pg_queue=pg_queue, min_priority=queue_id,max_priority=queue_id)
```


```
def get_pg_info(dev_port, queue_id):
    pipe_num = dev_port >> 7
    entry = bfrt.tf1.tm.port.cfg.get(dev_port, print_ents=False)
    pg_id = entry.data[b'pg_id']
    pg_queue = entry.data[b'egress_qid_queues'][queue_id]

    print('DEV_PORT: {} QueueID: {} --> Pipe: {}, PG_ID: {}, PG_QUEUE: {}'.format(dev_port, queue_id, pipe_num, pg_id, pg_queue))

    return pipe_num, pg_id, pg_queue
```


```
bfrt.tf1.tm.port.cfg> get(28)
Entry 0:
Entry key:
    dev_port                       : 0x0000001C
Entry data:
    pg_id                          : 0x07
    pg_port_nr                     : 0x00
    port_queues_count              : 0x08
    ingress_qid_map                : [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    egress_qid_queues              : [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    copy_to_cpu                    : False
    mirror_drop_destination        : False
    mirror_drop_pg_queue           : 0x00
    pkt_extraction_credits         : 0x00000014

Out[17]: Entry for tf1.tm.port.cfg table.

bfrt.tf1.tm.port.cfg> 
```