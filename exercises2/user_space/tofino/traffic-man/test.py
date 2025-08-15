from ipaddress import ip_address

p4 = bfrt.p4_hello_world.pipe


def get_pg_info(dev_port, queue_id):
    pipe_num = dev_port >> 7
    entry = bfrt.tf1.tm.port.cfg.get(dev_port, print_ents=False)
    pg_id = entry.data[b'pg_id']
    pg_queue = entry.data[b'egress_qid_queues'][queue_id]

    print('DEV_PORT: {} QueueID: {} --> Pipe: {}, PG_ID: {}, PG_QUEUE: {}'.format(dev_port, queue_id, pipe_num, pg_id, pg_queue))

    return pipe_num, pg_id, pg_queue



forward = p4.SwitchIngress.forward

# forward.add_with_hit(dst_addr="10.0.0.2",  port=5)
# forward.add_with_hit(dst_addr="10.0.0.1",  port=3)

myPorts=[3,5]
bfrt.tf1.tm.port.sched_cfg.mod(dev_port=3, max_rate_enable=True)
bfrt.tf1.tm.port.sched_cfg.mod(dev_port=5, max_rate_enable=True)
# bfrt.tf1.tm.port.sched_cfg.mod(dev_port=3, min_rate_enable=True)
# # bfrt.tf1.tm.port.sched_cfg.mod(dev_port=5, min_rate_enable=True)
bfrt.tf1.tm.port.sched_shaping.mod(dev_port=3, unit='PPS', provisioning='MIN_ERROR', max_rate=1, max_burst_size=1)
bfrt.tf1.tm.port.sched_shaping.mod(dev_port=5, unit='PPS', provisioning='MIN_ERROR', max_rate=1, max_burst_size=1)
#
for port_number in myPorts:
    for queue_id in range(8):
        pipe_num, pg_id, pg_queue=get_pg_info(port_number, queue_id)
        bfrt.tf1.tm.queue.sched_cfg.mod(pipe=pipe_num, pg_id=pg_id, pg_queue=pg_queue, min_priority=queue_id,max_priority=queue_id)
#

bfrt.complete_operations()
