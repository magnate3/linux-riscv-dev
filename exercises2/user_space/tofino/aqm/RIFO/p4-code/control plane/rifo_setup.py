from ipaddress import ip_address

p4 = bfrt.rifo.pipe

# Clear All tables
def clear_all():
    global p4

    # The order is important. We do want to clear from the top, i.e.
    # delete objects that use other objects, e.g. table entries use
    # selector groups and selector groups use action profile members

    # Clear Match Tables
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR']:
            print("Clearing table {}".format(table['full_name']))
            for entry in table['node'].get(regex=True):
                entry.remove()
    # Clear Selectors
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['SELECTOR']:
            print("Clearing ActionSelector {}".format(table['full_name']))
            for entry in table['node'].get(regex=True):
                entry.remove()
    # Clear Action Profiles
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['ACTION_PROFILE']:
            print("Clearing ActionProfile {}".format(table['full_name']))
            for entry in table['node'].get(regex=True):
                entry.remove()

#clear_all()
ipv4_host = p4.Ingress.ipv4_host
queue_length_lookup = p4.Ingress.queue_length_lookup
max_min_lookup = p4.Ingress.max_min_lookup
max_min_buffer_lookup = p4.Ingress.max_min_buffer_lookup
dividend_lookup = p4.Ingress.dividend_lookup

for i in range(1,16):
    val='0x'+format(1 << i, 'x').zfill(4)
    print('va',val)
    queue_length_lookup.add_with_set_exponent_buffer(available_queue=2**(i-1),available_queue_mask='0b'+format((1 << i) - 1, '016b'),exponent_value=int(i))

for i in range(1,16):
    max_min_lookup.add_with_set_exponent_max_min(max_min=2**(i-1),max_min_mask='0b'+format((1 << i) - 1, '016b'),exponent_value=int(i))

for i in range(1,16):
    dividend_lookup.add_with_set_exponent_dividend(dividend=2**(i-1),dividend_mask='0b'+format((1 << i) - 1, '016b'),exponent_value=int(i))

for i in range(1,17):
    for j in range(1,17):
        max_min_buffer_lookup.add_with_calculate_max_min_buffer_mul(max_min_exponent=i,buffer_exponent=j,mul=i*j)


ipv4_host.add_with_send(dst_addr=ip_address('192.168.40.1'),  port=156)
ipv4_host.add_with_send(dst_addr=ip_address('192.168.45.1'),  port=148)
ipv4_host.add_with_send(dst_addr=ip_address('192.168.40.20'),  port=140)
ipv4_host.add_with_send(dst_addr=ip_address('192.168.45.20'),  port=132)


bfrt.tf1.tm.port.group.mod_with_seq(pg_id=0x01, pipe=1, port_queue_count=[1, 0, 0, 0]) #-> dev_port: 132
bfrt.tf1.tm.port.group.mod_with_seq(pg_id=0x03, pipe=1, port_queue_count=[1, 0, 0, 0]) #-> dev_port: 140



prt = bfrt.port.port
bfrt.port.port.add(DEV_PORT=132, SPEED='BF_SPEED_100G', FEC='BF_FEC_TYP_REED_SOLOMON', PORT_ENABLE=True)
bfrt.port.port.add(DEV_PORT=140, SPEED='BF_SPEED_100G', FEC='BF_FEC_TYP_REED_SOLOMON', PORT_ENABLE=True)
bfrt.port.port.add(DEV_PORT=148, SPEED='BF_SPEED_100G', FEC='BF_FEC_TYP_REED_SOLOMON', PORT_ENABLE=True)
bfrt.port.port.add(DEV_PORT=156, SPEED='BF_SPEED_100G', FEC='BF_FEC_TYP_REED_SOLOMON', PORT_ENABLE=True)


myPorts=[132,140]
bfrt.tf1.tm.port.sched_cfg.mod(dev_port=132, max_rate_enable=True)
bfrt.tf1.tm.port.sched_cfg.mod(dev_port=140, max_rate_enable=True)
bfrt.tf1.tm.port.sched_shaping.mod(dev_port=132, unit='BPS', provisioning='MIN_ERROR', max_rate=15000000, max_burst_size=160)
bfrt.tf1.tm.port.sched_shaping.mod(dev_port=140, unit='BPS', provisioning='MIN_ERROR', max_rate=15000000, max_burst_size=160)

def get_pg_info(dev_port, queue_id):
    pipe_num = dev_port >> 7
    entry = bfrt.tf1.tm.port.cfg.get(dev_port, print_ents=False)
    pg_id = entry.data[b'pg_id']
    pg_queue = entry.data[b'egress_qid_queues'][queue_id]

    print('DEV_PORT: {} QueueID: {} --> Pipe: {}, PG_ID: {}, PG_QUEUE: {}'.format(dev_port, queue_id, pipe_num, pg_id, pg_queue))

    return pipe_num, pg_id, pg_queue

for port_number in myPorts:
    for queue_id in range(8):
        pipe_num, pg_id, pg_queue=get_pg_info(port_number, queue_id)
        bfrt.tf1.tm.queue.sched_cfg.mod(pipe=pipe_num, pg_id=pg_id, pg_queue=pg_queue, min_priority=queue_id,max_priority=queue_id)

bfrt.complete_operations()

# # Final programming
print("""
# ******************* PROGAMMING RESULTS *****************
# """)
print ("Table ipv4_host:")
ipv4_host.dump(table=True)
print ("Table queue_length_lookup:")
queue_length_lookup.dump(table=True)
print ("Table max_min_lookup:")
max_min_lookup.dump(table=True)
print ("Table max_min_buffer_lookup:")
max_min_buffer_lookup.dump(table=True)