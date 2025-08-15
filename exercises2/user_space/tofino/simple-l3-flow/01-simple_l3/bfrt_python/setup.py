from ipaddress import ip_address

p4 = bfrt.simple_l3_dir_cntr.pipe

ipv4_host = p4.Ingress.ipv4_host
ipv4_host.add_with_ipv4_host_send(ip_address('192.168.1.1'), 1)
ipv4_host.add_with_ipv4_host_send(ip_address('192.168.1.2'), 2)
ipv4_host.add_with_ipv4_host_drop(ip_address('192.168.1.3'))
ipv4_host.add_with_ipv4_host_send(ip_address('192.168.1.254'), 64)

ipv4_lpm =  p4.Ingress.ipv4_lpm
ipv4_lpm.add_with_send(ip_address('192.168.1.0'), 24, 1)
ipv4_lpm.add_with_drop(ip_address('192.168.0.0'), 16)
ipv4_lpm.add_with_send(ip_address('0.0.0.0'),      0, 64)

def clear_counters(table_node):
    for e in table_node.get(regex=True):
        e.data[b'$COUNTER_SPEC_BYTES'] = 0
        e.data[b'$COUNTER_SPEC_PKTS'] = 0
        e.push()
