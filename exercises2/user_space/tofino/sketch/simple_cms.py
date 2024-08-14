from ipaddress import ip_address
p4 = bfrt.cms.pipe
ipv4_table = p4.Ingress.ipv4_forward
ipv4_table.add_with_forward(128,136)
ipv4_table.add_with_forward(136,128)
