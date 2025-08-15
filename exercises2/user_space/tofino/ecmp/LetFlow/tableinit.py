from ipaddress import ip_address

p4 = bfrt.flowlet.pipe

path="../table/"

'''ecmp_select'''
table=p4.Ingress.ecmp_select_t
table.clear()
table.add_with_ecmp_select(dst_addr=ip_address("10.10.203.4") , ecmp=1,port=8)
table.add_with_ecmp_select(dst_addr=ip_address("10.10.203.4") , ecmp=0,port=8)
table.add_with_ecmp_select(dst_addr=ip_address("10.10.203.3") , ecmp=1,port=24)
table.add_with_ecmp_select(dst_addr=ip_address("10.10.203.3") , ecmp=0,port=24)


'''if_ecmp'''

table=p4.Ingress.if_ecmp_t
table.clear()
table.add_with_if_ecmp(dst_addr=ip_address("10.10.203.4"), sign=1)
table.add_with_if_ecmp(dst_addr=ip_address("10.10.203.3"), sign=1)

