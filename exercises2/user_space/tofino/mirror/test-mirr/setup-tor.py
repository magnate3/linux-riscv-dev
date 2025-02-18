from ipaddress import ip_address
p4 = bfrt.tofino_mirr.pipe
l2_table = p4.Ingress.l2_forwarding
l2_table.clear()
l2_table.add_with_l2_forward(24, port=8)
l2_table.add_with_l2_forward(8, port=24)
mirr_table = p4.Ingress.ing_mirror_table
mirr_table.clear()
mirr_table.add_with_ing_mirror(8,123)
rep_table = p4.Egress.tb_generate_report
rep_table.clear()
rep_table.set_default_with_do_report_encapsulation(0x9a4b89ad8a59,0x0090fb792055, 0xa000301,0xa000306,1234)
#rep_table.add_with_do_report_encapsulation(0x9a4b89ad8a59,0x0090fb792055, 0xa000301,0xa000306,1234)
#rep_table.add_with_do_report_encapsulation(mon_ip="10.0.3.3", mon_mac="3c:fd:fe:ed:1d:c1", mon_port="4321", src_ip="172.26.0.4", src_mac="02:42:ac:1c:00:67")
#bfrt.mirror.cfg.delete(123)
#mirror_session_id = 123 # defined in p4 ing_mirror_table
#bfrt.mirror.cfg.add_with_normal(sid=123, direction='EGRESS', session_enable=True, ucast_egress_port=4, ucast_egress_port_valid=1, max_pkt_len=16384)
bfrt.mirror.cfg.add_with_normal(sid=123, direction='INGRESS', session_enable=True, ucast_egress_port=4, ucast_egress_port_valid=1, max_pkt_len=16384)
#bfrt.mirror.cfg.add_with_normal(sid=123, direction='BOTH', session_enable=True, ucast_egress_port=4, ucast_egress_port_valid=1, max_pkt_len=16384)
