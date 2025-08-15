def Namespace():
    # 10.0.0.1~9 10.0.0.20~21
    pcie_port = 192
    #pcie_mac = 0x000200000300
    #port = [180, 164, 148, 132, pcie_port]
    #MAC = [0x1070fd190095, 0x1070fd2fd851, 0x1070fd2fe441, 0x1070fd2fd421, pcie_mac]
    port = [8,24]
    # server220,server221
    MAC = [0x6cfe543dba88,0x6cfe543d8a39]
    # In doc of TNA, 192 is CPU PCIE port and 64~67 is CPU Ethernet ports for 2-pipe TF1
    # 0x000200000300 is the MAC address of bf_pci0, it may not always be this value
    # And, I found that copy_to_cpu not need to be set if we use port 192, so copy_to_cpu is useless ?

    tbl = bfrt.simple_cast.pipe.Ingress.l2_forward_table
    tbl.clear()
    for worker in range(len(port)):
        tbl.add_with_l2_forward(MAC[worker], port[worker],)

    bfrt.simple_cast.pipe.Egress.dcqcn.wred.add(0, 0, 125, 2500, 0.01)
    # DCQCN
    # 0 ~ 10KB, 0 
    # 10 ~ 200KB, 0 ~ 0.01
    # 200KB ~, 1

    node_list = []

    for index, p in enumerate(port):
        if p == pcie_port: # this port do not need to be added, and add it will cause error in mgid.add()
            continue
        #bfrt.port.port.add(p, 'BF_SPEED_100G', 'BF_FEC_TYP_RS', 4, True, 'PM_AN_FORCE_DISABLE')
        node_id = index + 1
        node_list.append(node_id)
        bfrt.pre.node.add(node_id, 0, None, [p]) # node_id, rid, lag_id, dev_port

    mgid = 1 
    bfrt.pre.mgid.add(mgid, node_list, [False] * len(node_list), [0] * len(node_list)) # mgid, node_id, L1_XID_VALID, L1_XID

    bfrt.simple_cast.pipe.Ingress.l2_forward_table.add_with_l2_multicast(0xffffffffffff, mgid)

Namespace()
