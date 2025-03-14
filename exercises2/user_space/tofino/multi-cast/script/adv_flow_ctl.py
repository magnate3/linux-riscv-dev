def Namespace():
    # 10.0.0.1~9 10.0.0.20~21
    pcie_port = 192
    #base = 0
    base = 2785050624
    base = 2283831296
    pause_val = base + 1
    resume_val = base 
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
    tbl2 = bfrt.simple_cast.pipe.Egress.flow_ctl
    tbl2.clear()
    for worker in range(len(port)):
        tbl.add_with_l2_forward(MAC[worker], port[worker],1,pause_val)
        tbl2.add_with_set_adv_flow_ctl(port[worker],pause_val)

Namespace()
