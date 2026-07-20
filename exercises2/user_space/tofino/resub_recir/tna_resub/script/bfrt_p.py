def Namespace():
    tbl = bfrt.tna_resubmit.pipe.SwitchIngress.output_port
    tbl.clear()
    tbl.add_with_set_output_port(8, 24)
    #tbl.add_with_set_output_port(24,8)
    tbl.add_with_recirculate(24,40)
    tbl.add_with_set_output_port(40,8)
    tbl2 = bfrt.tna_resubmit.pipe.SwitchIngress.resubmit_ctrl
    tbl2.clear()
    tbl2.add_with_resubmit_add_hdr(8,24)
    #tbl2.add_with_resubmit_no_hdr(24,8)
  


Namespace()
