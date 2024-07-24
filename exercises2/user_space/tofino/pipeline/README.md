
# ChaCha-Tofino   

##  12 stages + 12 stages 


```
cat build/chacha/tofino/pipe/logs/table_summary.log 
Table allocation done 1 time(s), state = INITIAL
Number of stages in table allocation: 12
  Number of stages for ingress table allocation: 12
  Number of stages for egress table allocation: 12
Critical path length through the table dependency graph: 12
Number of tables allocated: 24
+-------+-------------------------+
|Stage  |Table Name               |
+-------+-------------------------+
|0      |MyEgressControl.tb_e0    |
|0      |MyIngressControl.tb_i0   |
|1      |MyEgressControl.tb_e1    |
|1      |MyIngressControl.tb_i1   |
|2      |MyEgressControl.tb_e2    |
|2      |MyIngressControl.tb_i2   |
|3      |MyEgressControl.tb_e3    |
|3      |MyIngressControl.tb_i3   |
|4      |MyEgressControl.tb_e4    |
|4      |MyIngressControl.tb_i4   |
|5      |MyEgressControl.tb_e5    |
|5      |MyIngressControl.tb_i5   |
|6      |MyEgressControl.tb_e6    |
|6      |MyIngressControl.tb_i6   |
|7      |MyEgressControl.tb_e7    |
|7      |MyIngressControl.tb_i7   |
|8      |MyEgressControl.tb_e8    |
|8      |MyIngressControl.tb_i8   |
|9      |MyEgressControl.tb_e9    |
|9      |MyIngressControl.tb_i9   |
|10     |MyEgressControl.tb_e10   |
|10     |MyIngressControl.tb_i10  |
|11     |MyEgressControl.tb_e11   |
|11     |MyIngressControl.tb_i11  |
+-------+-------------------------+
```

+   Ingress 12个stages        
```
control MyIngressControl(inout headers hdr,
                inout ig_metadata meta,
                in ingress_intrinsic_metadata_t ig_intr_md,
                in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
                inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
                inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {
    
    Random<bit<32>>() random32_0;
    Random<bit<32>>() random32_1;
    Random<bit<32>>() random32_2;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_0;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_1;
        
    #include "ig_actions.p4"
    #include "ig_tables.p4"

    apply {
        tb_i0.apply();
        tb_i1.apply();
        tb_i2.apply();
        tb_i3.apply();
        tb_i4.apply();
        tb_i5.apply();
        tb_i6.apply();
        tb_i7.apply();
        tb_i8.apply();
        tb_i9.apply();
        tb_i10.apply();
        tb_i11.apply();
    }

}
```

+   Egress 12个stages     
```
control MyEgressControl(
    inout headers hdr,
    inout eg_metadata meta,
    in egress_intrinsic_metadata_t eg_intr_md,
    in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t eg_oport_md) {
    
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_0;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_1;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_2;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_3;

    #include "eg_actions.p4"
    #include "eg_tables.p4"
    
    apply {
        tb_e0.apply();
        tb_e1.apply();
        tb_e2.apply();
        tb_e3.apply();
        tb_e4.apply();
        tb_e5.apply();
        tb_e6.apply();
        tb_e7.apply();
        tb_e8.apply();
        tb_e9.apply();
        tb_e10.apply();
        tb_e11.apply();
    }
}
```

#  my test


```
[--Werror=legacy] error: Assignment to a header field in the deparser is only allowed when the source is checksum update, mirror, resubmit or learning digest. Please move the assignment into the control flow AssignmentStatement
```

```

control EgressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                      hdr,
    in    my_egress_metadata_t                      meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
#if 1
    action action14() {
       //meta.out_item1 = 64;
       hdr.ipv4.ttl= 64;
   }
      table tab14 {
      key     = { hdr.ipv4.dst_addr : lpm;hdr.vlan_tag.vid:  exact;}
      actions = {
          action14;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }
#endif
    apply {
        tab14.apply();
        pkt.emit(hdr);
    }
}
```

将table tab14 移动到EgressDeparser    

```

    /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                      hdr,
    in    my_egress_metadata_t                      meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
#if 1
    action action14() {
       //meta.out_item1 = 64;
       hdr.ipv4.ttl= 64;
   }
      table tab14 {
      key     = { hdr.ipv4.dst_addr : lpm;hdr.vlan_tag.vid:  exact;}
      actions = {
          action14;
          @defaultonly NoAction;
      }
      const default_action = NoAction();
      size = 64;
      }
#endif
    apply {
        tab14.apply();
        pkt.emit(hdr);
    }
}
```