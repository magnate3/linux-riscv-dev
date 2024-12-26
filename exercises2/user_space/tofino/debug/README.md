



```
bf-sde.pipe_mgr> phv-dump 
Usage: -d <dev_id> -p <log_pipe_id> -s <stage_id> -i <direction>
bf-sde.pipe_mgr> dev
------------------------------------------------------------
Device|Type               |#pipe|#stg|#prsr|#macblk|#sub-dev
------|-------------------|-----|----|-----|-------|--------
0     |BFN-T20-128Q      |4    |20  |36   |-3     |1       
bf-sde.pipe_mgr> phv-dump  -d 0 -p 0 
Usage: -d <dev_id> -p <log_pipe_id> -s <stage_id> -i <direction>
bf-sde.pipe_mgr> phv-dump  -d 0 -p 0 -s 0 -i 1

Pipe Stage SnapS Contr[Len] Field[S:E] PHV[S:E] Name
   0     0   Y       84[ 8]       0: 0     1: 1 eg_intr_md_egress_port_valid
   0     0   Y       84[ 8]       0: 0     0: 0 eg_intr_md_for_dprsr_mirror_io_select_valid
   0     0   Y      172[16]       0: 8     0: 8 eg_intr_md_egress_port
   0     0   Y      173[16]       0: 0     0: 0 eg_intr_md_for_dprsr_mirror_io_select

bf-sde.pipe_mgr> phv-dump  -d 0 -p 0 -s 0 -i 0

Pipe Stage SnapS Contr[Len] Field[S:E] PHV[S:E] Name
   0     0   Y       81[ 8]       0: 0     0: 0 hdr_vlan_tag_1_valid
   0     0   Y       81[ 8]       0: 1     0: 1 hdr_vlan_tag__stkvalid
   0     0   Y       82[ 8]       0: 0     2: 2 hdr_ethernet_valid
   0     0   Y       82[ 8]       0: 0     3: 3 hdr_ipv4_valid
   0     0   Y       82[ 8]       0: 0     4: 4 hdr_ipv6_valid
   0     0   Y      160[16]       0: 8     0: 8 ig_intr_md_ingress_port

bf-sde.pipe_mgr> phv-dump  -d 0 -p 0 -s 1 -i 0

Pipe Stage SnapS Contr[Len] Field[S:E] PHV[S:E] Name
   0     1   Y       81[ 8]       0: 0     0: 0 hdr_vlan_tag_1_valid
   0     1   Y       81[ 8]       0: 1     0: 1 hdr_vlan_tag__stkvalid
   0     1   Y       82[ 8]       0: 0     2: 2 hdr_ethernet_valid
   0     1   Y       82[ 8]       0: 0     3: 3 hdr_ipv4_valid
   0     1   Y       82[ 8]       0: 0     4: 4 hdr_ipv6_valid
   0     1   Y       82[ 8]       0: 0     0: 0 ig_intr_md_for_tm_ucast_egress_port_valid
   0     1   N      192[16]       0: 8     0: 8 ig_intr_md_for_tm_ucast_egress_port

bf-sde.pipe_mgr> phv-dump  -d 0 -p 0 -s 2 -i 0

Pipe Stage SnapS Contr[Len] Field[S:E] PHV[S:E] Name
   0     2   Y       81[ 8]       0: 0     0: 0 hdr_vlan_tag_1_valid
   0     2   Y       81[ 8]       0: 1     0: 1 hdr_vlan_tag__stkvalid
   0     2   Y       82[ 8]       0: 0     2: 2 hdr_ethernet_valid
   0     2   Y       82[ 8]       0: 0     3: 3 hdr_ipv4_valid
   0     2   Y       82[ 8]       0: 0     4: 4 hdr_ipv6_valid
   0     2   Y       82[ 8]       0: 0     0: 0 ig_intr_md_for_tm_ucast_egress_port_valid
   0     2   N      192[16]       0: 8     0: 8 ig_intr_md_for_tm_ucast_egress_port

bf-sde.pipe_mgr> phv-dump  -d 0 -p 0 -s 3 -i 0

Pipe Stage SnapS Contr[Len] Field[S:E] PHV[S:E] Name
   0     3   Y       81[ 8]       0: 0     0: 0 hdr_vlan_tag_1_valid
   0     3   Y       81[ 8]       0: 1     0: 1 hdr_vlan_tag__stkvalid
   0     3   Y       82[ 8]       0: 0     2: 2 hdr_ethernet_valid
   0     3   Y       82[ 8]       0: 0     3: 3 hdr_ipv4_valid
   0     3   Y       82[ 8]       0: 0     4: 4 hdr_ipv6_valid
   0     3   Y       82[ 8]       0: 0     0: 0 ig_intr_md_for_tm_ucast_egress_port_valid
   0     3   N      192[16]       0: 8     0: 8 ig_intr_md_for_tm_ucast_egress_port

bf-sde.pipe_mgr> 
```