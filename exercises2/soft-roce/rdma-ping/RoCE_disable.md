```
[root@localhost ~]# mstconfig -d 4b:00.0 q | head -8

Device #1:
----------

Device type:        ConnectX6DX         
Name:               MCX623432AN-ADA_Ax  
Description:        ConnectX-6 Dx EN adapter card; 25GbE for OCP 3.0; with host management; Dual-port SFP28; PCIe 3.0/4.0 x16
Device:             4b:00.0
[root@localhost ~]# mstconfig -d 4b:00.0 q | grep -i roce
        ROCE_NEXT_PROTOCOL                          254                 
        ROCE_CC_LEGACY_DCQCN                        True(1)             
        ROCE_CC_PRIO_MASK_P1                        255                 
        ROCE_CC_PRIO_MASK_P2                        255                 
        ROCE_RTT_RESP_DSCP_P1                       0                   
        ROCE_RTT_RESP_DSCP_MODE_P1                  DEVICE_DEFAULT(0)   
        ROCE_RTT_RESP_DSCP_P2                       0                   
        ROCE_RTT_RESP_DSCP_MODE_P2                  DEVICE_DEFAULT(0)   
        ROCE_CONTROL                                ROCE_ENABLE(2)
```
By default ROCE is enabled.


List of all RoCE related configurations:
```
[root@localhost ~]# mstconfig -d 4b:00.0 i
List of configurations the device 4b:00.0 may support:
(...)
                ROCE 1 5 CONF:
                    ROCE_NEXT_PROTOCOL=<NUM>                The next protocol value set in the IPv4/IPv6 packets for RoCE v1.5.
(...)
                SW OFFLOAD CONF:
(...)
                    ROCE_ADAPTIVE_ROUTING_EN=<False|True>   When TRUE, Adaptive Routing for RDMA QPs is supported.
(...)
                GLOBAL ROCE CC CONF:
                    ROCE_CC_LEGACY_DCQCN=<False|True>       When TRUE, the device will only use legacy Congestion Control DCQCN algorithm
(...)
                ROCE CC:
                    IB_CC_SHAPER_COALESCE_P1=<DEVICE_DEFAULT|REMOTE_LID>Select CC algorithm shaper coalescing for IB
                                                            0x0: DEVICE_DEFAULT
                                                            0x2: REMOTE_LID - shaper is selected according to remote lid (IB)
                                                            other values are reserved
                    IB_CC_SHAPER_COALESCE_P2=<DEVICE_DEFAULT|REMOTE_LID>
                    ROCE_CC_ALGORITHM_P1=<ECN|QCN>          Select RDMA over Converged Ethernet (RoCE) algorithm
                                                            0x0: ECN
                                                            0x1: QCN
                    ROCE_CC_ALGORITHM_P2=<ECN|QCN>          
                    ROCE_CC_PRIO_MASK_P1=<NUM>              Each bit in this mask indicates if the RoCE should be enabled on the n-th IEEE priority.
                    ROCE_CC_PRIO_MASK_P2=<NUM>              
                    ROCE_CC_SHAPER_COALESCE_P1=<DEST_IP|DEVICE_DEFAULT|SOURCE_QP|_5_TUPLE>Select CC algorithm shaper coalescing for ROCE
                                                            0x0: DEVICE_DEFAULT
                                                            0x1: DEST_IP - shaper is selected according to dest IP
                                                            0x2: SOURCE_QP - shaper is selected according to source QP
                                                            0x3: _5_TUPLE - shaper is selected according to 5-tuples
                                                            other values are reserved
                    ROCE_CC_SHAPER_COALESCE_P2=<DEST_IP|DEVICE_DEFAULT|SOURCE_QP|_5_TUPLE>
(...)
                ROCE CC ECN:
                    CLAMP_TGT_RATE_AFTER_TIME_INC_P1=<False|True>When receiving a CNP, the target rate should be updated if the transmission rate was increased due to the timer, and not only due to the byte counter
                    CLAMP_TGT_RATE_AFTER_TIME_INC_P2=<False|True>
                    CLAMP_TGT_RATE_P1=<False|True>          If set, whenever a CNP is processed, the target rate is updated to be the current rate.
                    CLAMP_TGT_RATE_P2=<False|True>          
                    CNP_802P_PRIO_P1=<NUM>                  The 802.1p priority value of the generated CNP for this port
                    CNP_802P_PRIO_P2=<NUM>                  
                    CNP_DSCP_P1=<NUM>                       The DiffServ Code Point of the generated CNP for this port.
                    CNP_DSCP_P2=<NUM>                       
                    CNP_RES_PRIO_MODE_P1=<False|True>       If TRUE, CNP packets for this port contain priority from a received request. If FALSE, CNP responses use value set by CNP_802P_PRIO.
                    CNP_RES_PRIO_MODE_P2=<False|True>       
                    DCE_TCP_G_P1=<NUM>                      Used to update the congestion estimator (alpha) once every dce_tcp_rtt microseconds, according to the equation:
                                                            Alpha = (cnp_received * dceTcpG) + (1 - dceTcpG) * alpha .
                                                            dceTcpG is divided by 2^10.
                                                            cnp_received is set to one if a CNP was received for this flow during period since the previous update and the current update
                    DCE_TCP_G_P2=<NUM>                      
                    DCE_TCP_RTT_P1=<NUM>                    The time between updates of the alpha value, in microseconds.
                    DCE_TCP_RTT_P2=<NUM>                    
                    INITIAL_ALPHA_VALUE_P1=<NUM>            The initial value of alpha to use when receiving the first CNP for a flow. Expressed in a fixed point fraction of 2^10.
                    INITIAL_ALPHA_VALUE_P2=<NUM>            
                    MIN_TIME_BETWEEN_CNPS_P1=<NUM>          Minimum time between sending CNPs from the port, in microseconds.
                    MIN_TIME_BETWEEN_CNPS_P2=<NUM>          
                    RATE_REDUCE_MONITOR_PERIOD_P1=<NUM>     The minimum time between 2 consecutive rate reductions for a single flow. Rate reduction will occur only if a CNP is received during the relevant time interval.
                    RATE_REDUCE_MONITOR_PERIOD_P2=<NUM>     
                    RATE_TO_SET_ON_FIRST_CNP_P1=<NUM>       The rate that is set for the flow when a rate limiter is allocated to it upon first CNP received, in Mbps (=Full Port Speed).
                    RATE_TO_SET_ON_FIRST_CNP_P2=<NUM>       
                    RPG_AI_RATE_P1=<NUM>                    The rate, in megabits per second, used to increase rpTargetRate in the RPR_ACTIVE_INCREASE.
                    RPG_AI_RATE_P2=<NUM>                    
                    RPG_BYTE_RESET_P1=<NUM>                 Transmitted data between rate increases if no CNPs are received. Given in Bytes (0=DISABLED)
                    RPG_BYTE_RESET_P2=<NUM>                 
                    RPG_GD_P1=<NUM>                         If a CNP is received, the flow rate is reduced at the beginning of the next rate_reduce_monitor_period interval to (1-Alpha/Gd)*CurrentRate. rpg_gd is given as log2(Gd), where Gd may only be powers of 2.
                    RPG_GD_P2=<NUM>                         
                    RPG_HAI_RATE_P1=<NUM>                   The rate, in megabits per second, used to increase rpTargetRate in the RPR_HYPER_INCREASE state.
                    RPG_HAI_RATE_P2=<NUM>                   
                    RPG_MAX_RATE_P1=<NUM>                   The maximum rate, in Mbits per second, at which an RP can transmit. Once this limit is reached, the RP rate limited is released and the flow is not rate limited any more (0=Full Port Speed).
                    RPG_MAX_RATE_P2=<NUM>                   
                    RPG_MIN_DEC_FAC_P1=<NUM>                The minimum factor by which the current transmit rate can be changed when processing a CNP. Value is given as a percentage (1-100).
                    RPG_MIN_DEC_FAC_P2=<NUM>                
                    RPG_MIN_RATE_P1=<NUM>                   The minimum value, in megabits per second, for rate to limit.
                    RPG_MIN_RATE_P2=<NUM>                   
                    RPG_THRESHOLD_P1=<NUM>                  The number of times rpByteStage or rpTimeStage can count before the RP rate control state machine advances states.
                    RPG_THRESHOLD_P2=<NUM>                  
                    RPG_TIME_RESET_P1=<NUM>                 Time between rate increases if no CNPs are received. Given in microseconds.
                    RPG_TIME_RESET_P2=<NUM>
(...)
                ROCE CONF:
                    ROCE_RTT_RESP_DSCP_MODE_P1=<DEVICE_DEFAULT|FIXED_VALUE|RTT_REQUEST>Defines the method for setting IP.DSCP in RTT response packets
                                                            0x0: DEVICE_DEFAULT
                                                            0x1: FIXED_VALUE - taken from ROCE_RTT_RESP_DSCP
                                                            0x2: RTT_REQUEST - taken from the RTT request
                                                            other values are reserved
                    ROCE_RTT_RESP_DSCP_MODE_P2=<DEVICE_DEFAULT|FIXED_VALUE|RTT_REQUEST>
                    ROCE_RTT_RESP_DSCP_P1=<NUM>             The DiffServ Code Point of the generated RTT response for this port. If not set, RTT request value will be used. Overrides PCC_INT_NP_RTT_DSCP
                    ROCE_RTT_RESP_DSCP_P2=<NUM>
(...)
                HCA CONF:
(...)
                    ROCE_CONTROL=<DEVICE_DEFAULT|ROCE_DISABLE|ROCE_ENABLE>Control support for RDMA over Converged Ethernet (RoCE)
                                                            0x0: DEVICE_DEFAULT
                                                            0x1: ROCE_DISABLE
                                                            0x2: ROCE_ENABLE
```

### Disabling RoCE
Configuring the card:
```
[root@localhost ~]# mstconfig -d 4b:00.0 set ROCE_CONTROL=ROCE_DISABLE

Device #1:
----------

Device type:        ConnectX6DX         
Name:               MCX623432AN-ADA_Ax  
Description:        ConnectX-6 Dx EN adapter card; 25GbE for OCP 3.0; with host management; Dual-port SFP28; PCIe 3.0/4.0 x16
Device:             4b:00.0             

Configurations:                                     Next Boot       New
        ROCE_CONTROL                                ROCE_ENABLE(2)       ROCE_DISABLE(1)     

 Apply new Configuration? (y/n) [n] : y
Applying... Done!
-I- Please reboot machine to load new configurations.
```
Reseting the card:
```
oot@localhost ~]# mstfwreset reset -d 4b:00.0  -y -l 4

Requested reset level for device, 4b:00.0:

4: Warm Reboot
Continue with reset?[y/N] y
-I- Sending Reset Command To Fw             -Done
-I- Sending reboot command to machine       -Done
```
Validate the change:
```
[root@localhost ~]# mstconfig -d 4b:00.0 q | grep -i roce
        ROCE_NEXT_PROTOCOL                          254                 
        ROCE_CC_LEGACY_DCQCN                        True(1)             
        ROCE_CC_PRIO_MASK_P1                        255                 
        ROCE_CC_PRIO_MASK_P2                        255                 
        ROCE_RTT_RESP_DSCP_P1                       0                   
        ROCE_RTT_RESP_DSCP_MODE_P1                  DEVICE_DEFAULT(0)   
        ROCE_RTT_RESP_DSCP_P2                       0                   
        ROCE_RTT_RESP_DSCP_MODE_P2                  DEVICE_DEFAULT(0)   
        ROCE_CONTROL                                ROCE_DISABLE(1)
```

### Full dump
```
[root@localhost ~]# mstconfig -d 4b:00.0 q

Device #1:
----------

Device type:        ConnectX6DX         
Name:               MCX623432AN-ADA_Ax  
Description:        ConnectX-6 Dx EN adapter card; 25GbE for OCP 3.0; with host management; Dual-port SFP28; PCIe 3.0/4.0 x16
Device:             4b:00.0             

Configurations:                                     Next Boot
        MEMIC_BAR_SIZE                              0                   
        MEMIC_SIZE_LIMIT                            _256KB(1)           
        HOST_CHAINING_MODE                          DISABLED(0)         
        HOST_CHAINING_CACHE_DISABLE                 False(0)            
        HOST_CHAINING_DESCRIPTORS                   Array[0..7]         
        HOST_CHAINING_TOTAL_BUFFER_SIZE             Array[0..7]         
        FLEX_PARSER_PROFILE_ENABLE                  0                   
        PROG_PARSE_GRAPH                            False(0)            
        FLEX_IPV4_OVER_VXLAN_PORT                   0                   
        ROCE_NEXT_PROTOCOL                          254                 
        ESWITCH_HAIRPIN_DESCRIPTORS                 Array[0..7]         
        ESWITCH_HAIRPIN_TOT_BUFFER_SIZE             Array[0..7]         
        PF_BAR2_SIZE                                0                   
        PF_NUM_OF_VF_VALID                          False(0)            
        NON_PREFETCHABLE_PF_BAR                     False(0)            
        VF_VPD_ENABLE                               False(0)            
        PF_NUM_PF_MSIX_VALID                        False(0)            
        PER_PF_NUM_SF                               False(0)            
        STRICT_VF_MSIX_NUM                          False(0)            
        VF_NODNIC_ENABLE                            False(0)            
        NUM_PF_MSIX_VALID                           True(1)             
        NUM_OF_VFS                                  0                   
        NUM_OF_PF                                   2                   
        PF_BAR2_ENABLE                              False(0)            
        SRIOV_EN                                    False(0)            
        PF_LOG_BAR_SIZE                             5                   
        VF_LOG_BAR_SIZE                             0                   
        NUM_PF_MSIX                                 63                  
        NUM_VF_MSIX                                 11                  
        INT_LOG_MAX_PAYLOAD_SIZE                    AUTOMATIC(0)        
        PCIE_CREDIT_TOKEN_TIMEOUT                   0                   
        LAG_RESOURCE_ALLOCATION                     DEVICE_DEFAULT(0)   
        PHY_COUNT_LINK_UP_DELAY                     DELAY_NONE(0)       
        ACCURATE_TX_SCHEDULER                       False(0)            
        PARTIAL_RESET_EN                            False(0)            
        RESET_WITH_HOST_ON_ERRORS                   False(0)            
        PCI_SWITCH_EMULATION_NUM_PORT               16                  
        PCI_SWITCH_EMULATION_ENABLE                 False(0)            
        PCI_DOWNSTREAM_PORT_OWNER                   Array[0..15]        
        CQE_COMPRESSION                             BALANCED(0)         
        IP_OVER_VXLAN_EN                            False(0)            
        MKEY_BY_NAME                                False(0)            
        PRIO_TAG_REQUIRED_EN                        False(0)            
        UCTX_EN                                     True(1)             
        REAL_TIME_CLOCK_ENABLE                      False(0)            
        RDMA_SELECTIVE_REPEAT_EN                    False(0)            
        PCI_ATOMIC_MODE                             PCI_ATOMIC_DISABLED_EXT_ATOMIC_ENABLED(0)
        TUNNEL_ECN_COPY_DISABLE                     False(0)            
        LRO_LOG_TIMEOUT0                            6                   
        LRO_LOG_TIMEOUT1                            7                   
        LRO_LOG_TIMEOUT2                            8                   
        LRO_LOG_TIMEOUT3                            13                  
        LOG_TX_PSN_WINDOW                           7                   
        LOG_MAX_OUTSTANDING_WQE                     7                   
        TUNNEL_IP_PROTO_ENTROPY_DISABLE             False(0)            
        ICM_CACHE_MODE                              DEVICE_DEFAULT(0)   
        TLS_OPTIMIZE                                False(0)            
        TX_SCHEDULER_BURST                          0                   
        ZERO_TOUCH_TUNING_ENABLE                    False(0)            
        ROCE_CC_LEGACY_DCQCN                        True(1)             
        LOG_MAX_QUEUE                               17                  
        LOG_DCR_HASH_TABLE_SIZE                     11                  
        MAX_PACKET_LIFETIME                         0                   
        DCR_LIFO_SIZE                               16384               
        ROCE_CC_PRIO_MASK_P1                        255                 
        ROCE_CC_PRIO_MASK_P2                        255                 
        CLAMP_TGT_RATE_AFTER_TIME_INC_P1            True(1)             
        CLAMP_TGT_RATE_P1                           False(0)            
        RPG_TIME_RESET_P1                           300                 
        RPG_BYTE_RESET_P1                           32767               
        RPG_THRESHOLD_P1                            1                   
        RPG_MAX_RATE_P1                             0                   
        RPG_AI_RATE_P1                              5                   
        RPG_HAI_RATE_P1                             50                  
        RPG_GD_P1                                   11                  
        RPG_MIN_DEC_FAC_P1                          50                  
        RPG_MIN_RATE_P1                             1                   
        RATE_TO_SET_ON_FIRST_CNP_P1                 0                   
        DCE_TCP_G_P1                                1019                
        DCE_TCP_RTT_P1                              1                   
        RATE_REDUCE_MONITOR_PERIOD_P1               4                   
        INITIAL_ALPHA_VALUE_P1                      1023                
        MIN_TIME_BETWEEN_CNPS_P1                    4                   
        CNP_802P_PRIO_P1                            6                   
        CNP_DSCP_P1                                 48                  
        CLAMP_TGT_RATE_AFTER_TIME_INC_P2            True(1)             
        CLAMP_TGT_RATE_P2                           False(0)            
        RPG_TIME_RESET_P2                           300                 
        RPG_BYTE_RESET_P2                           32767               
        RPG_THRESHOLD_P2                            1                   
        RPG_MAX_RATE_P2                             0                   
        RPG_AI_RATE_P2                              5                   
        RPG_HAI_RATE_P2                             50                  
        RPG_GD_P2                                   11                  
        RPG_MIN_DEC_FAC_P2                          50                  
        RPG_MIN_RATE_P2                             1                   
        RATE_TO_SET_ON_FIRST_CNP_P2                 0                   
        DCE_TCP_G_P2                                1019                
        DCE_TCP_RTT_P2                              1                   
        RATE_REDUCE_MONITOR_PERIOD_P2               4                   
        INITIAL_ALPHA_VALUE_P2                      1023                
        MIN_TIME_BETWEEN_CNPS_P2                    4                   
        CNP_802P_PRIO_P2                            6                   
        CNP_DSCP_P2                                 48                  
        LLDP_NB_DCBX_P1                             False(0)            
        LLDP_NB_RX_MODE_P1                          OFF(0)              
        LLDP_NB_TX_MODE_P1                          OFF(0)              
        LLDP_NB_DCBX_P2                             False(0)            
        LLDP_NB_RX_MODE_P2                          OFF(0)              
        LLDP_NB_TX_MODE_P2                          OFF(0)              
        ROCE_RTT_RESP_DSCP_P1                       0                   
        ROCE_RTT_RESP_DSCP_MODE_P1                  DEVICE_DEFAULT(0)   
        ROCE_RTT_RESP_DSCP_P2                       0                   
        ROCE_RTT_RESP_DSCP_MODE_P2                  DEVICE_DEFAULT(0)   
        DCBX_IEEE_P1                                True(1)             
        DCBX_CEE_P1                                 True(1)             
        DCBX_WILLING_P1                             True(1)             
        DCBX_IEEE_P2                                True(1)             
        DCBX_CEE_P2                                 True(1)             
        DCBX_WILLING_P2                             True(1)             
        KEEP_ETH_LINK_UP_P1                         True(1)             
        KEEP_IB_LINK_UP_P1                          False(0)            
        KEEP_LINK_UP_ON_BOOT_P1                     False(0)            
        KEEP_LINK_UP_ON_STANDBY_P1                  False(0)            
        DO_NOT_CLEAR_PORT_STATS_P1                  False(0)            
        AUTO_POWER_SAVE_LINK_DOWN_P1                False(0)            
        KEEP_ETH_LINK_UP_P2                         True(1)             
        KEEP_IB_LINK_UP_P2                          False(0)            
        KEEP_LINK_UP_ON_BOOT_P2                     False(0)            
        KEEP_LINK_UP_ON_STANDBY_P2                  False(0)            
        DO_NOT_CLEAR_PORT_STATS_P2                  False(0)            
        AUTO_POWER_SAVE_LINK_DOWN_P2                False(0)            
        NUM_OF_VL_P1                                _4_VLs(3)           
        NUM_OF_TC_P1                                _8_TCs(0)           
        NUM_OF_PFC_P1                               8                   
        VL15_BUFFER_SIZE_P1                         0                   
        NUM_OF_VL_P2                                _4_VLs(3)           
        NUM_OF_TC_P2                                _8_TCs(0)           
        NUM_OF_PFC_P2                               8                   
        VL15_BUFFER_SIZE_P2                         0                   
        DUP_MAC_ACTION_P1                           LAST_CFG(0)         
        MPFS_MC_LOOPBACK_DISABLE_P1                 False(0)            
        MPFS_UC_LOOPBACK_DISABLE_P1                 False(0)            
        UNKNOWN_UPLINK_MAC_FLOOD_P1                 False(0)            
        SRIOV_IB_ROUTING_MODE_P1                    LID(1)              
        IB_ROUTING_MODE_P1                          LID(1)              
        DUP_MAC_ACTION_P2                           LAST_CFG(0)         
        MPFS_MC_LOOPBACK_DISABLE_P2                 False(0)            
        MPFS_UC_LOOPBACK_DISABLE_P2                 False(0)            
        UNKNOWN_UPLINK_MAC_FLOOD_P2                 False(0)            
        SRIOV_IB_ROUTING_MODE_P2                    LID(1)              
        IB_ROUTING_MODE_P2                          LID(1)              
        PHY_FEC_OVERRIDE_P1                         DEVICE_DEFAULT(0)   
        PHY_FEC_OVERRIDE_P2                         DEVICE_DEFAULT(0)   
        WOL_MAGIC_EN                                True(1)             
        PF_TOTAL_SF                                 0                   
        PF_SF_BAR_SIZE                              0                   
        PF_NUM_PF_MSIX                              63                  
        ROCE_CONTROL                                ROCE_ENABLE(2)      
        PCI_WR_ORDERING                             per_mkey(0)         
        MULTI_PORT_VHCA_EN                          False(0)            
        PORT_OWNER                                  True(1)             
        ALLOW_RD_COUNTERS                           True(1)             
        RENEG_ON_CHANGE                             True(1)             
        TRACER_ENABLE                               True(1)             
        IP_VER                                      IPv4(0)             
        BOOT_UNDI_NETWORK_WAIT                      0                   
        UEFI_HII_EN                                 True(1)             
        BOOT_DBG_LOG                                False(0)            
        UEFI_LOGS                                   DISABLED(0)         
        BOOT_VLAN                                   1                   
        LEGACY_BOOT_PROTOCOL                        PXE(1)              
        BOOT_INTERRUPT_DIS                          False(0)            
        BOOT_LACP_DIS                               True(1)             
        BOOT_VLAN_EN                                False(0)            
        BOOT_PKEY                                   0                   
        P2P_ORDERING_MODE                           DEVICE_DEFAULT(0)   
        ATS_ENABLED                                 False(0)            
        DYNAMIC_VF_MSIX_TABLE                       False(0)            
        EXP_ROM_UEFI_ARM_ENABLE                     True(1)             
        EXP_ROM_UEFI_x86_ENABLE                     True(1)             
        EXP_ROM_PXE_ENABLE                          True(1)             
        ADVANCED_PCI_SETTINGS                       False(0)            
        SAFE_MODE_THRESHOLD                         10                  
        SAFE_MODE_ENABLE                            True(1)
```
