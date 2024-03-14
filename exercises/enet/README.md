ENETC is a multi-port virtualized Ethernet controller supporting GbE
designs and Time-Sensitive Networking (TSN) functionality.
ENETC is operating as an SR-IOV multi-PF capable Root Complex Integrated
Endpoint (RCIE).  As such, it contains multiple physical (PF) and virtual
(VF) PCIe functions, discoverable by standard PCI Express.

# tsn

drivers/net/ethernet/freescale/enetc/enetc_tsn.c (for ENETC) 
drivers/net/ethernet/mscc/tsn_switch.c (for Felix) in the kernel side.

```
+#define ENETC_MMCSR_LPA		BIT(2) /* Local Preemption Active */
+#define ENETC_MMCSR_LPE		BIT(1) /* Local Preemption Enabled */
+#define ENETC_MMCSR_LPS		BIT(0) /* Local Preemption Supported *
+#define ENETC_MMCSR_VDIS	BIT(17) /* Verify Disabled */
+#define ENETC_MMCSR_ME		BIT(16) /* Merge Enabled */
```


##  ENETC_MMCSR_ME

```
         
static void enetc_configure_port_pmac(struct enetc_hw *hw)
{        
        u32 temp;

        /* Set pMAC step lock */
        temp = enetc_port_rd(hw, ENETC_PFPMR);
        enetc_port_wr(hw, ENETC_PFPMR, 
                      temp | ENETC_PFPMR_PMACE | ENETC_PFPMR_MWLM);
    
        temp = enetc_port_rd(hw, ENETC_MMCSR);
        enetc_port_wr(hw, ENETC_MMCSR, temp | ENETC_MMCSR_ME);
}   
   
```