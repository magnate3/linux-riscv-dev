# hinic_hw_if.c
```
static int
print_bars(struct pci_dev *pdev)
{
        int i, iom, iop;
        unsigned long flags;
        unsigned long addr, len;
        static const char *bar_names[PCI_STD_RESOURCE_END + 1]  = {
                "BAR0",
                "BAR1",
                "BAR2",
                "BAR3",
                "BAR4",
                "BAR5",
        };

        iom = 0;
        iop = 0;

        for (i = 0; i < ARRAY_SIZE(bar_names); i++) {
             pr_err("******bar name : %s", bar_names[i]);
                addr = pci_resource_start(pdev, i);
                len = pci_resource_len(pdev, i);
                if (len != 0 && addr != 0) {
                        flags = pci_resource_flags(pdev, i);
                        if (flags & IORESOURCE_MEM) {
                                iom++;
                        } else if (flags & IORESOURCE_IO) {
                                iop++;
                        }
                        pr_info("flags  %lx,  and addr %lx, and len %lx \n", flags & IORESOURCE_MEM,  addr, len);
                }
        }
        return 0;
}
```

#  bar
```
void __iomem *regs = pci_ioremap_bar(device, barNumber);
```
```
#define HINIC_PCI_CFG_REGS_BAR          0
#define HINIC_PCI_DB_BAR                4
func_to_io->db_base = pci_ioremap_bar(pdev, HINIC_PCI_DB_BAR);
hwif->cfg_regs_bar = pci_ioremap_bar(pdev, HINIC_PCI_CFG_REGS_BAR);
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hinic/pic/res.png)

##  cfg_regs_bar
```
static inline u32 hinic_hwif_read_reg(struct hinic_hwif *hwif, u32 reg)
{
        return be32_to_cpu(readl(hwif->cfg_regs_bar + reg));
}

static inline void hinic_hwif_write_reg(struct hinic_hwif *hwif, u32 reg,
                                        u32 val)
{
        writel(cpu_to_be32(val), hwif->cfg_regs_bar + reg);
}
   
```

### hinic_hwdev_msix_set

```
hinic_hwdev_msix_set -->  hinic_msix_attr_set
int hinic_msix_attr_set(struct hinic_hwif *hwif, u16 msix_index,
                        u8 pending_limit, u8 coalesc_timer,
                        u8 lli_timer, u8 lli_credit_limit,
                        u8 resend_timer)
{
        u32 msix_ctrl, addr;

        if (!VALID_MSIX_IDX(&hwif->attr, msix_index))
                return -EINVAL;

        msix_ctrl = HINIC_MSIX_ATTR_SET(pending_limit, PENDING_LIMIT)   |
                    HINIC_MSIX_ATTR_SET(coalesc_timer, COALESC_TIMER)   |
                    HINIC_MSIX_ATTR_SET(lli_timer, LLI_TIMER)           |
                    HINIC_MSIX_ATTR_SET(lli_credit_limit, LLI_CREDIT)   |
                    HINIC_MSIX_ATTR_SET(resend_timer, RESEND_TIMER);

        addr = HINIC_CSR_MSIX_CTRL_ADDR(msix_index);

        hinic_hwif_write_reg(hwif, addr, msix_ctrl);
        return 0;
}
```
# ceqe

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hinic/pic/ceqe.png)

# print_hwif_attr
```
static void print_hwif_attr(struct hinic_hwif *hwif)
{
        u32 addr, attr0, attr1;

        addr   = HINIC_CSR_FUNC_ATTR0_ADDR;
        attr0  = hinic_hwif_read_reg(hwif, addr);

        addr   = HINIC_CSR_FUNC_ATTR1_ADDR;
        attr1  = hinic_hwif_read_reg(hwif, addr);
        pr_info("in hinic driver attr0 %8x, and attr1 %8x \n",  attr0, attr1);

}
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hinic/pic/hinic_read_attr.png)

# e1000_probe

```

//将PCI的bar0寄存器映射到主存
hw->hw_addr = pci_ioremap_bar(pdev, BAR_0);
//一次申请并初始化其他5个bar的资源

for (i = BAR_1; i <= BAR_5; i++) {
    if (pci_resource_len(pdev, i) == 0)
        continue;
    if (pci_resource_flags(pdev, i) & IORESOURCE_IO) {
            hw->io_base = pci_resource_start(pdev, i);
            break;
    }
```