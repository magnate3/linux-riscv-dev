
# print_hwif_attr
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/igb-uio/pics/igb_read_attr.png)
```
#define rte_bswap32(x) __builtin_bswap32(x)
#define rte_be_to_cpu_32(x) rte_bswap32(x)
static inline u32 hinic_hwif_read_reg(void __iomem * bar, u32 reg)
{
        return be32_to_cpu(readl(bar + reg));
}
static void print_hwif_attr(void __iomem * bar)
{
        u32 addr, attr0, attr1;

        addr   = HINIC_CSR_FUNC_ATTR0_ADDR;
        attr0  = hinic_hwif_read_reg(bar, addr);

        addr   = HINIC_CSR_FUNC_ATTR1_ADDR;
        attr1  = hinic_hwif_read_reg(bar, addr);
        pr_info("attr0 %8x, and attr1 %8x \n",  attr0, attr1);

}
```