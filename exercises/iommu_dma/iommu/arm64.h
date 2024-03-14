#ifndef __ARM64_IOMMU__H
#define  __ARM64_IOMMU__H
typedef u64 arm_lpae_iopte;

#define ARM_LPAE_MAX_ADDR_BITS              48
#define ARM_LPAE_PTE_TYPE_SHIFT             0
#define ARM_LPAE_PTE_TYPE_MASK              0x3
#define ARM_LPAE_PTE_TYPE_PAGE              3
#define ARM_LPAE_PTE_TYPE_BLOCK             1
/* Register bits */
#define ARM_32_LPAE_TCR_EAE             (1 << 31)
#define ARM_64_LPAE_S2_TCR_RES1         (1 << 31)

#define ARM_LPAE_TCR_EPD1               (1 << 23)

#define ARM_LPAE_TCR_TG0_4K             (0 << 14)
#define ARM_LPAE_TCR_TG0_64K            (1 << 14)
#define ARM_LPAE_TCR_TG0_16K            (2 << 14)

#define ARM_LPAE_TCR_SH0_SHIFT          12
#define ARM_LPAE_TCR_SH0_MASK           0x3
#define ARM_LPAE_TCR_SH_NS              0
#define ARM_LPAE_TCR_SH_OS              2
#define ARM_LPAE_TCR_SH_IS              3

#define ARM_LPAE_TCR_ORGN0_SHIFT        10
#define ARM_LPAE_TCR_IRGN0_SHIFT        8
#define ARM_LPAE_TCR_RGN_MASK           0x3
#define ARM_LPAE_TCR_RGN_NC             0
#define ARM_LPAE_TCR_RGN_WBWA           1
#define ARM_LPAE_TCR_RGN_WT             2
#define ARM_LPAE_TCR_RGN_WB             3

#define ARM_LPAE_TCR_SL0_SHIFT          6
#define ARM_LPAE_TCR_SL0_MASK           0x3

#define ARM_LPAE_TCR_T0SZ_SHIFT         0
#define ARM_LPAE_TCR_SZ_MASK            0xf

#define ARM_LPAE_TCR_PS_SHIFT           16
#define ARM_LPAE_TCR_PS_MASK            0x7

#define ARM_LPAE_TCR_IPS_SHIFT          32
#define ARM_LPAE_TCR_IPS_MASK           0x7

#define ARM_LPAE_TCR_PS_32_BIT          0x0ULL
#define ARM_LPAE_TCR_PS_36_BIT          0x1ULL
#define ARM_LPAE_TCR_PS_40_BIT          0x2ULL
#define ARM_LPAE_TCR_PS_42_BIT          0x3ULL
#define ARM_LPAE_TCR_PS_44_BIT          0x4ULL
#define ARM_LPAE_TCR_PS_48_BIT          0x5ULL

#define ARM_LPAE_MAIR_ATTR_SHIFT(n)     ((n) << 3)
#define ARM_LPAE_MAIR_ATTR_MASK         0xff
#define ARM_LPAE_MAIR_ATTR_DEVICE       0x04

#define ARM_LPAE_MAX_LEVELS             4
#define ARM_LPAE_START_LVL(d)           (ARM_LPAE_MAX_LEVELS - (d)->levels)
#define ARM_LPAE_GRANULE(d)             (1UL << (d)->pg_shift)



/*
 * Calculate the right shift amount to get to the portion describing level l
 * in a virtual address mapped by the pagetable in d.
 */
#define ARM_LPAE_LVL_SHIFT(l,d)						\
	((((d)->levels - ((l) - ARM_LPAE_START_LVL(d) + 1))		\
	  * (d)->bits_per_level) + (d)->pg_shift)
#define ARM_LPAE_PAGES_PER_PGD(d)					\
	DIV_ROUND_UP((d)->pgd_size, ARM_LPAE_GRANULE(d))

/*
 * Calculate the index at level l used to map virtual address a using the
 * pagetable in d.
 */
#define ARM_LPAE_PGD_IDX(l,d)						\
	((l) == ARM_LPAE_START_LVL(d) ? ilog2(ARM_LPAE_PAGES_PER_PGD(d)) : 0)

#define ARM_LPAE_LVL_IDX(a,l,d)						\
	(((u64)(a) >> ARM_LPAE_LVL_SHIFT(l,d)) &			\
	 ((1 << ((d)->bits_per_level + ARM_LPAE_PGD_IDX(l,d))) - 1))

/* Calculate the block/page mapping size at level l for pagetable in d. */
#define ARM_LPAE_BLOCK_SIZE(l,d)					\
	(1ULL << (ilog2(sizeof(arm_lpae_iopte)) +			\
		((ARM_LPAE_MAX_LEVELS - (l)) * (d)->bits_per_level)))

/* IOPTE accessors */
#define iopte_deref(pte,d)					\
	(__va((pte) & ((1ULL << ARM_LPAE_MAX_ADDR_BITS) - 1)	\
	& ~(ARM_LPAE_GRANULE(d) - 1ULL)))

#define iopte_type(pte,l)					\
	(((pte) >> ARM_LPAE_PTE_TYPE_SHIFT) & ARM_LPAE_PTE_TYPE_MASK)

#define iopte_prot(pte)	((pte) & ARM_LPAE_PTE_ATTR_MASK)

#define iopte_leaf(pte,l)					\
	(l == (ARM_LPAE_MAX_LEVELS - 1) ?			\
		(iopte_type(pte,l) == ARM_LPAE_PTE_TYPE_PAGE) :	\
		(iopte_type(pte,l) == ARM_LPAE_PTE_TYPE_BLOCK))

#define iopte_to_pfn(pte,d)					\
	(((pte) & ((1ULL << ARM_LPAE_MAX_ADDR_BITS) - 1)) >> (d)->pg_shift)

#define pfn_to_iopte(pfn,d)					\
	(((pfn) << (d)->pg_shift) & ((1ULL << ARM_LPAE_MAX_ADDR_BITS) - 1))

#endif
