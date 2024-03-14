#include <linux/module.h>
#include <linux/slab.h>
#include <linux/init.h>


static int set_bits_ll(unsigned long *addr, unsigned long mask_to_set)
    {
        unsigned long val, nval;
    
        nval = *addr;
        do {
            val = nval;
            if (val & mask_to_set)
                return -EBUSY;
            cpu_relax();
        } while ((nval = cmpxchg(addr, val, val | mask_to_set)) != val);
    
        return 0;
    }
    
    static int clear_bits_ll(unsigned long *addr, unsigned long mask_to_clear)
    {
        unsigned long val, nval;
    
        nval = *addr;
        do {
            val = nval;
            if ((val & mask_to_clear) != mask_to_clear)
                return -EBUSY;
            cpu_relax();
        } while ((nval = cmpxchg(addr, val, val & ~mask_to_clear)) != val);
    
        return 0;
    }
    
    /*
     * bitmap_set_ll - set the specified number of bits at the specified position
     * @map: pointer to a bitmap
     * @start: a bit position in @map
     * @nr: number of bits to set
     *
     * Set @nr bits start from @start in @map lock-lessly. Several users
     * can set/clear the same bitmap simultaneously without lock. If two
     * users set the same bit, one user will return remain bits, otherwise
     * return 0.
     */
    static int bitmap_set_ll(unsigned long *map, int start, int nr)
    {
        unsigned long *p = map + BIT_WORD(start);
        const int size = start + nr;
        int bits_to_set = BITS_PER_LONG - (start % BITS_PER_LONG);
        unsigned long mask_to_set = BITMAP_FIRST_WORD_MASK(start);
    
        while (nr - bits_to_set >= 0) {
            if (set_bits_ll(p, mask_to_set))
                return nr;
            nr -= bits_to_set;
            bits_to_set = BITS_PER_LONG;
            mask_to_set = ~0UL;
            p++;
        }
        if (nr) {
            mask_to_set &= BITMAP_LAST_WORD_MASK(size);
            if (set_bits_ll(p, mask_to_set))
                return nr;
        }
    
       return 0;
  }
static int __init my_init(void)
{
#if 1
    unsigned long bits[2];
    unsigned long bitmap = 0;
    printk("test begin >>>>>>>>>>>>>>>>>>> \n");
    printk(" BITS_PER_LONG %d\n", BITS_PER_LONG);
    printk("Bitmap(0):   %#lx\n", BITMAP_FIRST_WORD_MASK(0));
    printk("Bitmap(1):   %#lx\n", BITMAP_FIRST_WORD_MASK(1));
    printk("Bitmap(2):   %#lx\n", BITMAP_FIRST_WORD_MASK(2));
    printk("Bitmap(3):   %#lx\n", BITMAP_FIRST_WORD_MASK(3));
    memset(bits, 0, sizeof(bits));
    printk("before set %#lx, %#lx\n", bits[0], bits[1]);
    /* set special bits */
    bitmap_set_ll(bits, 4, 64);
    printk("after set %#lx, %#lx\n", bits[0], bits[1]);
    set_bits_ll(&bitmap, 0xfff);
    printk("after set bits all  %#lx\n", bitmap);
    clear_bits_ll(&bitmap, 0xfff);
    printk("after clear set bits all  %#lx\n", bitmap);
#else
       
       	unsigned long bits[2];
    	printk("test begin >>>>>>>>>>>>>>>>>>> \n");
        memset(bits, 0, sizeof(bits));
	printk("before set %#lx, %#lx\n", bits[0], bits[1]);
	/* set special bits */
	__bitmap_set(bits, 4, 64);
	printk("after set %#lx, %#lx\n", bits[0], bits[1]);
#endif
    return 0;
}

static void __exit my_exit(void)
{
}

module_init(my_init);
module_exit(my_exit);

MODULE_AUTHOR("Jerry Cooperstein");
MODULE_DESCRIPTION("LF331:1.6 s_18/lab8_uio_api.c");
MODULE_LICENSE("GPL v2");
