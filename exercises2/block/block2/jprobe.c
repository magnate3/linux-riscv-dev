#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/bio.h>

static void submit_bio_probe(int rw, struct bio * bio) {
        if(bio && bio->bi_io_vec != NULL) {
                char b[BDEVNAME_SIZE];
		if (MAJOR(bio->bi_bdev->bd_dev) == 179){
		        printk(KERN_INFO "device: %s, command: %s, start: %10lld, count: %d , nr_vecs: %d\n",
                        	bdevname(bio->bi_bdev, b), rw & WRITE ? "write" : "read",
                                	bio->bi_sector, bio_sectors(bio), bio->bi_max_vecs);	
		}

        }
        jprobe_return();
}

static struct jprobe my_jprobe = {
        .entry = (kprobe_opcode_t *) submit_bio_probe,
        .kp = {
                // can not set both addr and symbo_name
                // either set addr or symbol_name
                // if not -21 while retured
                .addr = NULL, //(kprobe_opcode_t *) 0xc04e6e4,
                .symbol_name = "submit_bio",
        },
};
static int __init my_init(void) {
        int ret = 0;
        printk(KERN_INFO "submit_bio jprobe module install...\n");
        ret = register_jprobe(&my_jprobe);
        if(ret < 0) {
                printk(KERN_INFO "register_jprobe failed, returned %d\n", ret);
                return ret;
        }
        printk(KERN_INFO "Planted jprobe at %p, handler addr %p\n",
                my_jprobe.kp.addr, my_jprobe.entry);
        return ret;
}
static void __exit my_exit(void) {
        printk(KERN_INFO "submit_bio jprobe module uninstall...\n");
        unregister_jprobe(&my_jprobe);
        printk(KERN_INFO "jprobe at %p unregistered\n", my_jprobe.kp.addr);
}
module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
