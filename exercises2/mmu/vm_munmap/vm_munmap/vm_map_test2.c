/*
 * Kernel module for testing copy_to/from_user infrastructure.
 *
 * Copyright 2013 Google Inc. All Rights Reserved
 *
 * Authors:
 *      Kees Cook       <keescook@chromium.org>
 *
 * This software is licensed under the terms of the GNU General Public
 * License version 2, as published by the Free Software Foundation, and
 * may be copied, distributed, and modified under those terms.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#include <linux/mman.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>
#define NR_PAGES_ORDER  1
/*
 * Several 32-bit architectures support 64-bit {get,put}_user() calls.
 * As there doesn't appear to be anything that can safely determine
 * their capability at compile-time, we just have to opt-out certain archs.
 */
#if BITS_PER_LONG == 64 || (!(defined(CONFIG_ARM) && !defined(MMU)) && \
			    !defined(CONFIG_BLACKFIN) &&	\
			    !defined(CONFIG_M32R) &&		\
			    !defined(CONFIG_M68K) &&		\
			    !defined(CONFIG_MICROBLAZE) &&	\
			    !defined(CONFIG_MN10300) &&		\
			    !defined(CONFIG_NIOS2) &&		\
			    !defined(CONFIG_PPC32) &&		\
			    !defined(CONFIG_SUPERH))
# define TEST_U64
#endif
#define test(condition, msg)		\
({					\
	int cond = (condition);		\
	if (cond)			\
		pr_warn("%s\n", msg);	\
	cond;				\
})
const int total = (1 << NR_PAGES_ORDER)*PAGE_SIZE;
char * first_addr;
static int my_fault(struct vm_fault *vmf)
{
         struct page *page;
        unsigned long offset;
        //struct vm_area_struct *vma = vmf->vma;
        offset = ((unsigned long)vmf->address) - ((unsigned long)vmf->vma->vm_start);
        offset = offset >>  PAGE_SHIFT;
        if (offset > (1 << NR_PAGES_ORDER)) {
            printk(KERN_ERR "Invalid address deference, offset = %lu \n",
           offset);
          return 0;
        }
        printk(KERN_ERR "page index offset = %lu \n",offset);
        page = virt_to_page(first_addr) + offset;
        get_page(page);
        vmf->page = virt_to_page(page);
        return 0;
}
struct vm_operations_struct my_vm_ops = {
  .fault = my_fault
};
static int __init test_user_copy_init(void)
{
	int ret = 0;
	char *kmem;
	char __user *usermem;
	char *bad_usermem;
	unsigned long user_addr;
        struct vm_area_struct *vm_area;
	kmem = kmalloc(total, GFP_KERNEL);
	if (!kmem)
		return -ENOMEM;
	user_addr = vm_mmap(NULL, 0, total,
			    PROT_READ | PROT_WRITE | PROT_EXEC,
			    MAP_ANONYMOUS | MAP_PRIVATE, 0);
	if (user_addr >= (unsigned long)(TASK_SIZE)) {
		pr_warn("Failed to allocate user memory\n");
                goto err1;
	}
         
        vm_area = find_vma(current->mm, user_addr);
        if(NULL == vm_area)
        {
            goto err2;
        }
        first_addr = (char *)__get_free_pages(GFP_KERNEL, NR_PAGES_ORDER);
        if(NULL == first_addr)
        {
            goto err2;
        }
#if 0
        //casue coredump
	vm_area->vm_ops = &my_vm_ops;
#endif
	usermem = (char __user *)user_addr;
	bad_usermem = (char *)user_addr;
	/*
	 * Legitimate usage: none of these copies should fail.
	 */
	memset(kmem, 0x3a, total);
	ret |= test(copy_to_user(usermem, kmem, PAGE_SIZE),
		    "legitimate copy_to_user failed");
	memset(kmem, 0x0, PAGE_SIZE);
	ret |= test(copy_from_user(kmem, usermem, PAGE_SIZE),
		    "legitimate copy_from_user failed");
	ret |= test(memcmp(kmem, kmem + PAGE_SIZE, PAGE_SIZE),
		    "legitimate usercopy failed to copy data");
	vm_munmap(user_addr, total);
        free_pages((unsigned long)first_addr,NR_PAGES_ORDER);    
	kfree(kmem);
	if (ret == 0) {
		pr_info("tests passed.\n");
		return 0;
	}
	return -EINVAL;
err2:
	vm_munmap(user_addr, total);
err1:
        kfree(kmem);
	return -ENOMEM;
}
module_init(test_user_copy_init);
static void __exit test_user_copy_exit(void)
{
	pr_info("unloaded.\n");
}
module_exit(test_user_copy_exit);
MODULE_AUTHOR("Kees Cook <keescook@chromium.org>");
MODULE_LICENSE("GPL");
