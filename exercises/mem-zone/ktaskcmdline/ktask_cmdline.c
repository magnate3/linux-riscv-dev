#include <linux/module.h>
#include <linux/version.h>
#include <linux/types.h>
#include <linux/sched.h>
#include <linux/string.h>
#include <linux/highmem.h>
#include <linux/binfmts.h>
#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include "ktask_memcache.h"
#include "ktask_cmdline.h"
#include "ktask_cache.h"

#ifdef CONFIG_MMU
//get user-pages by read-only mode
int ktask_get_user_pages(struct task_struct* tsk,
                        struct mm_struct* mm,
                        unsigned long pos,
                        struct page** ppage,
                        struct vm_area_struct** pvma)
{
    int ret = 0;
    #if LINUX_VERSION_CODE >= KERNEL_VERSION(4,9,0)
       unsigned int gnu_flags = FOLL_FORCE;
       ret = get_user_pages_remote(tsk, mm, pos,
                   1,gnu_flags,ppage,pvma);
    #elif LINUX_VERSION_CODE >= KERNEL_VERSION(4,6,0)
        ret = get_user_pages_remote(tsk, mm, pos,
                1, 0, 1,ppage, pvma);
    #else
        ret = get_user_pages(tsk,mm, pos,
                1, 0, 1,ppage, pvma);
    #endif

    return ret;
}
#endif


#if LINUX_VERSION_CODE > KERNEL_VERSION(2,6,22) || defined(RHEL_RELEASE_CODE)
    #ifdef CONFIG_MMU
        static struct page *ktask_get_arg_page(struct linux_binprm *bprm, unsigned long pos)
        {
        	struct page *page;
        	int ret;

        	ret = ktask_get_user_pages(current,bprm->mm,pos,&page,NULL);
        	if (ret <= 0)
        		return NULL;

        	return page;
        }

        static void ktask_put_arg_page(struct page *page)
        {
            kunmap(page);
        	put_page(page);
        }

    #else
        static struct page *ktask_get_arg_page(struct linux_binprm *bprm, unsigned long pos)
        {
        	struct page *page;

        	page = bprm->page[pos / PAGE_SIZE];

        	return page;
        }

        static void ktask_put_arg_page(struct page *page)
        {
            kunmap(page);
        }

    #endif /* CONFIG_MMU */
#else
    static struct page *ktask_get_arg_page(struct linux_binprm *bprm, unsigned long pos)
    {
        struct page *page;

        page = bprm->page[pos / PAGE_SIZE];

        return page;
    }

    static void ktask_put_arg_page(struct page *page)
    {
        kunmap(page);
    }
#endif


static int get_one_arg(struct linux_binprm* bprm,int argc,
                unsigned long pos,char* buffer,int buflen)
{
    int len = 0;
    int cross_page = 0;
    char *kaddr = NULL;
    unsigned long kpos = 0;
    struct page *kmapped_page = NULL;

    do {
        int n = 0;
        struct page *page = NULL;
        int offset, bytes_to_copy;

        offset = pos % PAGE_SIZE;
        bytes_to_copy = PAGE_SIZE - offset;

        if (bytes_to_copy > (buflen - len)) {
            bytes_to_copy = buflen - len;
        }

        if (!kmapped_page || kpos != (pos & PAGE_MASK)) {
            page = ktask_get_arg_page(bprm, pos);
            if (!page) { goto out; }

            if (kmapped_page) {
                ktask_put_arg_page(kmapped_page);
            }
            kmapped_page = page;
            kpos = pos & PAGE_MASK;
            kaddr = kmap(kmapped_page);
        }

        /*Note:
         * kernel include the NUL character to process user-space cmdline.
         *so we use strnlen to caculate the length of cmdline
         */
        n = strnlen(kaddr + offset,bytes_to_copy);
        memcpy(buffer + len,kaddr + offset,n);
        len += n;
        //cross page,we must process it
        cross_page = (n == bytes_to_copy);
        if(!cross_page) {
            //one character for NUL character
            n++;
            //add one space character to seperate arguments
            if((argc > 0) && (len + 1 < buflen)) {
                buffer[len++] = ' ';
            }
        } else {
            printk("cross_page\n");
        }
        pos += n;
    } while(cross_page && (buflen > len)); //cross page to process next arguments

out:
    if (kmapped_page) {
        ktask_put_arg_page(kmapped_page);
    }
    return len;
}

static int do_get_cmdline(struct linux_binprm* bprm,char* buffer,int buflen)
{
    int n = 0;
    int len = 0;
    int argc = bprm->argc;
    unsigned long pos = bprm->p;

	while (argc-- > 0) {
		if(len >= buflen) {
            break;
        }

        n = get_one_arg(bprm,argc,pos,
            buffer + len,buflen - len);
        if(n <= 0) { break; }

        pos += n;
        len += n;
	}

	return len;
}

char* ktask_get_cmdline(struct kmem_cache* cachep,
        struct linux_binprm *bprm,unsigned* plen)
{
    int len = 0;
    int rc = -ENOMEM;
    char* cmdline = NULL;

    //we just get one page argument string
    cmdline = ktask_mem_cache_zalloc(cachep,GFP_KERNEL);
    if(!cmdline) { goto out; }

    rc = -EFAULT;
    //one more space for '\0'
    len = kmem_cache_size(cachep) - 1;
    len = do_get_cmdline(bprm,cmdline,len);
    if(len <= 0) { goto out; }

    rc = 0;
    *plen = len;
    cmdline[len] = '\0';

out:
    if(rc) {
        if(cmdline) { ktask_mem_cache_free(cachep,cmdline); }
        cmdline = ERR_PTR(rc);
    }
    return cmdline;
}
