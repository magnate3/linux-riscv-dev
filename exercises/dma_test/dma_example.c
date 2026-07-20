#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/fcntl.h>
#include <linux/cdev.h>
#include <linux/version.h>
#include <linux/vmalloc.h>
#include <linux/ctype.h>
#include <linux/pagemap.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/device.h>

#include<linux/dmaengine.h>  
#include<linux/dma-mapping.h> 


#include <linux/jiffies.h>

#define XFER_TIMES 4000 
#define XFER_LEN  1<<20

void *dma_src,*dma_dest;
struct dma_chan *chan = NULL;

static int __init dmatest_init(void)
{

    int xfer_order = get_order(XFER_LEN);
    int i,ret ;
    dma_cap_mask_t mask;
    dma_cookie_t cookie;
    enum dma_status status;
    u64 j1,j2;

    dma_src = __get_free_pages(GFP_KERNEL | GFP_DMA, xfer_order);
    if (!dma_src) {
        printk(KERN_ALERT "dma_src :alloc memory fail.n");
        ret = -ENOMEM;
        goto CLEAN;

    }

    dma_dest = __get_free_pages(GFP_KERNEL | GFP_DMA, xfer_order);
    if (!dma_dest) {
        printk(KERN_ALERT "dma_dest :alloc memory fail.n");
        ret = -ENOMEM;
        goto CLEAN;
    }
    printk(KERN_NOTICE"dma_src=%#x,dma_dest=%#xn",dma_src,dma_dest);    
    dma_cap_zero(mask);
    dma_cap_set(DMA_MEMCPY, mask);
    chan = dma_request_channel(mask, NULL, NULL);

    if (chan) {
        printk(KERN_NOTICE "dma_request_channel ok,current channel is : %sn",dma_chan_name(chan));
    }else {
        printk(KERN_NOTICE "dma_request_channel fail,no dma channel available.n");
        ret = -1;
        goto CLEAN;
    }

    j1 = get_jiffies_64();
    for(i =0;i<XFER_TIMES;i++) {
        cookie = dma_async_memcpy_buf_to_buf(chan, dma_dest, dma_src, XFER_LEN);
        if (dma_submit_error(cookie)) {
            printk(KERN_NOTICE"submit errorn");
            ret = -1;
            goto CLEAN;
        }
    }

    dma_async_memcpy_issue_pending(chan);
    do {
        status = dma_async_memcpy_complete(chan, cookie, NULL, NULL);

    } while (status == DMA_IN_PROGRESS);

    if (status != DMA_SUCCESS) 
        printk(KERN_NOTICE "dma xfer dont accomplish,status=%dn",status);
    j2 = get_jiffies_64();
    printk(KERN_NOTICE"dma xfer time cost:%d ms.n",jiffies_to_msecs(j2-j1));

    j1 = get_jiffies_64();
    for(i =0;i<XFER_TIMES;i++){
        memcpy(dma_dest, dma_src, XFER_LEN);
    }
    j2 = get_jiffies_64();
    printk(KERN_NOTICE"memcpy time cost:%d ms.n",jiffies_to_msecs(j2-j1));
    return 0;

CLEAN:
    if (chan)
        dma_release_channel(chan);

    if (dma_src)
        free_pages(dma_src,xfer_order);

    if (dma_dest)
        free_pages(dma_dest,xfer_order);

    return ret;

}
/* when compiled-in wait for drivers to load first */
module_init(dmatest_init);

static void __exit dmatest_exit(void)
{
    if (chan)
        dma_release_channel(chan);

    if (dma_src)
        free_pages(dma_src,get_order(XFER_LEN));

    if (dma_dest)
        free_pages(dma_dest,get_order(XFER_LEN));

}
module_exit(dmatest_exit);
MODULE_LICENSE("GPL");