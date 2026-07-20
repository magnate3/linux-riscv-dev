
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/device.h> 
#include <linux/cdev.h> 
#include <linux/dmaengine.h>
#include <linux/wait.h>
#include <linux/string.h>
#include <linux/dma-mapping.h>
#include <linux/slab.h>
#include <linux/jiffies.h>
 
 
#define DEBUG_PRINT printk
 
#define MEMCPY_NO_DMA 0
#define MEMCPY_DMA    1
#define BUFF_SIZE     (512*1024)
 
struct cdev my_cdev;
static int major_ret;
static struct class *pdma_class;
static struct device *pdma_device;
 
static dma_addr_t *src = NULL;
static dma_addr_t src_phys ;
static dma_addr_t *dst = NULL;
static dma_addr_t dst_phys ;
 
static volatile int dma_finished = 0;
static DECLARE_WAIT_QUEUE_HEAD(wq);
 
 
static void do_memcpy_no_dma(void)
{
    unsigned long t1 , t2,diff,msec;
    int i ;
    t1  = jiffies;
    for(i = 0;i < 1000;i++)
    {
        memcpy(dst,src,BUFF_SIZE);    
    }
    t2 = jiffies;
 
    diff = (long)t2 - (long)t1;
    msec = diff *1000/HZ;
 
    DEBUG_PRINT("used:%ld ms\n",msec);
    
}
 
static void tx_callback(void *dma_async_param)
{
    //DEBUG_PRINT("callback here\n");
    dma_finished = 1;
    wake_up_interruptible(&wq);
}
 
static int do_memcpy_with_dma(void)
{
    struct dma_chan *chan = NULL;
    dma_cap_mask_t mask;
    
    struct dma_async_tx_descriptor *tx = NULL;
 
    dma_cookie_t dma_cookie;
    
    memset(src,0xAA,BUFF_SIZE);
    memset(dst,0x55,BUFF_SIZE);    
    
    dma_cap_zero(mask);
    
    dma_cap_set(DMA_MEMCPY, mask);
    
    chan = dma_request_channel(mask, NULL, NULL);
    if(NULL == chan )
    {
        printk("err:%s:%d\n",__FILE__,__LINE__);        
        return -1;
    }
    
 
    
    
    unsigned long t1 , t2,diff,msec;
    int i ;
    t1  = jiffies;
    for(i=0;i<1000;i++)
    {
        dma_finished = 0;
        //tx = dmaengine_prep_dma_cyclic(chan, src_phys, BUFF_SIZE, 1024, DMA_MEM_TO_DEV, DMA_PREP_INTERRUPT|DMA_CTRL_ACK);
        tx = dmaengine_prep_dma_memcpy(chan, dst_phys, src_phys, BUFF_SIZE, DMA_PREP_INTERRUPT|DMA_CTRL_ACK);
 
        if(NULL == tx)
        {
            printk("err:%s:%d\n",__FILE__,__LINE__);    
            dma_release_channel(chan);
            return -1;
        }
 
        tx->callback = tx_callback;
        
        dma_cookie = dmaengine_submit(tx);
        if (dma_submit_error(dma_cookie))
        {
            printk("Failed to do DMA tx_submit");
        }
        
        dma_async_issue_pending(chan);    
 
        wait_event_interruptible(wq, dma_finished);
        
    }
 
    t2  = jiffies;
    diff = (long)t2 - (long)t1;
    msec = diff *1000/HZ;
 
    DEBUG_PRINT("used:%ld ms\n",msec);
 
    printk("ok !\n");
    if(memcmp(src, dst, BUFF_SIZE) == 0)
    {
        printk("memcpy succ !\n");
    }
    else
    {
        printk("memcpy failed !\n");
        for(i=0;i<8;i++)
        {
            printk("%llx | %llx\n",src[i],dst[i]);
        }
    }    
 
    
    dma_release_channel(chan);
    
    return 0;
}
 
 
static long dma_ioctl(struct file *file, unsigned int cmd, unsigned long data)
{
    switch (cmd)
    {
        case MEMCPY_NO_DMA:
            do_memcpy_no_dma();
            break;
        case MEMCPY_DMA:
            do_memcpy_with_dma();
            break;
    }
    return 0;
}
 
 
static const struct file_operations fops =
{
    .owner = THIS_MODULE,
    .unlocked_ioctl = dma_ioctl,
};
 
static int __init dma_init(void)
{
    dev_t devno = 0;
    
    alloc_chrdev_region(&devno, 0, 1, "my-dma");
    major_ret = MAJOR(devno);
    cdev_init(&my_cdev, &fops);
    cdev_add(&my_cdev, devno, 1);
 
    pdma_class = class_create(THIS_MODULE, "my-dma-class");
 
    pdma_device = device_create(pdma_class, NULL, MKDEV(major_ret,0), NULL, "my-dma");
 
    src = dma_alloc_coherent(NULL, BUFF_SIZE, &src_phys, GFP_KERNEL);
 
    if(NULL == src)
    {
        printk("err:%s:%d\n",__FILE__,__LINE__);
        goto _FAILED_ALLOC_SRC;
    }
    
    dst = dma_alloc_coherent(NULL, BUFF_SIZE, &dst_phys, GFP_KERNEL);    
 
    if(NULL == dst)
    {        
        printk("err:%s:%d\n",__FILE__,__LINE__);
        goto _FAILED_ALLOC_DST;
    }
    
    return 0;
_FAILED_ALLOC_DST:    
    
    dma_free_coherent(NULL, BUFF_SIZE, src, src_phys);
_FAILED_ALLOC_SRC:
    device_destroy(pdma_class, MKDEV(major_ret,0)); 
    class_destroy(pdma_class);    
    cdev_del(&my_cdev);
    unregister_chrdev_region(MKDEV(major_ret, 0), 1);
 
    return -1;
    
}
 
static void __exit dma_exit(void)
{    
    //printk("hello dma openwrt exit\n");
    device_destroy(pdma_class, MKDEV(major_ret,0));    
    class_destroy(pdma_class);
 
    dev_t devno = MKDEV(major_ret, 0);
    cdev_del(&my_cdev);
    unregister_chrdev_region(devno, 1);    
 
    dma_free_coherent(NULL, BUFF_SIZE, src, src_phys);
    dma_free_coherent(NULL, BUFF_SIZE, dst, dst_phys);    
    
}
 
module_init(dma_init);
module_exit(dma_exit);
 
MODULE_AUTHOR("hello world");
MODULE_DESCRIPTION("dma driver");
MODULE_LICENSE("GPL");
//MODULE_ALIAS("platform:" DRV_NAME);
 
