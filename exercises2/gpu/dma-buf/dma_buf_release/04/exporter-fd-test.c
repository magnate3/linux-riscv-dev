/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

#include <linux/dma-buf.h>
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include "dma_buf_exporter_kmd.h"

struct dma_buf *dmabuf_exported;
EXPORT_SYMBOL(dmabuf_exported);

static struct sg_table *exporter_map_dma_buf(struct dma_buf_attachment *attachment,
					 enum dma_data_direction dir)
{
	return NULL;
}

static void exporter_unmap_dma_buf(struct dma_buf_attachment *attachment,
			       struct sg_table *table,
			       enum dma_data_direction dir)
{
}

static void exporter_release(struct dma_buf *dmabuf)
{
        pr_info("exporter_release: releasing dma_buf \n");
        //dump_stack();
	//kfree(dmabuf->priv);
	dmabuf_exported = NULL;
}

static void *exporter_kmap_atomic(struct dma_buf *dmabuf, unsigned long page_num)
{
	return NULL;
}

static void *exporter_kmap(struct dma_buf *dmabuf, unsigned long page_num)
{
	return NULL;
}

static int exporter_mmap(struct dma_buf *dmabuf, struct vm_area_struct *vma)
{
	void *vaddr = dmabuf->priv;

	return remap_pfn_range(vma, vma->vm_start, virt_to_pfn(vaddr),
				PAGE_SIZE, vma->vm_page_prot);
}

static const struct dma_buf_ops exp_dmabuf_ops = {
	.map_dma_buf = exporter_map_dma_buf,
	.unmap_dma_buf = exporter_unmap_dma_buf,
	.release = exporter_release,
	.map = exporter_kmap,
	.map_atomic = exporter_kmap_atomic,
	.mmap = exporter_mmap,
};

static struct dma_buf *exporter_alloc_page(void)
{
	DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
	struct dma_buf *dmabuf;
	void *vaddr;

	vaddr = kzalloc(PAGE_SIZE, GFP_KERNEL);
	if (!vaddr)
		return NULL;

	exp_info.ops = &exp_dmabuf_ops;
	exp_info.size = PAGE_SIZE;
	exp_info.flags = O_CLOEXEC;
	exp_info.priv = vaddr;

	dmabuf = dma_buf_export(&exp_info);
	if (IS_ERR(dmabuf)) {
		kfree(vaddr);
		return NULL;
	}

	sprintf(vaddr, "hello world!");

	return dmabuf;
}
static void dma_buf_exporter_free(struct dma_buf *dma_buf)
{
	kfree(dma_buf->priv);
}
static long exporter_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
#if 1
        struct dma_exporter_buf_alloc_data data;
        int fd;
        pr_info("dma_buf_exporter_ioctl %u \n", cmd);
        if (copy_from_user(&data, (void __user *)arg, sizeof(data))) {
                pr_info("dma_buf_exporter: failed to copy user data. ");
                return -EFAULT;
        }
        switch (cmd) {
                case DMA_BUF_EXPORTER_ALLOC: {
	        dmabuf_exported = exporter_alloc_page();
	        fd = dma_buf_fd(dmabuf_exported, O_CLOEXEC);
                data.fd = fd;
                if (copy_to_user((int __user *)arg, &data, sizeof(data)))
	             	return -EFAULT;

	             return 0;
                }
                case DMA_BUF_EXPORTER_FREE: { 
                      //dma_buf = dma_buf_get(data.fd); 
                      dma_buf_exporter_free(dmabuf_exported);
                      return 0;
                }
                default: {
                        pr_info("dma_buf_exporter: invalid IOCTL code %u ", cmd);
                        return -EINVAL;
                }
        }
#else
	int fd = dma_buf_fd(dmabuf_exported, O_CLOEXEC);
        pr_info("exporter dma buf fd %d \n", fd);
        if (copy_to_user((int __user *)arg, &fd, sizeof(fd)))
		return -EFAULT;
	return 0;
#endif
}
 
static struct file_operations exporter_fops = {
	.owner		= THIS_MODULE,
	.unlocked_ioctl	= exporter_ioctl,
};
 
static struct miscdevice mdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "exporter",
	.fops = &exporter_fops,
};
 
static int __init exporter_init(void)
{
#if 0
        dmabuf_exported = exporter_alloc_page();
	if (!dmabuf_exported) {
		pr_err("error: exporter alloc page failed\n");
		return -ENOMEM;
	}
#endif
	return misc_register(&mdev);
}

static void __exit exporter_exit(void)
{
	misc_deregister(&mdev);
}

module_init(exporter_init);
module_exit(exporter_exit);

MODULE_AUTHOR("linux");
MODULE_DESCRIPTION("DMA-BUF Exporter example for cpu-access (mmap)");
MODULE_LICENSE("GPL v2");

