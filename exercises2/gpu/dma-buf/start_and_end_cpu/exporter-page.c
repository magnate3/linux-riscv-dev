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
#include <linux/highmem.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/miscdevice.h>
#include "dma_buf_exporter_kmd.h"

static struct dma_buf *dmabuf_exported;

static int exporter_attach(struct dma_buf *dmabuf, struct device *dev,
			struct dma_buf_attachment *attachment)
{
	pr_info("dmabuf attach device: %s\n", dev_name(dev));
	return 0;
}

static void exporter_detach(struct dma_buf *dmabuf, struct dma_buf_attachment *attachment)
{
	pr_info("dmabuf detach device: %s\n", dev_name(attachment->dev));
}

static struct sg_table *exporter_map_dma_buf(struct dma_buf_attachment *attachment,
					 enum dma_data_direction dir)
{
	struct page *page = attachment->dmabuf->priv;
	struct sg_table *table;
	int err;

        dump_stack();
	table = kmalloc(sizeof(*table), GFP_KERNEL);
	if (!table)
		return ERR_PTR(-ENOMEM);

	err = sg_alloc_table(table, 1, GFP_KERNEL);
	if (err) {
		kfree(table);
		return ERR_PTR(err);
	}

	sg_set_page(table->sgl, page, PAGE_SIZE, 0);
	sg_dma_address(table->sgl) = dma_map_page(NULL, page, 0, PAGE_SIZE, dir);

	return table;
}

static void exporter_unmap_dma_buf(struct dma_buf_attachment *attachment,
			       struct sg_table *table,
			       enum dma_data_direction dir)
{
	dma_unmap_page(NULL, sg_dma_address(table->sgl), PAGE_SIZE, dir);
	sg_free_table(table);
	kfree(table);
}

static void exporter_release(struct dma_buf *dma_buf)
{
	//struct page *page = dma_buf->priv;

	pr_info("dmabuf release\n");
	//put_page(page);
}

static void *exporter_vmap(struct dma_buf *dma_buf)
{
	struct page *page = dma_buf->priv;

	return vmap(&page, 1, 0, PAGE_KERNEL);
}

static void exporter_vunmap(struct dma_buf *dma_buf, void *vaddr)
{
	vunmap(vaddr);
}

static void *exporter_kmap_atomic(struct dma_buf *dma_buf, unsigned long page_num)
{
	struct page *page = dma_buf->priv;

	return kmap_atomic(page);
}

static void exporter_kunmap_atomic(struct dma_buf *dma_buf, unsigned long page_num, void *addr)
{
	kunmap_atomic(addr);
}

static void *exporter_kmap(struct dma_buf *dma_buf, unsigned long page_num)
{
	struct page *page = dma_buf->priv;

	return kmap(page);
}

static void exporter_kunmap(struct dma_buf *dma_buf, unsigned long page_num, void *addr)
{
	struct page *page = dma_buf->priv;

	return kunmap(page);
}

static int exporter_mmap(struct dma_buf *dma_buf, struct vm_area_struct *vma)
{
	struct page *page = dma_buf->priv;

	return remap_pfn_range(vma, vma->vm_start, page_to_pfn(page),
				PAGE_SIZE, vma->vm_page_prot);
}

static int exporter_begin_cpu_access(struct dma_buf *dmabuf,
				      enum dma_data_direction dir)
{
	struct dma_buf_attachment *attachment;
	struct sg_table *table;

        //dump_stack();
	if (list_empty(&dmabuf->attachments))
		return 0;

	attachment = list_first_entry(&dmabuf->attachments, struct dma_buf_attachment, node);
	table = attachment->priv;
	dma_sync_sg_for_cpu(NULL, table->sgl, 1, dir);

	return 0;
}

static int exporter_end_cpu_access(struct dma_buf *dmabuf,
				enum dma_data_direction dir)
{
	struct dma_buf_attachment *attachment;
	struct sg_table *table;

        //dump_stack();
	if (list_empty(&dmabuf->attachments))
		return 0;

	attachment = list_first_entry(&dmabuf->attachments, struct dma_buf_attachment, node);
	table = attachment->priv;
	dma_sync_sg_for_device(NULL, table->sgl, 1, dir);

	return 0;
}

static const struct dma_buf_ops exp_dmabuf_ops = {
	.attach = exporter_attach,
	.detach = exporter_detach,
	.map_dma_buf = exporter_map_dma_buf,
	.unmap_dma_buf = exporter_unmap_dma_buf,
	.release = exporter_release,
	.map = exporter_kmap,
	.unmap = exporter_kunmap,
	.map_atomic = exporter_kmap_atomic,
	.unmap_atomic = exporter_kunmap_atomic,
	.mmap = exporter_mmap,
	.vmap = exporter_vmap,
	.vunmap = exporter_vunmap,
	.begin_cpu_access = exporter_begin_cpu_access,
	.end_cpu_access = exporter_end_cpu_access,
};

static struct dma_buf *exporter_alloc_page(void)
{
	DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
	struct dma_buf *dmabuf;
	struct page *page;

	page = alloc_page(GFP_KERNEL);
	if (!page)
		return NULL;

	exp_info.ops = &exp_dmabuf_ops;
	exp_info.size = PAGE_SIZE;
	exp_info.flags = O_CLOEXEC;
	exp_info.priv = page;

	dmabuf = dma_buf_export(&exp_info);
	if (IS_ERR(dmabuf)) {
		put_page(page);
		return NULL;
	}

	sprintf(page_address(page), "hello world!");

	return dmabuf;
}

 
static void dma_buf_exporter_free(struct dma_buf *dma_buf)
{
	struct page *page = dma_buf->priv;
	pr_info("dmabuf page free \n");
	put_page(page);
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
	.owner   = THIS_MODULE,
	.unlocked_ioctl   = exporter_ioctl,
};
 
static struct miscdevice mdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "exporter",
	.fops = &exporter_fops,
};
 
static int __init exporter_init(void)
{
	return misc_register(&mdev);
}

static void __exit exporter_exit(void)
{
	misc_deregister(&mdev);
}

module_init(exporter_init);
module_exit(exporter_exit);

MODULE_AUTHOR("Leon He <343005384@qq.com>");
MODULE_DESCRIPTION("DMA-BUF exporter example for page alloc");
MODULE_LICENSE("GPL v2");

