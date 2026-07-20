# DMA Buffer Exporter Kernel Mode Driver

## Features

 * Creates a new character device /dev/dma_buf_exporter
 * Provides IOCTL interfaces DMA_BUF_EXPORTER_ALLOC and DMA_BUF_EXPORTER_FREE to allocate and free dma_buf
 * Exports interfaces to attach, detach, map and unmap dma_buf in kernel mode

## Prerequisites

* make
* gcc
* kernel headers

## Build, install load

```shell
sudo make all        # builds, installs, cleans and loads module.
make build           # builds module
sudo make install    # installs module into the system and enable autorun.
sudo make uninstall  # uninstalls module from the system
sudo make load       # loads module into kernel
sudo make unload     # unloads module from kernel
make clean           # cleans build files
```
To check module version number:
```shell
modinfo dma_buf_exporter_intel
```

After installation you can use **modprobe** and other Linux's tool for modules.


## Build flags

* -Wall:   Enables all compilers warning messages

#  sg_set_page and  dma_map_sg

```
static struct sg_table *dma_buf_exporter_map_dma_buf(struct dma_buf_attachment *attachment,
					 enum dma_data_direction dir)
{
	struct dma_buf_exporter_data *data = attachment->dmabuf->priv;
	struct sg_table *table;
	struct scatterlist *sg;
	int i;

	pr_info("dma_buf_exporter: mapping dma_buf \n");
	table = kmalloc(sizeof(*table), GFP_KERNEL);
	if (!table)
		return ERR_PTR(-ENOMEM);

	if (sg_alloc_table(table, data->num_pages, GFP_KERNEL)) {
		kfree(table);
		return ERR_PTR(-ENOMEM);
	}

	sg = table->sgl;
	for (i = 0; i < data->num_pages; i++) {
		sg_set_page(sg, data->pages[i], PAGE_SIZE, 0);
		sg = sg_next(sg);
	}

	if (!dma_map_sg(NULL, table->sgl, table->nents, dir)) {
		sg_free_table(table);
		kfree(table);
		return ERR_PTR(-ENOMEM);
	}

	return table;
}
```

# test


```
ls /dev/dma_buf_exporter 
/dev/dma_buf_exporter
```

```
root@ubuntux86:# insmod dma_buf_exporter_kmd_intel.ko 
root@ubuntux86:# ./user_test 
ion alloc success: size = 12288, dmabuf_fd = 4
root@ubuntux86:# 
```

# references

[Linux内核笔记之DMA_BUF](https://saiyn.github.io/homepage/2018/04/18/linux-kernel-dmabuf/)   