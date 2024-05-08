

# dma

SPDK (Software Development Performance Kit) supports user-space device management and is usually used for low-latency high-performance NVMe devices, but can be also used for hig performance I/OAT DMA devices by eliminating kernel - user space context switches for DMA operations.


+ 1. Initializing SPDK & Allocating I/OAT Channel#
To use SPDK, first we should initialize the SPDK environment using:

```
/* Initialize the default value of opts. */
void spdk_env_opts_init(struct spdk_env_opts *opts);
/* Initialize or reinitialize the environment library.
 * For initialization, this must be called prior to using any other functions
 * in this library.
 */
int spdk_env_init(const struct spdk_env_opts *opts);
struct spdk_env_opts opts;
spdk_env_opts_init(&opts);
/* modify opts.* as you want */
spdk_env_init(&opts);
```

   And then, use spdk_ioat_probe() to probe I/OAT DMA devices. Before calling it, target I/OAT devices should be setup with UIO/VFIO with scripts/setup.sh src:

```
$ sudo scripts/setup.sh
/* Callback for spdk_ioat_probe() enumeration. */
typedef bool (*spdk_ioat_probe_cb)(void *cb_ctx, struct spdk_pci_device *pci_dev);
/**
 * Callback for spdk_ioat_probe() to report a device that has been attached to
 * the userspace I/OAT driver.
 */
typedef void (*spdk_ioat_attach_cb)(void *cb_ctx, struct spdk_pci_device *pci_dev,
                                    struct spdk_ioat_chan *ioat);

/** 
 * Enumerate the I/OAT devices attached to the system and attach the userspace
 * I/OAT driver to them if desired.
 */
int spdk_ioat_probe(void *cb_ctx, spdk_ioat_probe_cb probe_cb, spdk_ioat_attach_cb attach_cb);
spdk_ioat_probe() probes all I/OAT devices and calls probe_cb() callback function for each probed device, and also tries to attach the device into the userspace process with attach_cb(), if the corresponding probe_cb() returns true for the device.

struct spdk_ioat_chan *ioat_chan = NULL;

static bool probe_cb(void *cb_ctx, struct spdk_pci_deivce *pci_device) {
    if (ioat_chan) {
        return false;
    } else {
        return true;
    }
}

static void attach_cb(volid *cb_ctx, struct spdk_pci_device *pci_device, struct spdk_ioat_chan *ioat) {
    // Check if that device/channel supports copy operations
    if (!(spdk_ioat_get_dma_capabilities(ioat) & SPDK_IOAT_ENGINE_COPY_SUPPORTED)) {
        return;
    }

    ioat_chan = ioat;
    printf("Attaching to the ioat device!\n");
}
```


if (spdk_ioat_probe(NULL, probe_cb, attach_cb)) { /* handle error */ }
probe_cb() returns true if there is no attached spdk_ioat_chan channel. Then spdk_ioat_probe() calls attach_cb() for the device that probe_cb returns true, then attach_cb() checks the memory copy capability and attach it.


+  2. DMA Transfer and Configuring Completion Callback#
SPDK requires to call spdk_ioat_build_copy() to build a DMA request, and then flush it into the device via spdk_ioat_flush(). You can use spdk_ioat_submit_copy() to do thw two at once for one request.

```
/* Signature for callback function invoked whan a request is completed. */
typedef void (*spdk_ioat_req_cb)(void *arg);
/** 
 * Build a DMA engine memory copy request (a descriptor).
 * The caller must also explicitly call spdk_ioat_flush to submit the
 * descriptor, possibly after building additional descriptors.
 */
int spdk_ioat_build_copy(struct spdk_ioat_chan *chan,
                         void *cb_arg, spdk_ioat_req_cb cb_fn,
                         void *dst, const void *src, uint64_t nbytes);

/* Flush previously built descriptors. */
void spdk_ioat_flush(struct spdk_ioat_chan *chan);

/* Build and submit a DMA engine memory copy request. */
int spdk_ioat_submit_copy(struct spdk_ioat_chan *chan,
                          void *cb_arg, spdk_ioat_req_cb cb_fn,
                          void *dst, const void *src, uint64_t nbytes);
spdk_ioat_build_copy() and spdk_ioat_submit_copy() receives void *cb_arg and spdk_ioat_req_cb cb_fn, the callback function to be called with a given argument pointer. The callback function will be called when a request is completed, and the userspace process can use it to determine when operation is done. Simple implementation can be:
```


```
bool copy_done = false;
static op_done_cb(void *arg) {
    *(bool*)arg = true;
}

spdk_ioat_submit_copy(ioat_chan,
                      &copy_done,   // optional
                      op_done_cb,   // optional: if given, will be called by ioat_process_channel_events().
                      dst,
                      src,
                      io_size);

int result;
do {
    result = spdk_ioat_process_events(ioat_chan);
} while (result == 0);

assert(copy_done); // must be true!
Note that the target buffer (both src and dst) must be DMAable; you should use spdk_mem_register():

/**
 * Register the specified memory region for address translation.
 * The memory region must map to pinned huge pages (2MB or greater).
 */
int spdk_mem_register(void *vaddr, size_t len);

void *dev_addr = mmap(NULL, device_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
spdk_mem_register(dev_addr, device_size);

/* e.g. void *src = dev_addr + 0x1000; */
```

As the device is handled by a userspace process, not the kernel, the process cannot receive an interrupt but should poll the completion to check whether the submitted operation is completed. spdk_ioat_process_events() is the function for this purpose.


```
/**
 * Check for completed requests on an I/OAT channel.
 * \return number of events handled on success, negative errno on failure.
 */
int spdk_ioat_process_events(struct spdk_ioat_chan *chan);
spdk_ioat_process_events() immediately returns how many requests are completed. You can use the returned value or copy_done variable in the example above to check whether the operation is completed.

Note that, op_done_cb() callback will be called by spdk_ioat_process_events():

int spdk_ioat_process_events(struct spdk_ioat_chan *ioat) {
    return ioat_process_channel_events(ioat);
}

static int ioat_process_channel_events(struct spdk_ioat_chan *ioat) {
    
    uint64_t status = *ioat->comp_update;
    uint64_t completed_descriptor = status & SPDK_IOAT_CHANSTS_COMPLETED_DESCRIPTOR_MASK;
    if (completed_descriptor == ioat->last_seen) {
        return 0;
    }

    do {
        uint32_t tail = ioat_get_ring_index(ioat, ioat->tail);
        struct ioat_descriptor *desc = &ioat->ring[tail];

        // Here, the given callback function is called.
        if (desc->callback_fn) {
            desc->callback_fn(desc->callback_arg);
        }

        hw_desc_phys_addr = desc->phys_addr; // This breaks the loop
        ioat->tail++;
        events_count++;
    } while(hw_desc_phys_addr != completed_descriptor);

    ioat->last_seen = hw_desc_phys_addr;

    return events_count;
}
```
so that you should call spdk_ioat_process_events() to get notified.