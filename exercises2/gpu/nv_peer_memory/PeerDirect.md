# Client Registration
In order for a device to use PeerDirect with MLNX OFED, it needs to have its driver register a peer_memory_client struct against the ib_core module (defined in include/rdma/ib_peer_mem.h). The struct defines various callbacks needed by the core to implement PeerDirect functionality. The next few sections explain these callbacks and how they are used.

The below are examples of peer memory clients:
1) The io_peer_mem module exposes any MMIO memory that was mapped to the process.   
2)The nv_peer_memory module provides PeerDirect support for NVIDIA GPUs, using the NVIDIA kernel APIs described in Developing a Linux Kernel Module using GPUDirect RDMA.   
3)The AMD ROCnRDMA module provides PeerDirect support for AMD GPUs as part of the ROCm project.   

```C
void * gpu_buffer;
struct ibv_mr *mr;
const int size = 64*1024;
cudaMalloc(&gpu_buffer,size); // TODO: Check errors
mr = ibv_reg_mr(pd,gpu_buffer,size,IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
```

# Mellanox kernel module
Graphics card gets registered as peer memory client to infiniband driver.   
// From nv_peer_mem.c   
```C
static int __init nv_mem_client_init(void)
{
  strcpy(nv_mem_client.name, DRV_NAME);
  strcpy(nv_mem_client.version, DRV_VERSION);
  reg_handle = ib_register_peer_memory_client(&nv_mem_client,
               &mem_invalidate_callback);
  if (!reg_handle)
    return -EINVAL;

  return 0;
}
```

## Peer Memory Registration/Deregistration
Memory Regions (MRs) are typically registered by a user application with a given range of virtual addresses. The overall process for registering and deregistering a peer device MR is shown in the diagram below.   


Without PeerDirect, the adapter kernel driver would pin the pages belonging to the virtual range passed from user-space using ***get_user_pages()*** call, and passes their physical addresses to the adapter. With PeerDirect, when registering an MR, each peer client gets a chance to check whether it needs to handle the mapping (step a in the diagram). The core calls each ***client's acquire()*** function, and if the client returns a positive number, that client is responsible for translating the virtual range to physical pages.

The acquire() function is defined as follows:   
```C
int (*acquire) (unsigned long addr, size_t size, void *peer_mem_private_data, char *peer_mem_name, void **client_context);   
```
It receives the address range (addr, size) that is being registered, and a pointer to a context hint peer_mem_private_data attached to the process's ib_ucontext struct. In addition to the client name, the function accepts a pointer to a context that is returned by the client on successful calls (client_context). This context is saved by the caller and is passed on to future calls to help track this region.   

Once the right peer client is found, the core will call the get_pages() callback (step b in the diagram) to translate the virtual memory range into a collection of pages.   
```C
int (*get_pages)(unsigned long addr, size_t size, int write, int force, struct sg_table *sg_head, void *client_context, u64 core_context);   
```
The function accepts the virtual address range. The write and force parameters behave similarly to get_user_pages(). write means the memory will be written as well, while force will force write access even if the current mapping is read-only. The core always sets the write flag to force copy-on-write (CoW) to occur during registration, and only sets force on MRs that are enabled for write operations.   
 
The sg_head parameter points to an sg_table to return the resulting addresses. The client allocates the table and fills it with the physical addresses matching the requested range. The scatterlist elements should be made of pages. These pages can be larger than the minimal operating system page size. In order to take advantage of that, the core can call the get_page_size() callback with the client_context to ask the client what the page size for that range is.   

The client_context value returned from acquire() is passed back to get_pages(). get_pages() also receives a core_context 64-bit value to be used by the peer client if it needs to invalidate the mapping (see below).   

Finally, the core calls the dma_map() callback (step c in the diagram) to get the bus addresses of the peer memory. These addresses must be accessible by the adapter.   
```C
int (*dma_map) (struct sg_table *sg_head, void *client_contex, struct device *dma_device, int dmasync, int *nmap);
```
The client receives the table it allocated during get_pages() and the client_context. It also receives the struct device belonging to the adapter, in case it needs to map the addresses specifically for that device. The dmasync has a similar meaning to the DMA_ATTR_WRITE_BARRIER attribute (see the DMA-attributes.txt kernel document). It means that a DMA write to the region by the adapter should force all previous DMA writes to the peer device to complete first. The function returns the number of mappings in the nmap parameter.   

De-registration is performed by calling the dma_unmap, put_pages, and release after stopping the adapter from using these pages.   

# Invalidations

In some cases, the peer client device may want to prevent the adapter from using pages it has already pinned using get_pages(). For example, a GPU performing a context switch may need to reallocate its MMIO pages. In these cases, the peer client module can use the core_context value passed to it in the get_pages() call to invalidate the adapter usage. The invalidation callback is returned from the client registration function (ib_register_peer_memory_client). Its signature is:

int (*invalidate_peer_memory)(void *reg_handle, void *core_context);

The invalidate_peer_memory function accepts the registration handle, reg_handle, returned from ib_register_peer_memory_client(), and the core_context value from put_pages(). The function marks the memory region so that the adapter stops using it, and waits until it is guaranteed to do so.