(a.1) The size of guest physical memory is 0x200000 (2MB).  
It is allocated on the host at line 107:
```c
vm->mem = mmap(NULL, mem_size, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
```
since it is dynamic, the address is not static, but I added a print and got the following address 0x7fa78d66e000
And it is set for the kvm on lines 116-124:
```c
memreg.slot = 0;
memreg.flags = 0;
memreg.guest_phys_addr = 0;
memreg.memory_size = mem_size;
memreg.userspace_addr = (unsigned long)vm->mem;
      if (ioctl(vm->fd, KVM_SET_USER_MEMORY_REGION, &memreg) < 0) {
  perror("KVM_SET_USER_MEMORY_REGION");
              exit(1);
}
```
(a.2) 
Additional memory is mapped for the VCPU.  
This memory stores the state of the cpu as well as its registers.  
This memory is allocated at line 149:
```c
vcpu->kvm_run = mmap(NULL, vcpu_mmap_size, PROT_READ | PROT_WRITE,
			     MAP_SHARED, vcpu->fd, 0);
```
<!-- The memory is set with page table mapping information and the code itself.  
In `setup_long_mode` the page tables information is set:
```c
uint64_t pml4_addr = 0x2000;
uint64_t *pml4 = (void *)(vm->mem + pml4_addr);

uint64_t pdpt_addr = 0x3000;
uint64_t *pdpt = (void *)(vm->mem + pdpt_addr);

uint64_t pd_addr = 0x4000;
uint64_t *pd = (void *)(vm->mem + pd_addr);

pml4[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | pdpt_addr;
pdpt[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | pd_addr;
pd[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | PDE64_PS;
```
The addresses are relative to the guest physical memory space (which was set to start from 0 and is 2MB long).  
In line 346 the code section is set with the code of the VM (which source is in `guest.c`):
```c
memcpy(vm->mem, guest64, guest64_end-guest64);
``` -->

(a.3) The setup for the guest code is done at line 436:
```c
memcpy(vm->mem, guest64, guest64_end-guest64);
```
it is set to start in the guest physical address 0x0 (since the memcpy dest is `vm->mem` which is the start).  
The guest page table data is set on lines 383:394 :
```c
uint64_t pml4_addr = 0x2000;
uint64_t *pml4 = (void *)(vm->mem + pml4_addr);

uint64_t pdpt_addr = 0x3000;
uint64_t *pdpt = (void *)(vm->mem + pdpt_addr);

uint64_t pd_addr = 0x4000;
uint64_t *pd = (void *)(vm->mem + pd_addr);

pml4[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | pdpt_addr;
pdpt[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | pd_addr;
pd[0] = PDE64_PRESENT | PDE64_RW | PDE64_USER | PDE64_PS;
```
the addresses are self explanatory, they are all offset from the guest physical memory.
The stack is set to be at the end of the guest physical address and that is done on line 428 via the rsp special register:
```c
/* Create stack at top of 2 MB page and grow down. */
regs.rsp = 2 << 20;
```
(a.4) The page table consists of 3 levels.  
It occupies TODO

(a.5) The code is set to memory address 0.  
At line 427, the instruction pointer is set to 0:
```c
regs.rip = 0;
```
(a.6)
The port guest uses port number 0xE9.  
The hypervisor gets an exit status `KVM_EXIT_IO_OUT` and can then inspect the VCPU state (line 174):
```c
vcpu->kvm_run->io.port == 0xE9
```
The memory buffer used is a memory buffer allocated after the `kvm_run` structure, and is pointed by `io.data_offset`.  
There occurs "hello world!".length number of exits 
