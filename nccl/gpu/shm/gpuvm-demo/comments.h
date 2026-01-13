    // cuCtxGetCacheConfig();
    // cuCtxSetLimit();    
    // cuMemCreate
    // cuMemRelease

    // Calculates either the minimal or recommended granularity.
    // CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop,
    //                                        CUmemAllocationGranularity_flags option);

    // // Allocate an address range reservation.
    // CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
    //                              unsigned long long flags);
    // // Free an address range reservation.
    // CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);

    // // Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.
    // CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop,
    //                      unsigned long long flags);
    // // Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.
    // CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
    // // Retrieve the contents of the property structure defining properties for this handle.
    // CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle);

    // // Maps an allocation handle to a reserved virtual address range.
    // CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
    //                   unsigned long long flags);
    // // Unmap the backing memory of a given address range.
    // CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);

    // // Get the access flags set for the given location and ptr.
    // CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr);
    // // Set the access flags for each location specified in desc for the given virtual address range.
    // CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);