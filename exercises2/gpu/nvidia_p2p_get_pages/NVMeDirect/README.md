
 ![images](../pic/peer3.png)
#  NV_PIN_GPU_MEMORY |  NV_UNPIN_GPU_MEMORY
```
static int pin(struct pin_buf_priv *pb)
{
    struct nv_gpu_mem gpumem = {
        .address = (__u64) pb->pub.address,
        .size = pb->pub.bufsize,
        .p2pToken = pb->tokens.p2pToken,
        .vaSpaceToken = pb->tokens.vaSpaceToken,
    };

    int ret = ioctl(devfd, NV_PIN_GPU_MEMORY, &gpumem);

    pb->pub.handle = gpumem.handle;

    return ret;
}

static int unpin(struct pin_buf_priv *pb)
{
    struct nv_gpu_mem gpumem = {
        .address = (__u64) pb->pub.address,
        .size = pb->pub.bufsize,
        .p2pToken = pb->tokens.p2pToken,
        .vaSpaceToken = pb->tokens.vaSpaceToken,
    };

    int ret = ioctl(devfd, NV_UNPIN_GPU_MEMORY, &gpumem);

    return ret;
}
```

init_pin_buf  -->   pin(pb)   

```

static int init_pin_buf(struct pin_buf_priv *pb, size_t bufsize)
{
    pb->pub.mmap = NULL;
    pb->pub.bufsize = bufsize;
    int err;
    if ((err = cudaMalloc(&pb->pub.address, bufsize)) != cudaSuccess) {
        errno = ENOMEM;
//        fprintf(stderr, "cudaMalloc failed %s\n",  cudaGetErrorString(err));
        return -ENOMEM;
    }

    if (cuPointerGetAttribute(&pb->tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS,
                              (CUdeviceptr) pb->pub.address) != CUDA_SUCCESS)
    {
        errno = EIO;
        goto free_buf;
    }

    if (pin(pb))
        goto free_buf;

    return 0;

free_buf:
    cudaFree(pb->pub.address);
//    fprintf(stderr, "free cuda pinned memory\n");
    return -errno;
}
```