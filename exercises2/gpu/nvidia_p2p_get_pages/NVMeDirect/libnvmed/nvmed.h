#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

int nvmed_init(int threads);
int nvmed_deinit();
size_t nvmed_send(int fd, void **gpuMemPtr, size_t size, unsigned long offset);
size_t nvmed_recv(int fd, void *gpuMemPtr, size_t size, unsigned long offset);
#ifdef __cplusplus
}
#endif
