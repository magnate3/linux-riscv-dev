#include <stdio.h>
#include <nvshmem.h>

int main(void) {
    nvshmem_init();
    int mype = nvshmem_my_pe();
    printf("Olá do PE %d\n", mype);
    nvshmem_finalize();
    return 0;
}
