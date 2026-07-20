#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "myMemPool.h"

int main(void){

    mem_pool_manager manager;

    mem_pool_init(&manager);

    void *p = allocate(&manager, 20);
    deallocate(&manager, p, 20);

    while(1)
        sleep(10);
}
