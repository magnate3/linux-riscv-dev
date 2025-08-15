#include<stdio.h>
#include <assert.h>
#include<stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/queue.h> // for TAILQ_ENTRY
#define SPDK_CONTAINEROF(ptr, type, member) ((type *)((uintptr_t)ptr - offsetof(type, member)))
struct spdk_thread {
   int                             msg_fd;
   /* User context allocated at the end */
   uint8_t                         ctx[0];
};
struct nvmf_lw_thread {
    TAILQ_ENTRY(nvmf_lw_thread) link;
    bool resched;
};

void *
spdk_thread_get_ctx(struct spdk_thread *thread)
{
     return thread->ctx;
}
struct spdk_thread *
spdk_thread_get_from_ctx(void *ctx)
{
        if (ctx == NULL) {
                assert(false);
                return NULL;
        }
       return SPDK_CONTAINEROF(ctx, struct spdk_thread, ctx);
}
int main()
{
     struct  spdk_thread sp, *p;
     void * ctx = spdk_thread_get_ctx(&sp);
     p = spdk_thread_get_from_ctx(ctx);
     p->msg_fd = 110;
     printf("spdk thread msg_fd %d \n",sp.msg_fd);
     return 0;
}
