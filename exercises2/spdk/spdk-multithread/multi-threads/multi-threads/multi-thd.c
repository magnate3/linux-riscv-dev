#include "spdk/nvme.h"
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/thread.h"
#include "spdk_internal/event.h"


static void msg_func(void *ctx)
{
     uint32_t  current_core;
     current_core = spdk_env_get_current_core();
     SPDK_NOTICELOG("%s current core : %u\n",__func__, current_core);
}
/* Main program after the hello is started */
static void hello_start(void *arg) {
       
     uint32_t  i,current_core;
     struct spdk_lw_thread *lw_thread;
     // lw_thread = spdk_thread_get_ctx(thread);
     struct spdk_thread *thread ;
     struct spdk_reactor *reactor;
     current_core = spdk_env_get_current_core();
     SPDK_NOTICELOG("current core : %u\n", current_core);
     SPDK_ENV_FOREACH_CORE(i) {


          if (i != current_core) {
		 reactor = spdk_reactor_get(i);
		 if (reactor == NULL) {
		             continue;
		  }
		  TAILQ_FOREACH(lw_thread, &reactor->threads, link) {

                          thread = spdk_thread_get_from_ctx(lw_thread);
			  spdk_thread_send_msg(thread, msg_func, NULL) ;
		   }
	  }
     }
}

/* cleanup to do after termination of app */
static void cleanup(void) {

}

int main(int argc, char **argv) {
    struct spdk_app_opts opts = {};
    /* Initialize the app event framework */
    spdk_app_opts_init(&opts, sizeof(opts));

    /* Use all four cores, 0xF = 0b1111 */
    opts.reactor_mask = "0x3F";
     SPDK_NOTICELOG("Total cores available: %d\n", spdk_env_get_core_count());
    /* Start the event app framework */
    if (spdk_app_start(&opts, hello_start, NULL)) {
        fprintf(stderr, "Failed to start app framework\n");
        return 1;
    }
    printf("app program successfully terminated\n");
    cleanup();

    spdk_app_fini();
    return 0;
}
