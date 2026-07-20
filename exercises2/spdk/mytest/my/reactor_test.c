#include <stdio.h>
#include <spdk/bdev.h>
#include <spdk/thread.h>
#include <spdk/queue.h>
#include "spdk/thread.h"
#include "spdk/env.h"
#include "spdk/event.h"
/*
 *  * Usage function for printing parameters that are specific to this application
 *   */
static void
hello_reacor_usage(void)
{
	        printf(" -b <bdev>                 name of the bdev to use\n");
}

/*
 *  * This function is called to parse the parameters that are specific to this application
 *   */
static int
hello_reacor_parse_arg(int ch, char *arg)
{
       switch (ch) {
	        case 'b':
                   //g_bdev_name = arg;
                break;
	        default:
                return -EINVAL;
        }
        return 0;
}
// refer to accel_perf_start(void *arg1)
static void hello_start(void *arg1)
{
    struct spdk_thread* first_reader_thread =
            spdk_thread_create("first_reader_thread", NULL);

    if (first_reader_thread == NULL)
    {
        printf("First thread creation failed...\n");
        return ;
    }

    struct spdk_thread* second_reader_thread =
            spdk_thread_create("second_reader_thread", NULL);
    if (second_reader_thread == NULL)
    {
        printf("Second thread creation failed...\n");
        return ;
    }

    printf("first reader thread id is: %"PRIu64"\n",
           spdk_thread_get_id(first_reader_thread));
    printf("second reader thread id is: %"PRIu64"\n",
           spdk_thread_get_id(second_reader_thread));
#if 0
    spdk_app_stop(0);
#endif
}
int main(int argc, char **argv)
{
    struct spdk_app_opts opts = {};
    int rc;
    /* Set default values in opts structure. */
    spdk_app_opts_init(&opts, sizeof(opts));
	            opts.name = "hello_reactor";
    if ((rc = spdk_app_parse_args(argc, argv, &opts, "b:", NULL, hello_reacor_parse_arg, hello_reacor_usage)) != SPDK_APP_PARSE_ARGS_SUCCESS) {
	                      exit(rc);
    }
    rc = spdk_app_start(&opts, hello_start, NULL);
    if (rc) {
              SPDK_ERRLOG("ERROR starting application\n");
    }
    printf("Hello World!\n");
cleanup:
    spdk_app_fini();
    return 0;
}
