/*************************************************************************
  > File Name:       hello_nvme_bdev.c
  > Author:          Zeyuan Hu
  > Mail:            iamzeyuanhu@utexas.edu
  > Created Time:    8/25/18
  > Description:
    
    This program is very much the same as the hello_bdev.c provided in SPDK.
    The program demos how to create a block device and write a hello world message into it
    and read the message from it. hello_bdev.c provides a unified interface that can support
    various underlying device (i.e. bdev). It is done by reading in a configuration file,
    which contains various block device (e.g. Passthru, Malloc, NVMe)

    - The explanation of configuration file:

        https://github.com/spdk/spdk/blob/master/etc/spdk/nvmf.conf.in

    - The explanation of various block devices:

        http://www.spdk.io/doc/bdev.html#bdev_ug_introduction


    We modify hello_bdev.c by use NVMe bdev by default and adjust the configuration file accordingly.
    This program is also used as a testbed for the project build system.
 ************************************************************************/

#include "spdk/stdinc.h"
#include "spdk/thread.h"
#include "spdk/bdev.h"
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/log.h"
#include "spdk/string.h"
#include "spdk/bdev_module.h"
//#include "spdk/bdev_zone.h"
/*
 * hzy: we use the NVMe bdev by default.
 */
//static char *g_bdev_name = "Nvme0n1";
static char *g_bdev_name = "Malloc0";

/*
 * We'll use this struct to gather housekeeping hello_context to pass between
 * our events and callbacks.
 */
struct hello_context_t
{
    struct spdk_bdev *bdev;
    struct spdk_bdev_desc *bdev_desc;
    struct spdk_io_channel *bdev_io_channel;
    char *buff;
    char *bdev_name;
};

/*
 * Usage function for printing parameters that are specific to this application
 */
static void
hello_bdev_usage(void)
{
    printf(" -b <bdev>                 name of the bdev to use\n");
}
#if 0
static void
hello_bdev_event_cb(enum spdk_bdev_event_type type, struct spdk_bdev *bdev,
		                    void *event_ctx)
{
	        SPDK_NOTICELOG("Unsupported bdev event: type %d\n", type);
}
#endif
/*
 * This function is called to parse the parameters that are specific to this application
 */
static int 
hello_bdev_parse_arg(int ch, char *arg)
{
    switch (ch)
    {
        case 'b':
 	   g_bdev_name = arg;
           break;
	default:
      	 return -EINVAL;
    }
    return 0;
}

/*
 * Callback function for read io completion.
 */
static void
read_complete(struct spdk_bdev_io *bdev_io, bool success, void *cb_arg)
{
    struct hello_context_t *hello_context = cb_arg;

    if (success)
    {
        SPDK_NOTICELOG("Read string from bdev : %s\n", hello_context->buff);
    }
    else
    {
        SPDK_ERRLOG("bdev io read error\n");
    }

    /* Complete the bdev io and close the channel */
    spdk_bdev_free_io(bdev_io);
    spdk_put_io_channel(hello_context->bdev_io_channel);
    spdk_bdev_close(hello_context->bdev_desc);
    SPDK_NOTICELOG("Stopping app\n");
    spdk_app_stop(success ? 0 : -1);
}

/*
 * Callback function for write io completion.
 */
static void
write_complete(struct spdk_bdev_io *bdev_io, bool success, void *cb_arg)
{
    struct hello_context_t *hello_context = cb_arg;
    int rc;
    uint32_t blk_size;

    /* Complete the I/O */
    spdk_bdev_free_io(bdev_io);

    if (success)
    {
        SPDK_NOTICELOG("bdev io write completed successfully\n");
    }
    else
    {
        SPDK_ERRLOG("bdev io write error: %d\n", EIO);
        spdk_put_io_channel(hello_context->bdev_io_channel);
        spdk_bdev_close(hello_context->bdev_desc);
        spdk_app_stop(-1);
        return;
    }

    /* Zero the buffer so that we can use it for reading */
    blk_size = spdk_bdev_get_block_size(hello_context->bdev);
    memset(hello_context->buff, 0, blk_size);

    SPDK_NOTICELOG("Reading io\n");
    rc = spdk_bdev_read(hello_context->bdev_desc, hello_context->bdev_io_channel,
                        hello_context->buff, 0, blk_size, read_complete, hello_context);

    if (rc)
    {
        SPDK_ERRLOG("%s error while reading from bdev: %d\n", spdk_strerror(-rc), rc);
        spdk_put_io_channel(hello_context->bdev_io_channel);
        spdk_bdev_close(hello_context->bdev_desc);
        spdk_app_stop(-1);
        return;
    }
}

/*
 * Our initial event that kicks off everything from main().
 */
static void
hello_start(void *arg1)
{
    struct hello_context_t *hello_context = arg1;
    uint32_t blk_size, buf_align;
    int rc = 0;
    hello_context->bdev = NULL;
    hello_context->bdev_desc = NULL;

    SPDK_NOTICELOG("Successfully started the application\n");
#if 1
    /*
     * hzy: here, we list all possible bdev that we can use.
     */
    for (struct spdk_bdev *first = spdk_bdev_first(); first != NULL; first = spdk_bdev_next(first))
    {
        //SPDK_NOTICELOG("bdev name: %s\n", first->name);
        SPDK_ERRLOG("bdev name: %s\n", first->name);
        SPDK_NOTICELOG("bdev product_name: %s\n", first->product_name);
        SPDK_NOTICELOG("bdev module name: %s\n", first->module->name);
    }

    /*
     * Get the bdev. There can be many bdevs configured in
     * in the configuration file but this application will only
     * use the one input by the user at runtime so we get it via its name.
     */
    hello_context->bdev = spdk_bdev_get_by_name(hello_context->bdev_name);
    if (hello_context->bdev == NULL)
    {
        SPDK_ERRLOG("Could not find the bdev: %s\n", hello_context->bdev_name);
        spdk_app_stop(-1);
        return;
    }

    /*
     * Open the bdev by calling spdk_bdev_open()
     * The function will return a descriptor
     */
    SPDK_NOTICELOG("Opening the bdev %s\n", hello_context->bdev_name);
    rc = spdk_bdev_open(hello_context->bdev, true, NULL, NULL, &hello_context->bdev_desc);
    if (rc)
    {
        SPDK_ERRLOG("Could not open bdev: %s\n", hello_context->bdev_name);
        spdk_app_stop(-1);
        return;
    }
#else

            /*
	     *          * There can be many bdevs configured, but this application will only use
	     *                   * the one input by the user at runtime.
	     *                            *
	     *                                     * Open the bdev by calling spdk_bdev_open_ext() with its name.
	     *                                              * The function will return a descriptor
	     *                                                       */
           SPDK_NOTICELOG("Opening the bdev %s\n", hello_context->bdev_name);
	   rc = spdk_bdev_open_ext(hello_context->bdev_name, true, hello_bdev_event_cb, NULL,
		                                    &hello_context->bdev_desc);
	   if (rc) {
	           SPDK_ERRLOG("Could not open bdev: %s\n", hello_context->bdev_name);
	           spdk_app_stop(-1);
	            return;
	    }

	   /* A bdev pointer is valid while the bdev is opened. */
	   hello_context->bdev = spdk_bdev_desc_get_bdev(hello_context->bdev_desc);
#endif
    SPDK_NOTICELOG("Opening io channel\n");
    /* Open I/O channel */
    hello_context->bdev_io_channel = spdk_bdev_get_io_channel(hello_context->bdev_desc);
    if (hello_context->bdev_io_channel == NULL)
    {
        SPDK_ERRLOG("Could not create bdev I/O channel!!\n");
        spdk_bdev_close(hello_context->bdev_desc);
        spdk_app_stop(-1);
        return;
    }

    /* Allocate memory for the write buffer.
     * Initialize the write buffer with the string "Hello World!"
     */
    blk_size = spdk_bdev_get_block_size(hello_context->bdev);
    buf_align = spdk_bdev_get_buf_align(hello_context->bdev);
    hello_context->buff = spdk_dma_zmalloc(blk_size, buf_align, NULL);
    if (!hello_context->buff)
    {
        SPDK_ERRLOG("Failed to allocate buffer\n");
        spdk_put_io_channel(hello_context->bdev_io_channel);
        spdk_bdev_close(hello_context->bdev_desc);
        spdk_app_stop(-1);
        return;
    }
    snprintf(hello_context->buff, blk_size, "%s", "Hello World!\n");

    SPDK_NOTICELOG("Writing to the bdev\n");
    rc = spdk_bdev_write(hello_context->bdev_desc, hello_context->bdev_io_channel,
                         hello_context->buff, 0, blk_size, write_complete, hello_context);
    if (rc)
    {
        SPDK_ERRLOG("%s error while writing to bdev: %d\n", spdk_strerror(-rc), rc);
        spdk_bdev_close(hello_context->bdev_desc);
        spdk_put_io_channel(hello_context->bdev_io_channel);
        spdk_app_stop(-1);
        return;
    }
}

int
main(int argc, char **argv)
{
    struct spdk_app_opts opts = {};
    int rc = 0;
    struct hello_context_t hello_context = {};

    /* Set default values in opts structure. */
    spdk_app_opts_init(&opts,sizeof(opts));
    opts.name = "hello_bdev";
    //opts.config_file = "bdev.json";

    /*
     * The user can provide the config file and bdev name at run time.
     * For example, to use Malloc0 in file bdev.conf run with params
     * ./hello_bdev -c bdev.conf -b Malloc0
     * To use passthru bdev PT0 run with params
     * ./hello_bdev -c bdev.conf -b PT0
     * hzy: If none of the parameters are provide the application will use the
     * default parameters(-c bdev.conf -b Nvme0n1).
     */
    if ((rc = spdk_app_parse_args(argc, argv, &opts, "b:", NULL,hello_bdev_parse_arg,
                                  hello_bdev_usage)) != SPDK_APP_PARSE_ARGS_SUCCESS)
    {
        SPDK_ERRLOG("ERROR app parse \n");
        exit(rc);
    }
    hello_context.bdev_name = g_bdev_name;

    /*
     * spdk_app_start() will block running hello_start() until
     * spdk_app_stop() is called by someone (not simply when
     * hello_start() returns), or if an error occurs during
     * spdk_app_start() before hello_start() runs.
     */
    rc = spdk_app_start(&opts, hello_start, &hello_context);
    if (rc)
    {
        SPDK_ERRLOG("ERROR starting application\n");
    }

    /* When the app stops, free up memory that we allocated. */
    spdk_dma_free(hello_context.buff);

    /* Gracefully close out all of the SPDK subsystems. */
    spdk_app_fini();
    return rc;
}
