#include "spdk/nvme.h"
#include "spdk/env.h"
#include "spdk/event.h"

/* This program demonstrates the multithreaded hello world 
 * using the event framework provided in the SPDK library */

static int rc_count = 0;

/* Main context struct to pass to different callback functions */
struct io_ctx {
    struct spdk_nvme_qpair *qpair;
    bool is_on_cmb;
    void *buffer;
    size_t load;
    uint64_t starting_lba;
    bool is_complete;
    int core_idx;
    char *msg;
};

/* Initalize I/O messages */
static char msg2[] = "Core 1 reporting!";
static char msg3[] = "Core 2 reporting!";
static char msg4[] = "Core 3 reporting!";
static char *msgs[3] = {msg2, msg3, msg4};
static size_t loads[3] = {sizeof(msg2), sizeof(msg3), sizeof(msg4)};

static struct spdk_app_opts app_opts;

/* To simplify, only use one namespace for the following operation */
static struct spdk_nvme_ctrlr *nvme_ctrlr = NULL;
static struct spdk_nvme_ns *nvme_ns = NULL;

/* Callback on probing NVME controller */
static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
        struct spdk_nvme_ctrlr_opts *opts) {
    /* Only attach a controller if a valid namespace
     * has not been defined */
    return (nvme_ns == NULL);
}

/* Callback function on detaching an NVME controller */
static void remove_cb(void *cb_ctx, struct spdk_nvme_ctrlr *ctrlr) {
    printf("The device with transfer address %s has been detached\n", 
            spdk_nvme_ctrlr_get_transport_id(ctrlr)->traddr);
}

/* Callback on attaching an NVME controller */
static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
        struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts) {
    printf("The controller %s has been attached\n", trid->traddr);
    int first_active_ns_id;
    if ((first_active_ns_id = spdk_nvme_ctrlr_get_first_active_ns(ctrlr)) == 0) {
        printf("There is no active namespace on controller %s\n", trid->traddr);
        /* Detach this nmve contoller because it does not have a valid namespaace */
        if (spdk_nvme_detach(ctrlr)) {
            printf("Failed to detach controller\n");
        }
        return;
    }
    /* register the chosen controller and namespace in the global pointer */
    nvme_ns = spdk_nvme_ctrlr_get_ns(ctrlr, first_active_ns_id);
    nvme_ctrlr = ctrlr;
    printf("Namespace %d of NVME controller %s is used, "
            "future controllers will be detached immediately\n", 
            first_active_ns_id, trid->traddr);
}

/* Intialize ONE nvme controller and the associated ONE activae namespace */
static void ctrlr_ns_init(void) {
    if (spdk_nvme_probe(NULL, NULL, probe_cb, attach_cb, remove_cb)) {
        fprintf(stderr, "Failed to probe for nvme controllers\n");
        exit(1);
    }
}

/* Threads report complete status to thread 0 */
static void report_complete(void *arg1, void *arg2) {
    struct io_ctx *ctx = arg1;

    printf("%s\n", (char *) ctx->buffer);

    /* Deallocate read buffer */
    if (ctx->is_on_cmb) {
        spdk_nvme_ctrlr_free_cmb_io_buffer(nvme_ctrlr, ctx->buffer, ctx->load);
        printf("Read buffer for core %d on cmb is deallocated\n", ctx->core_idx);
    } else {
        spdk_free(ctx->buffer);
        printf("Read buffer for core %d on host memory is deallocated\n", ctx->core_idx);
    }

    rc_count++;
    if (rc_count == 3) {
        spdk_app_stop(0);
    }
}

/* Callback on successful read */
static void read_complete(void *arg, const struct spdk_nvme_cpl *cpl) {
    struct io_ctx *ctx = arg;

    /* Mark task as complete */
    ctx->is_complete = true;
}

/* Callback on successful write */
static void write_complete(void *arg, const struct spdk_nvme_cpl *cpl) {
    struct io_ctx *ctx = arg;

    /* Deallocate write buffer */
    if (ctx->is_on_cmb) {
        spdk_nvme_ctrlr_free_cmb_io_buffer(nvme_ctrlr, ctx->buffer, ctx->load);
        printf("Write buffer on cmb on core %d is deallocated\n", ctx->core_idx);
    } else {
        spdk_free(ctx->buffer);
        printf("Write buffer on host memory on core %d is deallocated\n", ctx->core_idx);
    }

    /* After write complete, do read */
    /* Allocate buffer on CMB or host for read */
    void *read_buffer = NULL;
    if ((read_buffer = spdk_nvme_ctrlr_alloc_cmb_io_buffer(nvme_ctrlr, ctx->load))) {
        ctx->is_on_cmb = true;
        printf("Read buffer for core %d is now on cmb\n", ctx->core_idx);
    } else {
        read_buffer = spdk_malloc(ctx->load, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
        printf("Read buffer for core %d is on host memory\n", ctx->core_idx);
    }
    ctx->buffer = read_buffer;

    /* Submit read command */
    if (spdk_nvme_ns_cmd_read(nvme_ns, ctx->qpair, read_buffer, ctx->starting_lba, 1,
                read_complete, ctx, 0)) {
        fprintf(stderr, "Submit read command failed\n");
    }
}

/* Event function that prints a message */
static void print_msg(void *arg1, void *arg2) {
    struct io_ctx *ctx = (struct io_ctx *) arg1;

    /* Allocate queue pair */
    struct spdk_nvme_qpair *qpair = 
        spdk_nvme_ctrlr_alloc_io_qpair(nvme_ctrlr, NULL, 0);
    ctx->qpair = qpair;

    /* Allcoate buffer on CMB or host for write */
    void *write_buffer = NULL;
    if ((write_buffer = spdk_nvme_ctrlr_alloc_cmb_io_buffer(nvme_ctrlr, ctx->load))) {
        ctx->is_on_cmb = true;
        printf("Write buffer for core %d is on cmb\n", ctx->core_idx);
    } else {
        write_buffer = spdk_malloc(ctx->load, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
        printf("Write buffer for core %d is on host memory\n", ctx->core_idx);
    }
    ctx->buffer = write_buffer;

    /* Write message to the allocated memory buffer */
    memcpy(write_buffer, ctx->msg, ctx->load);

    /* Submit write command */
    if (spdk_nvme_ns_cmd_write(nvme_ns, qpair, write_buffer, ctx->starting_lba, 
                1, write_complete, ctx, 0)) {
        fprintf(stderr, "Failure in submitting write command on core %d\n", ctx->core_idx);
    }
    printf("Submitted write command on core %d to sector %ld\n", ctx->core_idx, ctx->starting_lba);

    /* Continous process qpair until the two jobs are complete */
    while (!ctx->is_complete) {
        if (spdk_nvme_qpair_process_completions(qpair, 0) < 0) {
            fprintf(stderr, "Error occurred in processing queue pairs on core %d", ctx->core_idx);
        }
    }

    /* Notify thread 0 of completition */
    struct spdk_event *event = spdk_event_allocate(0, report_complete, ctx, NULL);
    spdk_event_call(event);
}

/* Prepare tasks for each reactor */
static struct io_ctx ctxs[3];

/* Prepare four events to the four reactors */
static struct spdk_event *io_events[3];

/* Main program after the hello is started */
static void hello_start(void *arg) {
    /* Intialize NMVe controller and namespace */
    ctrlr_ns_init();
    ctxs[0].is_complete = false;
    ctxs[0].starting_lba = 0;
    ctxs[0].msg = msgs[0];
    ctxs[0].is_on_cmb = false;
    ctxs[0].load = loads[0];
    ctxs[0].core_idx = spdk_env_get_next_core(spdk_env_get_first_core());
    io_events[0] = spdk_event_allocate(ctxs[0].core_idx, print_msg, ctxs, NULL);
    for (int i = 1; i < 3; i++) {
        ctxs[i].is_complete = false;
        ctxs[i].starting_lba = i;
        ctxs[i].msg = msgs[i];
        ctxs[i].is_on_cmb = false;
        ctxs[i].load = loads[i];
        ctxs[i].core_idx = spdk_env_get_next_core(ctxs[i - 1].core_idx);
        io_events[i] = spdk_event_allocate(ctxs[i].core_idx, print_msg, ctxs + i, NULL);
    }
    for (int i = 0; i < 3; i++) {
        spdk_event_call(io_events[i]);
    }
}

/* cleanup to do after termination of app */
static void cleanup(void) {
    /* Detach the controller after succesful termination */
    if (spdk_nvme_detach(nvme_ctrlr)) {
        fprintf(stderr, "Failed to detach controller %s\n", 
                spdk_nvme_ctrlr_get_transport_id(nvme_ctrlr)->traddr);
    } else {
        printf("Detached NVME driver %s\n",
                spdk_nvme_ctrlr_get_transport_id(nvme_ctrlr)->traddr);
    }
}

int main(int argc, char **argv) {
    /* Initialize the app event framework */
    spdk_app_opts_init(&app_opts);

    /* Use all four cores, 0xF = 0b1111 */
    app_opts.reactor_mask = "0xF";

    /* Start the event app framework */
    if (spdk_app_start(&app_opts, hello_start, NULL)) {
        fprintf(stderr, "Failed to start app framework\n");
        return 1;
    }
    printf("app program successfully terminated\n");
    cleanup();

    spdk_app_fini();
    return 0;
}
