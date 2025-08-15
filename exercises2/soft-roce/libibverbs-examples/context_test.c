#include <stdio.h>
#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

int main(void)
{
    struct ibv_device **device_list;
    int num_devices;
    int i;
    int rc;
    struct ibv_device_attr device_attr;
    device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
        fprintf(stderr, "Error, ibv_get_device_list() failed\n");
        return -1;
    }

    printf("%d RDMA device(s) found:\n\n", num_devices);

    for (i = 0; i < num_devices; ++ i) {
        struct ibv_context *ctx;
        struct ibv_comp_channel *event_channel;

        ctx = ibv_open_device(device_list[i]);
        if (!ctx) {
            fprintf(stderr, "Error, failed to open the device '%s'\n",
                ibv_get_device_name(device_list[i]));
            rc = -1;
            goto out;
        }

        printf("The device '%s' was opened\n", ibv_get_device_name(ctx->device));
        printf("cmd_fd %d and async_fd %d \n", ctx->cmd_fd, ctx->async_fd);

        ibv_query_device (ctx, &device_attr);
        printf("         max_mr_size   : %lu\n", device_attr.max_mr_size);
        printf("         max_mr_size   : %lu\n", device_attr.max_mr_size);
        
        event_channel = ibv_create_comp_channel(ctx);

        if (!event_channel) {
	fprintf(stderr, "Error, ibv_create_comp_channel() failed\n");
           goto err1;	
        }
#if 0
        char buf[1024] = {'\0'};
        char file_path[1024] = {'\0'};
        snprintf(buf,sizeof(buf), "/proc/self/fd/%d", event_channel->fd);
        readlink(buf,file_path,sizeof(file_path)-1);
        printf("event channel fd %d and file_path %s\n", event_channel->fd, file_path);
#endif
        if (ibv_destroy_comp_channel(event_channel)) {
	fprintf(stderr, "Error, ibv_destroy_comp_channel() failed\n");
        goto err1;	
       }
err1:
        rc = ibv_close_device(ctx);
        if (rc) {
            fprintf(stderr, "Error, failed to close the device '%s'\n",
                ibv_get_device_name(ctx->device));
            rc = -1;
            goto out;
        }
    }
        
    ibv_free_device_list(device_list);

    return 0;

out:
    ibv_free_device_list(device_list);
    return rc;
}
