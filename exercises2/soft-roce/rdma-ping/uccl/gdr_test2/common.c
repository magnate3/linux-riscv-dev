#include "common.h"

#include <string.h>

void parse_arguments(int argc, char **argv, enum mode_enum *mode, int *tcp_port)
{
    if (argc < 2) {
        printf("usage: %s <rpc|queue> [tcp port]\n", argv[0]);
        exit(1);
    }

    if (strcmp(argv[1], "rpc") == 0) {
        *mode = MODE_RPC_SERVER;
    } else if (strcmp(argv[1], "queue") == 0) {
        *mode = MODE_QUEUE;
    } else {
        printf("Unknown mode '%s'\n", argv[1]);
        exit(1);
    }

    if (argc < 3) {
        *tcp_port = 0;
    } else {
        *tcp_port = atoi(argv[2]);
    }
}
