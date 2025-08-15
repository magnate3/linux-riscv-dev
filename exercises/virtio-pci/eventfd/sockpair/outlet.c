// outlet.c

#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>

#define SOCKET_NAME "/tmp/connection-uds-fd"

int make_connection()
{
    int fd;
    int connection_fd;
    int result;
    struct sockaddr_un sun;
    
    fd = socket(PF_UNIX, SOCK_STREAM, 0);

    printf("server fd %d \n", connection_fd);
    sun.sun_family = AF_UNIX;
    strncpy(sun.sun_path, SOCKET_NAME, sizeof(sun.sun_path) - 1);

    result = bind(fd, (struct sockaddr*) &sun, sizeof(sun));
    listen(fd, 10);
    connection_fd = accept(fd, NULL, NULL);    

    return connection_fd;
}

void send_fds(int socket_fd)
{
    struct msghdr msg;
    struct cmsghdr *cmsg = NULL;
    //stdin, stdout, and stderr are 0, 1, and 2,
    int io[3] = { 0, 1, 2 }; //  send data
    char buf[CMSG_SPACE(sizeof(io))];
    struct iovec iov;
    int dummy;

    memset(&msg, 0, sizeof(struct msghdr));

    iov.iov_base = &dummy;
    iov.iov_len = 1;

    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len = CMSG_LEN(sizeof(io));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;

    memcpy(CMSG_DATA(cmsg), io, sizeof(io));

    msg.msg_controllen = cmsg->cmsg_len;

    int size;
    size = sendmsg(socket_fd, &msg, 0);
    // printf("%d %d\n", size, errno);
}

int main(int argc, char const *argv[])
{
    int connection_fd;
    int count = 3;
    printf("write from outlet\n");
    connection_fd = make_connection();
    printf("connection_fd %d \n", connection_fd);
    send_fds(connection_fd);

    while(1) {
        sleep(1);
        if (--count >= 0)
            printf("server output %d \n", count);
    }

    return 0;
}
