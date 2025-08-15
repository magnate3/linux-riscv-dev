// inner.c

#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>

#define SOCKET_NAME "/tmp/connection-uds-fd"

int connect_socket()
{
    int fd;
    int result;
    struct sockaddr_un sun;

    fd = socket(PF_UNIX, SOCK_STREAM, 0);

    sun.sun_family = AF_UNIX;
    strncpy(sun.sun_path, SOCKET_NAME, sizeof(sun.sun_path) - 1);

    connect(fd, (struct sockaddr *)&sun, sizeof(sun));
    
    return fd;
}

void receive_fds(int socket_fd)
{
    int dummy = 0;
    int fds[3];
    char input[64];
    struct iovec iov;
    iov.iov_base = &dummy;
    iov.iov_len = 1;

    char buf[CMSG_SPACE(sizeof(fds))];
    
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = buf;
    msg.msg_controllen = sizeof(buf);

    struct cmsghdr *cmsg;
    cmsg             = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len   = CMSG_LEN(sizeof(fds));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type  = SCM_RIGHTS;

    memcpy(CMSG_DATA(cmsg), fds, sizeof(fds));

    int size;
    size = recvmsg(socket_fd, &msg, 0);
    // printf("%d \n", size);

    memcpy(fds, CMSG_DATA(cmsg), sizeof(fds));
    printf("%d %d %d \n", fds[0], fds[1], fds[2]);

    char msg_buf[128];
    strcpy(msg_buf, "write from inner\n");
    write(fds[1], msg_buf, strlen(msg_buf));
    read(fds[0], input, sizeof(input));
    printf("input: %s \n", input);
    write(fds[1], "from client process write : ", strlen("from client process write : "));
    write(fds[1], input, strlen(input));
}

int main(int argc, char const *argv[])
{
    int socket_fd;
    int fds[3];

    socket_fd = connect_socket();
    printf("client socket_fd  %d \n", socket_fd );
    receive_fds(socket_fd);
    getchar();
    return 0;
}
