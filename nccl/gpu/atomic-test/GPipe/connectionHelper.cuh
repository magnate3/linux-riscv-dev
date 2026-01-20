#ifndef CONNECTION_HELPER
#define CONNECTION_HELPER

#include <sys/socket.h>
#include <sys/un.h>

static void report_error(const char* message)
{
	perror(message);
	exit(-1);
}

void ipcCloseSocket(int socket_fd, const char* socket_name)
{
	unlink(socket_name);
	close(socket_fd);
}

static struct sockaddr_un create_socket_address(const char* socket_name)
{
	struct sockaddr_un addr;
	addr.sun_family = AF_UNIX;
	strcpy(addr.sun_path, socket_name);
	return addr;
}

static int create_socket(const char* socket_name)
{
	/*
	 * open Unix domain server socket
	 */
	int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		report_error("Failed on create socket");

	struct sockaddr_un addr = create_socket_address(socket_name);
	if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
		report_error("Failed on bind");

	if (listen(sockfd, 1) < 0)
		report_error("Failed on listen");

	return sockfd;
}

static int recv_arguments(const char* socket_name, void* arguments, size_t arguments_size)
{
	int				client;
	struct msghdr	msg;
    struct iovec	iov;
	struct cmsghdr *cmsg;
	char			cmsg_buf[CMSG_SPACE(sizeof(int))];
	int				ipc_handle = -1;

	// sleep(3);
	struct sockaddr_un addr = create_socket_address(socket_name);

	/* connect to the server socket */
	client = socket(AF_UNIX, SOCK_STREAM, 0);
	if (client < 0)
		report_error("Failed on socket in recv_arguments function");

	// TODO: check for better ways to sync socket creation
	while (connect(client, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) != 0) {}
		// report_error("Failed on connect in recv_arguments function");

	/* receive a file descriptor using SCM_RIGHTS */
	memset(&msg, 0, sizeof(msg));
	msg.msg_control = cmsg_buf;
	msg.msg_controllen = sizeof(cmsg_buf);

	iov.iov_base = arguments;
	iov.iov_len = arguments_size;

	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;

	if (recvmsg(client, &msg, 0) <= 0)
		report_error("Failed on 'recvmsg' in recv_arguments function");

	cmsg = CMSG_FIRSTHDR(&msg);
	if (!cmsg)
		dbg_printf("message has no control header");

	if (cmsg->cmsg_len == CMSG_LEN(sizeof(int)) &&
		cmsg->cmsg_level == SOL_SOCKET &&
		cmsg->cmsg_type == SCM_RIGHTS)
	{
		memcpy(&ipc_handle, CMSG_DATA(cmsg), sizeof(int));
	}
	else
		dbg_printf("unexpected control header");

	ipcCloseSocket(client, socket_name);
	return ipc_handle;
}

static inline void send_args(int sockfd, int ipc_handle, void* arguments, size_t arguments_size)
{
	int				client;
	struct msghdr	msg;
	struct iovec	iov;
	char			cmsg_buf[CMSG_SPACE(sizeof(int))];
	struct cmsghdr *cmsg;

	// TODO: fix this. why sleep??????
	sleep(1);
	client = accept(sockfd, NULL, NULL);
	if (client < 0)
		report_error("failed on accept in send args function");

	/* send a file descriptor using SCM_RIGHTS */
	memset(&msg, 0, sizeof(msg));
	msg.msg_control = cmsg_buf;
	msg.msg_controllen = sizeof(cmsg_buf);

	cmsg = CMSG_FIRSTHDR(&msg);
	cmsg->cmsg_len = CMSG_LEN(sizeof(int));
	cmsg->cmsg_level = SOL_SOCKET;
	cmsg->cmsg_type = SCM_RIGHTS;

	memcpy(CMSG_DATA(cmsg), &ipc_handle, sizeof(ipc_handle));

	iov.iov_base = arguments;
	iov.iov_len = arguments_size;
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;

	if (sendmsg(client, &msg, 0) < 0)
		report_error("failed on send msg");

	close(client);
}


#endif
