#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <assert.h>
#define SCM_MAX_FD 10
#define N 6
//#define SCM_MAX_FD 253

int
sendfd(int s, int fd)
{
	char buf[1];
	struct iovec iov;
	struct msghdr msg;
	struct cmsghdr *cmsg;
	int n;
	char cms[CMSG_SPACE(sizeof(int))];
	
	buf[0] = 0;
	iov.iov_base = buf;
	iov.iov_len = 1;

	memset(&msg, 0, sizeof msg);
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	msg.msg_control = cms;
	msg.msg_controllen = CMSG_LEN(sizeof(int));

	cmsg = CMSG_FIRSTHDR(&msg);
	cmsg->cmsg_len = CMSG_LEN(sizeof(int));
	cmsg->cmsg_level = SOL_SOCKET;
	cmsg->cmsg_type = SCM_RIGHTS;
	memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

	n = sendmsg(s, &msg, 0);
	if (n != iov.iov_len)
		return -1;
	return 0;
}

int
recvfds(int s, int fds[], int n)
{
	int rc;
	int fd;
	char buf[1];
	struct iovec iov;
	struct msghdr msg;
	struct cmsghdr *cmsg;
	size_t kControlBufferSize = sizeof(int) * n;
	char control_buffer[kControlBufferSize];
	int j;

	iov.iov_base = buf;
	iov.iov_len = 1;

	memset(&msg, 0, sizeof msg);
	msg.msg_name = 0;
	msg.msg_namelen = 0;
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	msg.msg_control = control_buffer;
	msg.msg_controllen = sizeof(control_buffer);

	rc = recvmsg(s, &msg, 0);
	if (rc <= 0) {
		return -1;
	}
	rc = 0;
	for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
		for (j = 0; j < (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int); j++) {
			fds[rc] = ((int *) CMSG_DATA(cmsg))[j];
			rc++;
		}
	}
	assert (!(msg.msg_flags & MSG_TRUNC));
	assert (!(msg.msg_flags & MSG_CTRUNC));
	return rc;
}

int
xpipe(int fd[2]) {
	return socketpair(AF_UNIX, SOCK_SEQPACKET, 0, fd);
}

#ifndef N
#define N (253 + 64)
#endif
static int doorbells[N];
int main(int argc, char *argv[]) {
	int pfd[2];
	unsigned long val;

	int rc;
        int i;
	xpipe(pfd);

	rc = fork();
	switch (rc) {
	case 0: {
		int max_recv_fd[SCM_MAX_FD];
		close(pfd[0]);
		val = 100;
		for (;;) {
			int rc = recvfds(pfd[1], max_recv_fd, SCM_MAX_FD);
			if (rc <= 0)
				break;
			if (rc != 1) printf("BATCH OF %d\n", rc);
			for ( i = 0; i < rc; i++) { 
				write(max_recv_fd[i], &val, sizeof(val));
				val++;
				close(max_recv_fd[i]);
			}
		}
		break;
	}
	default:
		for (i = 0; i < N; i++) {
			doorbells[i] = eventfd(0, 0);
			sendfd(pfd[0], doorbells[i]);
		}
		for (i = 0; i < N; i++) {
			read(doorbells[i], &val, sizeof(val));
			printf("parent val %lu\n", val);
		}
		close(pfd[0]);
		close(pfd[1]);
		wait(NULL);
	}
}
