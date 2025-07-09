/*
 * Copyright (c) 2005 Mellanox Technologies. All rights reserved.
 *
 */
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

#include "rdma_mt.h"
#include "rdma_utils.h"


#ifndef DEBUG
#define DEBUG_ 0
#endif
#ifndef DEBUG_DATA
#define DEBUG_DATA 0
#endif

#define MEASUREMENTS 200
#define USECSTEP 10
#define USECSTART 100

/*
 Use linear regression to calculate cycles per microsecond.
 http://en.wikipedia.org/wiki/Linear_regression#Parameter_estimation
*/
static double sample_get_cpu_mhz(void)
{
	struct timeval tv1, tv2;
	cycles_t start;
	double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
	double tx, ty;
	int i;

	/* Regression: y = a + b x */
	long x[MEASUREMENTS];
	cycles_t y[MEASUREMENTS];
	double a; /* system call overhead in cycles */
	double b; /* cycles per microsecond */
	double r_2;

	for (i = 0; i < MEASUREMENTS; ++i) {
		start = get_cycles();

		if (gettimeofday(&tv1, NULL)) {
			ERROR("gettimeofday failed.\n");
			return 0;
		}

		do {
			if (gettimeofday(&tv2, NULL)) {
				ERROR("gettimeofday failed.\n");
				return 0;
			}
		} while ((tv2.tv_sec - tv1.tv_sec) * 1000000 +
			(tv2.tv_usec - tv1.tv_usec) < USECSTART + i * USECSTEP);

		x[i] = (tv2.tv_sec - tv1.tv_sec) * 1000000 +
			tv2.tv_usec - tv1.tv_usec;
		y[i] = get_cycles() - start;
		if (DEBUG_DATA)
			ERROR("x=%ld y=%Ld.\n", x[i], (long long)y[i]);
	}

	for (i = 0; i < MEASUREMENTS; ++i) {
		tx = x[i];
		ty = y[i];
		sx += tx;
		sy += ty;
		sxx += tx * tx;
		syy += ty * ty;
		sxy += tx * ty;
	}
	
	b = (MEASUREMENTS * sxy - sx * sy) / (MEASUREMENTS * sxx - sx * sx);
	a = (sy - b * sx) / MEASUREMENTS;


	DEBUG("a = %g\n", a);
	DEBUG("b = %g\n", b);
	DEBUG("a / b = %g\n", a / b);
	
	r_2 = (MEASUREMENTS * sxy - sx * sy) * (MEASUREMENTS * sxy - sx * sy) /
		(MEASUREMENTS * sxx - sx * sx) /
		(MEASUREMENTS * syy - sy * sy);


	DEBUG("r^2 = %g\n", r_2);
	if (r_2 < 0.9) {
		ERROR("Correlation coefficient r^2: %g < 0.9.\n", r_2);
		return 0;
	}

	return b;
}

static double proc_get_cpu_mhz(int no_cpu_freq_fail)
{
	FILE* f;
	char buf[256];
	double mhz = 0.0;

	f = fopen("/proc/cpuinfo","r");
	if (!f)
		return 0.0;
	while(fgets(buf, sizeof(buf), f)) {
		double m;
		int rc;
		rc = sscanf(buf, "cpu MHz : %lf", &m);
		if (rc != 1) {	/* PPC has a different format */
			rc = sscanf(buf, "clock : %lf", &m);
			if (rc != 1)
				continue;
		}
		if (mhz == 0.0) {
			mhz = m;
			continue;
		}
		if (mhz != m) {
			ERROR("Conflicting CPU frequency values detected: %lf != %lf.\n", mhz, m);
			if (no_cpu_freq_fail) {
				ERROR("Test integrity may be harmed!\n");
			}else{
				return 0.0;
			}
			continue;
		}
	}
	fclose(f);
	return mhz;
}


double get_cpu_mhz(int no_cpu_freq_fail)
{
	double sample, proc, delta;
	sample = sample_get_cpu_mhz();
	proc = proc_get_cpu_mhz(no_cpu_freq_fail);

	if (!proc || !sample)
		return 0;

	delta = proc > sample ? proc - sample : sample - proc;
	if (delta / proc > 0.01) {
			ERROR("Warning: measured timestamp frequency %g differs from nominal %g MHz.\n", sample, proc);
			return sample;
	}
	return proc;
}


void sock_init(struct sock_t *sock)
{
    sock->sock_fd = -1;
}


int close_sock_fd(int sock_fd)
{
    int rc;

    if (sock_fd == -1) {
        ERROR("sock_fd is not allocated.\n");
        return -1;
    }

    rc = close(sock_fd);
    if (rc != 0) {
        ERROR("close socket failed: %s.\n", strerror(errno));
        return -1;
    }

    return 0;
}


static int bind_close(struct sock_bind_t *sock_bind)
{
    return close_sock_fd(sock_bind->socket_fd);
}


static int sock_daemon_wait(unsigned short port, struct sock_bind_t *sock_bind, struct sock_t *sock)
{
	struct sockaddr_in remote_addr;
	struct protoent *protoent;
	struct sockaddr_in my_addr;
    socklen_t addr_len;
    int rc;
    int tmp;

    sock->port = port;
    sock->is_daemon = 1;

    if (sock_bind->counter == 0) {
        sock_bind->socket_fd = socket(PF_INET, SOCK_STREAM, 0);
        if (sock_bind->socket_fd == -1) {
            ERROR("socket failed, reason: %s.\n", strerror(errno));
            return -1;
        }
        DEBUG(("TCP socket was created.\n"));

        tmp = 1;
        setsockopt (sock_bind->socket_fd, SOL_SOCKET, SO_REUSEADDR, (char *) &tmp, sizeof (tmp));

        my_addr.sin_family = PF_INET;
        my_addr.sin_port = htons(port);
        my_addr.sin_addr.s_addr = INADDR_ANY;
        rc = bind(sock_bind->socket_fd, (struct sockaddr *)(&my_addr), sizeof(my_addr));
        if (rc == -1) {
            ERROR("bind failed: %s.\n", strerror(errno));
            return -1;
        }
        DEBUG("Start listening on port %u.\n",  port);

        rc = listen(sock_bind->socket_fd, 1);
        if (rc == -1) {
            ERROR("listen failed: %s.\n", strerror(errno));
            return -1;
        }
    }

    addr_len = sizeof(remote_addr);
    sock->sock_fd = accept(sock_bind->socket_fd, (struct sockaddr *)(&remote_addr), &addr_len);
    if (sock->sock_fd == -1) {
        ERROR("accept failed: %s.\n", strerror(errno));
        return -1;
    }
    DEBUG("Accepted connection from IP = %s and port = %u.\n", inet_ntoa(remote_addr.sin_addr) ,remote_addr.sin_port);

    protoent = getprotobyname ("tcp");
    if (!protoent) {
        ERROR("getprotobyname error.\n");
        return -1;
    }

    tmp = 1;
    setsockopt (sock->sock_fd, protoent->p_proto, TCP_NODELAY, (char *) &tmp, sizeof (tmp));
    signal (SIGPIPE, SIG_IGN);  /* If we don't do this, then gdbserver simply exits when the remote side dies.  */
    sock_bind->counter ++;

    DEBUG("Connection was established on port %u, socket_fd = 0x%x.\n", port, sock->sock_fd);
    return 0;
}


static int sock_client_connect(char *ip, unsigned short port, struct sock_t *sock)
{
    struct in_addr remote_addr;
	struct sockaddr_in server_addr;
    int rc;
    int i;

    sock->sock_fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock->sock_fd == -1) {
        ERROR("socket failed: %s.\n", strerror(errno));
        return -1;
    }

    sock->port = port;
    strncpy(sock->ip, ip, IP_STR_LENGTH);
    if (inet_aton(ip,  &remote_addr) == 0) {
        ERROR("inet_aton failed: %s.\n", strerror(errno));
        return -1;
    }

    server_addr.sin_family = PF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr = remote_addr;
    DEBUG("Connecting to IP %s, port %u.\n", sock->ip, port);

    i = 0;
    do {
        rc = connect(sock->sock_fd, (struct sockaddr *)(&server_addr), sizeof(server_addr));
        if (rc == -1) {
            sleep(1);
            DEBUG("connect[%d] failed, reason %s.\n", i, strerror(errno));
        }

        i ++;

    } while ((i < SOCKET_RETRY) &&  (rc == -1));

    if (rc == -1) {
        ERROR("Connecting failed: %s.\n", strerror(errno));
        return -1;
    }

    DEBUG("Connection was established to IP %s, port %u, socket_fd = 0x%x.\n", sock->ip, port, sock->sock_fd);
    sock->is_daemon = 0;
    return 0;
}


int sock_connect(struct sock_bind_t *sock_bind, struct sock_t *sock)
{
    int rc;

    if (sock_bind->is_daemon) {
        rc = sock_daemon_wait(sock_bind->port, sock_bind, sock);
        rc |= bind_close(sock_bind); /// bug
    } else {
        rc = sock_client_connect(sock_bind->ip, sock_bind->port, sock);
    }

    return rc;
}


int sock_close(struct sock_t *sock)
{
    return close_sock_fd(sock->sock_fd);
}


int sock_connect_multi(struct sock_bind_t *sock_bind, struct sock_t *sock)
{
    int rc;

    if (sock_bind->is_daemon) {
		DEBUG("Running sock_daemon_wait.\n");
        rc = sock_daemon_wait(sock_bind->port, sock_bind, sock);
    } else {
    	DEBUG("Running sock_client_connect, Server IP: %s\n", sock_bind->ip);
        rc = sock_client_connect(sock_bind->ip, sock_bind->port, sock);
    }

    return rc;
}


int sock_close_multi(struct sock_t *sock, struct sock_bind_t *sock_bind)
{
    int rc;

	ERROR("Closing sock->sock_fd.\n");
    rc = close_sock_fd(sock->sock_fd);
    if (sock->is_daemon) {
        if (sock_bind->counter == 0) {
            ERROR("counter of the bind contains zero.\n");
            return -1;
        }

        sock_bind->counter --;
        if (sock_bind->counter == 0) {
			ERROR("Closing sock_bind->socket_fd.\n");
            rc |= close_sock_fd(sock_bind->socket_fd);
        }
    }

    return rc;
}


static int sock_recv(struct sock_t *sock, unsigned int size, void* buf)
{
    int rc;

    if (sock->sock_fd == -1) {
        ERROR("socket_fd is not allocated.\n");
        return -1;
    }

retry_after_signal:

    rc = recv(sock->sock_fd, buf, size, MSG_WAITALL);
    if (rc != size) {
        DEBUG("recv failed, sock_fd: %d, rc: %d, error: %s.\n", sock->sock_fd, rc, strerror(errno));

        if ((errno == EINTR) && (rc != 0)) {  // ???
            goto retry_after_signal;
        }
		
        if (rc) {
            return rc;
        } else {
        	return errno;
        }
    }

    DEBUG("socket_fd = 0x%x, Received buf=%p size=%d rc=%d.\n", sock->sock_fd, buf, size, rc);
    return 0;
}


static int sock_send(struct sock_t *sock, unsigned int size, void* buf)
{
    int rc;

    if (sock->sock_fd == -1) {
        ERROR("socket_fd is not allocated.\n");
        return EINVAL;
    }

retry_after_signal:
    rc = send(sock->sock_fd, buf, size, 0);
    if (rc != size) {
        if ((errno == EINTR) && (rc != 0)) // ???
            goto retry_after_signal;
        if (rc)
            return rc;
        else
            return errno;
    }

	DEBUG("socket_fd = 0x%x, Sent buf=%p size=%d rc=%d.\n", sock->sock_fd, buf, size, rc);
    return 0;
}


int sock_sync_data(struct sock_t *sock, unsigned int size, void *out_buf, void *in_buf)
{
    int rc;

    if (sock->is_daemon) {
        rc = sock_send(sock, size, out_buf);
        if (rc != 0)
            return rc;
        rc = sock_recv(sock, size, in_buf);
        if (rc != 0)
            return rc;
    } else {
        rc = sock_recv(sock, size, in_buf);
        if (rc != 0)
            return rc;
        rc = sock_send(sock, size, out_buf);
        if (rc != 0)
            return rc;
    }

    return 0;
}


int sock_sync_ready(struct sock_t *sock)
{
    char cm_buf = 'a';
    return sock_sync_data(sock, sizeof(cm_buf), &cm_buf, &cm_buf);
}


int sock_d2c(struct sock_t *sock, unsigned int size, void *buf)
{
	return ((sock->is_daemon) ? sock_send(sock, size, buf) : sock_recv(sock, size, buf));
}


int sock_c2d(struct sock_t *sock, unsigned int size, void* buf)
{
	return ((!sock->is_daemon) ? sock_send(sock, size, buf) : sock_recv(sock, size, buf));
}
