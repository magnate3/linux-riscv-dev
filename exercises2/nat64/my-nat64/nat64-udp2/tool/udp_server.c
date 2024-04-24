#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h>
#include <sys/time.h>

int
usage(const char *pgm, int exitval)
{
	fprintf(stderr, "Usage: %s [-{4|6}] [-q] [-p port] [-s pid2signal]\n",
		pgm);

	fprintf(stderr, "	-4		Use IPv4 protocol (default)\n"
			"	-6		Use IPv6 protocol\n"
			"	-p port		Port number to use\n"
			"	-q		Be quiet and don't "
						"report transfers\n"
			"	-s pid		PID to signal with USR1 "
						"after port is opened\n"
		);

	exit(exitval);
}


int
main(int argc, char **argv)
{
	int		opt;
	char		*port_arg = "50037";
	int		port;
	char		*pid_arg = NULL;
	pid_t		pid2signal;
	int		flag_4 = 0, flag_6 = 0, flag_q = 0;
	int		family = AF_INET;
	int		sfd;
	int		retval;
	size_t		sasz = 0;
	union {
		struct sockaddr_in  sin;
		struct sockaddr_in6 sin6;
	} sockaddr;

	while ((opt = getopt(argc, argv, "46p:qs:")) != -1) {
		switch(opt) {
		case '4':
			flag_4 = 1;
			family = AF_INET;
			break;
		case '6':
			flag_6 = 1;
			family = AF_INET6;
			break;
		case 'p':
			port_arg = optarg;
			break;
		case 'q':
			flag_q = 1;
			break;
		case 's':
			pid_arg = optarg;
			break;
		default:
			usage(argv[0], 1);
		}
	}

	if (flag_4 && flag_6) {
		fprintf(stderr, "Options -4 and -6 are mutually exclusive.\n");
		usage(argv[0], 1);
	}

	port = atoi(port_arg);
	if ((port & ~0xffff) || (port == 0)) {
		fprintf(stderr, "Port value of %d is out of range.\n", port);
		usage(argv[0], 1);
	}

	if (pid_arg) {
		pid2signal = atoi(pid_arg);
		if (kill(pid2signal, 0) == -1) {
			fprintf(stderr, "Cannot signal pid %d [%s (%d)].\n",
				pid2signal, strerror(errno), errno);
			usage(argv[0], 1);
		}
	}

	if (optind != argc) {
		fprintf(stderr, "No operands expected.\n");
		usage(argv[0], 1);
	}

	if ((sfd = socket(family, SOCK_DGRAM, 0)) == -1) {
		perror("socket failed");
		return 1;
	}

	switch (family) {
	case AF_INET:
		sockaddr.sin = (struct sockaddr_in){0};
		sockaddr.sin.sin_family = family;
		sockaddr.sin.sin_port = htons(port);
		sockaddr.sin.sin_addr.s_addr = htonl(INADDR_ANY);
		sasz = sizeof(sockaddr.sin);
		break;

	case AF_INET6:
		sockaddr.sin6 = (struct sockaddr_in6){0};
		sockaddr.sin6.sin6_family = family;
		sockaddr.sin6.sin6_port = htons(port);
		sockaddr.sin6.sin6_addr = in6addr_any;
		sasz = sizeof(sockaddr.sin6);
		break;
	}

	if (bind(sfd, (struct sockaddr *)&sockaddr, sasz) == -1) {
		perror("bind failed");
		return 1;
	}

	if (pid_arg && kill(pid2signal, SIGUSR1) == -1) {
		perror("kill failed");
		return 1;
	}

	while (1) {
		union {
			struct sockaddr_in	claddr_in;
			struct sockaddr_in6	claddr_in6;
		} claddr;
		socklen_t		clientlen;
		int			length;
		struct timeval		tv;
		int			rv;
		char			buffer[1024];

		clientlen = (family == AF_INET) ?
			sizeof(claddr.claddr_in) : sizeof(claddr.claddr_in6);
		length = recvfrom(sfd, buffer, sizeof(buffer) - 1, 0,
			(struct sockaddr *) &claddr, &clientlen);

		if (length < 0) {
			perror("recvfrom failed");
			retval = 1;
			break;
		}

		if (gettimeofday(&tv, NULL) == -1) {
			perror("gettimeofday failed");
			retval = 1;
			break;
		}

		buffer[length] = '\0';
		if (!flag_q) {
			printf("%ld.%06ld: Received %d bytes: '%s'\n",
				tv.tv_sec, tv.tv_usec, length, buffer);
			fflush(stdout);
		}

		clientlen = (family == AF_INET) ?
			sizeof(claddr.claddr_in) : sizeof(claddr.claddr_in6);
		if ((rv = sendto(sfd, buffer, length, 0,
				(struct sockaddr *)&claddr, clientlen)) == -1) {
			perror("sendto failed");
			retval = 1;
			break;
		}

		if (gettimeofday(&tv, NULL) == -1) {
			perror("gettimeofday failed");
			retval = 1;
			break;
		}

		if (!flag_q) {
			printf("%ld.%06ld: Sent %d/%d bytes: '%s'\n",
				tv.tv_sec, tv.tv_usec, rv, length, buffer);
			fflush(stdout);
		}
	}

	if (close(sfd) == -1) {
		perror("close failed");
		retval = 1;
	}

	return retval;
}
