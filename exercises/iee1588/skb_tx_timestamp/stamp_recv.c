#include <arpa/inet.h>
#include <linux/net_tstamp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <linux/errqueue.h>
#include <sys/ioctl.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <unistd.h>
#include <time.h>
#include <poll.h>
#include <linux/if.h>

void die(char* s)
{
	perror(s);
	exit(1);
}

int main(int argc, char* argv[]) {
	int sock;
	int server_port = 1234;
	struct in_addr sourceIP;

	if(argc != 2) {
		fprintf(stderr,"Error. You should specify the interface name.\n");
		exit(1);
	}

        if ((sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
            die("UDP socket()");
        }

	struct ifreq ifindexreq;
	struct sockaddr_in si_server;

	// Get source IP address
	strncpy(ifindexreq.ifr_name, argv[1], IFNAMSIZ);
	ifindexreq.ifr_addr.sa_family = AF_INET;
	if(ioctl(sock,SIOCGIFADDR,&ifindexreq)!=-1) {
		sourceIP=((struct sockaddr_in*)&ifindexreq.ifr_addr)->sin_addr;
	} else {
		die("SIOCGIFADDR ioctl()");
	}

	bzero(&si_server,sizeof(si_server));
	si_server.sin_family = AF_INET;
	si_server.sin_port = htons(server_port);
	si_server.sin_addr.s_addr = sourceIP.s_addr;
	fprintf(stdout,"source IP: %s\n",inet_ntoa(sourceIP));

	// bind() to interface
	if(bind(sock, (struct sockaddr *)&si_server, sizeof(si_server))<0) {
		die("bind()");
	}
	
	fprintf(stdout,"Test started.\n");

	int flags;
	struct ifreq hwtstamp;
	struct hwtstamp_config hwconfig;

	// Set hardware timestamping
	memset(&hwtstamp, 0, sizeof(hwtstamp));
	memset(&hwconfig, 0, sizeof(hwconfig));

	// Set ifr_name and ifr_data
	strncpy(hwtstamp.ifr_name, argv[1], sizeof(hwtstamp.ifr_name));
	hwtstamp.ifr_data = (void *)&hwconfig;

	hwconfig.tx_type = HWTSTAMP_TX_ON;
	hwconfig.rx_filter = HWTSTAMP_FILTER_ALL;

	// Issue request to the driver
	if (ioctl(sock,SIOCSHWTSTAMP,&hwtstamp)<0) {
		die("ioctl()");
	}

	flags=SOF_TIMESTAMPING_RX_HARDWARE | SOF_TIMESTAMPING_TX_HARDWARE | SOF_TIMESTAMPING_RAW_HARDWARE;

	if(setsockopt(sock,SOL_SOCKET, SO_TIMESTAMPING, &flags, sizeof(flags)) < 0) {
		die("setsockopt()");
	}

	// Recv 10 packets
	const int n_packets = 10;
	for (int i = 0; i < n_packets; ++i) {
		// Obtain the recv packet timestamp.
		char data[256];
		struct msghdr msg;
		struct iovec entry;
		char ctrlBuf[CMSG_SPACE(sizeof(struct scm_timestamping))];

		memset(&msg, 0, sizeof(msg));
		msg.msg_iov = &entry;
		msg.msg_iovlen = 1;
		entry.iov_base = data;
		entry.iov_len = sizeof(data);
		msg.msg_name = NULL;
		msg.msg_namelen = 0;
		msg.msg_control = &ctrlBuf;
		msg.msg_controllen = sizeof(ctrlBuf);

		if (recvmsg(sock, &msg, 0) < 0) {
			die("recvmsg()");
		}
		printf("Recv pakage: %s\n", data);
		// Extract and print ancillary data (SW or HW tx timestamps)
		struct cmsghdr *cmsg = NULL;
		struct scm_timestamping hw_ts;
		struct timespec ts;

		for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
			if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SO_TIMESTAMPING) {
				hw_ts = *((struct scm_timestamping *)CMSG_DATA(cmsg));
				fprintf(stdout,"HW: %lu s, %lu ns\n", hw_ts.ts[2].tv_sec, hw_ts.ts[2].tv_nsec);
				fprintf(stdout,"ts[1]: %lu s, %lu ns\n", hw_ts.ts[1].tv_sec, hw_ts.ts[1].tv_nsec);
				fprintf(stdout,"SW: %lu s, %lu ns\n", hw_ts.ts[0].tv_sec, hw_ts.ts[0].tv_nsec);
				memcpy(&ts, &hw_ts.ts[2], sizeof(struct timespec));
			}
		}
		fprintf(stdout,"Hardwave recv timestamp: %lu s, %lu ns\n\n", ts.tv_sec, ts.tv_nsec);

		// Wait 1s before sending next packet
		sleep(1);
	}

	close(sock);
	return 0;
}
