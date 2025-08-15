#include <stdio.h>

#include "myheader.h"

#define DBG	0

static void printe(const char *error)
{
	printf("%s : %s\n", error, strerror(errno));
	exit(1);
}



static int socket_config(	char* if_name,
							int so_timestamp,
							int so_timestampns,
							int so_timestamping_flags)
{
	/*variables*/
	int sock;
	int enabled=0;
	int val;
	int len;
	
	struct ifreq device;
	struct ifreq hwtstamp;
	struct hwtstamp_config hwconfig, hwconfig_requested;
	
	
	/*create datagram socket*/
	sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (sock < 0)
		printe("socket");
	printf("[INFO] socket created successfully.\n");
	
	/*get ip addr -> linux socket config struct*/
	memset(&device, 0, sizeof(device));
	strncpy(device.ifr_name, if_name, sizeof(device.ifr_name));
	if (ioctl(sock, SIOCGIFADDR, &device) < 0)
		printe("ioctl - IP address");
	
	//configure driver options
	memset(&hwtstamp, 0, sizeof(hwtstamp));
	strncpy(hwtstamp.ifr_name, if_name, sizeof(hwtstamp.ifr_name));
	hwtstamp.ifr_data = (void *)&hwconfig;
	memset(&hwconfig, 0, sizeof(hwconfig));
	hwconfig.tx_type	= 	(so_timestamping_flags & SOF_TIMESTAMPING_TX_HARDWARE) ?
								HWTSTAMP_TX_ON : HWTSTAMP_TX_OFF;
	hwconfig.rx_filter 	= 	(so_timestamping_flags & SOF_TIMESTAMPING_RX_HARDWARE) ?
								HWTSTAMP_FILTER_PTP_V1_L4_SYNC : HWTSTAMP_FILTER_NONE;
	
	hwconfig_requested = hwconfig;
	
	if (ioctl(sock, SIOCSHWTSTAMP, &hwtstamp) < 0) 
	{
		if ((errno == EINVAL || errno == ENOTSUP) &&
		    hwconfig_requested.tx_type == HWTSTAMP_TX_OFF &&
		    hwconfig_requested.rx_filter == HWTSTAMP_FILTER_NONE)
			printf("SIOCSHWTSTAMP: disabling hardware time stamping not possible\n");
		else if(errno == ERANGE)
			printf("SIOCSHWTSTAMP not supported, cannot timestamp the packets.");
		else
			printe("SIOCSHWTSTAMP");
	}
	printf("SIOCSHWTSTAMP: tx_type %d requested, got %d; rx_filter %d requested, got %d\n",
	       		hwconfig_requested.tx_type, 
	       		hwconfig.tx_type,
	       		hwconfig_requested.rx_filter, 
	       		hwconfig.rx_filter);
	
	
	/* set socket options for time stamping */
	if (so_timestamp &&	setsockopt(	sock, 
									SOL_SOCKET, 
									SO_TIMESTAMP,
			   						&enabled, 
			   						sizeof(enabled)) < 0)
		printe("setsockopt SO_TIMESTAMP");
	else
	{
		printf("[INFO] so_timestamp = %d", 
					so_timestamp);
	}

	if (so_timestampns && setsockopt(	sock, 
										SOL_SOCKET, 
										SO_TIMESTAMPNS,
			   							&enabled, 
			   							sizeof(enabled)) < 0)
		printe("setsockopt SO_TIMESTAMPNS");
	else
	{
		printf("[INFO] so_timestampns = %d", 
					so_timestampns);
	}

	if (so_timestamping_flags && setsockopt(sock, 
											SOL_SOCKET, 
											SO_TIMESTAMPING,
			  	 							&so_timestamping_flags,
			   								sizeof(so_timestamping_flags)) < 0)
		printe("setsockopt SO_TIMESTAMPING");
	else
	{
		printf("[INFO] so_timestamping_flags = %d", 
					so_timestamping_flags);
	}

	/* request IP_PKTINFO for debugging purposes */
	if (setsockopt(	sock, 
					SOL_IP, 
					IP_PKTINFO,
		       		&enabled, 
		       		sizeof(enabled)) < 0)
		printe("setsockopt IP_PKTINFO");

	/* verify socket options */
	len = sizeof(val);
	if (getsockopt(	sock, 
					SOL_SOCKET, 
					SO_TIMESTAMP, 
					&val, 
					&len) < 0)
		printe("getsockopt SO_TIMESTAMP");
	else
		printf("SO_TIMESTAMP %d\n", val);

	if (getsockopt(	sock, 
					SOL_SOCKET, 
					SO_TIMESTAMPNS, 
					&val, 
					&len) < 0)
		printf("getsockopt SO_TIMESTAMPNS");
	else
		printf("SO_TIMESTAMPNS %d\n", val);

	if (getsockopt(	sock, 
					SOL_SOCKET, 
					SO_TIMESTAMPING, 
					&val, 
					&len) < 0)
		printe("getsockopt SO_TIMESTAMPING");
	else
	{
		printf("SO_TIMESTAMPING %d\n", val);
		if (val != so_timestamping_flags)
			printf("   not the expected value %d\n", so_timestamping_flags);
	}
	
	return sock;
}

int main()
{
	int sock, sock1, sock2;
	int exit=0;
	char userin;

	struct sockaddr_in tx_addr, recv_addr;
	socklen_t recv_struct_len = sizeof(recv_addr);
	
	char tx_data[TX_DATA_BUFF_LEN];
	
	/**********************CONFIG**********************/
	/*	
	*	SO_TIMESTAMP - generates SOFTWARE timestamp for each INCOMING packet 
	* 	Reports timestamp via recvmsg() in control message in usec resolution
	*	__kernel_sock_timeval for SO_TIMESTAMP_NEW
	*/
	int so_timestamp = 0;
	/*
	*	SO_TIMESTAMPNS - similar to SO_TIMESTAMP, generates timestamp in nsec
	*/
	int so_timestampns = 0;
	int siocgstamp = 1;
	int siocgstampns = 1;
	int so_timestamping_flags=0;
	
	/*
	*	Timestamp Generation - include below flags
	*		SOF_TIMESTAMPING_RX_HARDWARE
	*		SOF_TIMESTAMPING_RX_SOFTWARE
	*		SOF_TIMESTAMPING_TX_HARDWARE
	*		SOF_TIMESTAMPING_TX_SOFTWARE
	*/
	//so_timestamping_flags |= SOF_TIMESTAMPING_TX_HARDWARE;
	so_timestamping_flags |= SOF_TIMESTAMPING_TX_SOFTWARE;
	//so_timestamping_flags |= SOF_TIMESTAMPING_RX_HARDWARE;
	so_timestamping_flags |= SOF_TIMESTAMPING_RX_SOFTWARE;

	/*
	* 	Timestamping Reporting - timestamp is generated in control message (cmsg)
	*		SOF_TIMESTAMPING_SOFTWARE 		-- software timestamp
	*		SOF_TIMESTAMPING_SYS_HARDWARE 	-- deprecated and ignored
	*		SOF_TIMESTAMPING_RAW_HARDWARE 	-- hardware timestamp
	*/
	so_timestamping_flags |= SOF_TIMESTAMPING_SOFTWARE;
	//so_timestamping_flags |= SOF_TIMESTAMPING_RAW_HARDWARE;
	
	/*****************************************************/
	
	sock1 = socket_config(	INTERFACE,
							so_timestamp,
							so_timestampns,
							so_timestamping_flags);
	
	sock2 = socket_config(	INTERFACE,
							so_timestamp,
							so_timestampns,
							so_timestamping_flags);
	
	/*tx_addr not needed*/
//		/*transmitter binding*/
		tx_addr.sin_family = AF_INET;
		tx_addr.sin_port = htons(TX_PORT);
		//tx_addr.sin_addr.s_addr = htons(INADDR_ANY);
		tx_addr.sin_addr.s_addr = inet_addr(TX_ADDR);
		if(bind(sock1,
			 	(struct sockaddr *)&tx_addr,
			 	sizeof(struct sockaddr_in)) < 0)
			printe("bind");
		printf("[INFO] sock1 binding successful.\n");
	
		tx_addr.sin_port = htons(TX_PORT2);
		if(bind(sock2,
			 	(struct sockaddr *)&tx_addr,
			 	sizeof(struct sockaddr_in)) < 0)
			printe("bind");
		printf("[INFO] sock2 binding successful.\n");
	
	
	
	/*receiver*/
	recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(RX_PORT);
	recv_addr.sin_addr.s_addr = inet_addr(RX_ADDR); // receivers address
	
	float tbreak;
	int pcount;
	int i=1;
	int whichsock;
	while(exit != 1)
	{
		printf("send packets or exit ? (y/e) : ");
		scanf("%c", &userin);
		
		if(userin=='y')
		{
			//printf("time interval bet packets (msec) : ");
			//scanf("%f", &tbreak);
			//tbreak = tbreak/1000; //convert to msec
			
			printf("no of packets to be sent"
					"  (0=continuous): ");
			scanf("%d", &pcount);
			
			// printf("which socket"
			// 		"  1. sock1(6666)"
			// 		"  2. sock2(7777)\n"
			// 		"    Value (1/2) : ");
			// scanf("%d", &whichsock);

			memset(tx_data, '\0', sizeof(tx_data));
			memset(tx_data, '1', TX_PAYLOAD_SIZE);

			//if(1<=whichsock<=2)
			{
				// if(whichsock==1)
				// 	sock=sock1;
				// else if(whichsock==2)
				// 	sock=sock2;
				
				i=1;
				while(i<=pcount)
				{
					if (sendto(	sock1, 
								tx_data, 
								strlen(tx_data), 
								0,
								(struct sockaddr*)&recv_addr, 
								recv_struct_len) < 0)
					{
						printe("send_packet->sendto");
						return -1;
					}
					printf("[INFO] sent (port:6666) %d\t", i);
					i++;

					if (sendto(	sock2, 
								tx_data, 
								strlen(tx_data), 
								0,
								(struct sockaddr*)&recv_addr, 
								recv_struct_len) < 0)
					{
						printe("send_packet->sendto");
						return -1;
					}
					printf("[INFO] sent (port:7777) %d\n", i);
					i++;
				}
			}
		}
		else if(userin=='e')
		{
			exit=1;
		}
		else
			printf("enter again.\n");
	}
	
	printf("exiting...\n");
	sleep(1);
	close(sock1);
	close(sock2);		
	return 0;
}







