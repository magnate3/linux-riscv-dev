#include <stdlib.h>
#include <stdio.h>
#include "ibmsg.h"

#define APPLICATION_NAME "ibmsg-rdma-send"
#define MSGSIZE (4096)

int
main(int argc, char** argv)
{
	char* ip;
	unsigned short port = 12345;

	if(argc < 2 || argc > 3)
	{
		fprintf(stderr, APPLICATION_NAME": error: illegal number of arguments\n");
		fprintf(stderr, APPLICATION_NAME": usage: %s IP [PORT]\n", APPLICATION_NAME);
		exit(EXIT_FAILURE);
	}
	ip = argv[1];
	if(argc == 3)
		port = atoi(argv[2]);

	/* Setup */
	ibmsg_event_loop event_loop;
	if(ibmsg_init_event_loop(&event_loop))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not initialize event loop\n");
		exit(EXIT_FAILURE);
	}


	/* Connect */
	ibmsg_socket connection;
	if(ibmsg_connect(&event_loop, &connection, ip, port))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not connect to remote host\n");
		exit(EXIT_FAILURE);
	}
	while(connection.status != IBMSG_CONNECTED)
	{
		ibmsg_dispatch_event_loop(&event_loop);
		if (connection.status == IBMSG_ERROR)
		{
			fprintf(stderr, APPLICATION_NAME": error: could not connect to remote host\n");
			exit(EXIT_FAILURE);
		}
	}


	/* Data transfers */
	ibmsg_buffer msg;
	if(ibmsg_alloc_msg(&msg, &connection, MSGSIZE))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not allocate message memory\n");
		exit(EXIT_FAILURE);
	}

	if(ibmsg_post_send(&connection, &msg))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not send message\n");
		exit(EXIT_FAILURE);
	}

	while(msg.status != IBMSG_SENT)
		if(ibmsg_dispatch_event_loop(&event_loop))
		{
			fprintf(stderr, APPLICATION_NAME": error: something went wrong while working in the event loop\n");
			exit(EXIT_FAILURE);
		}

	if(ibmsg_free_msg(&msg))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not free memory\n");
	}


	/* Disconnect */
	if(ibmsg_disconnect(&connection))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not disconnect\n");
	}
	while(connection.status != IBMSG_UNCONNECTED)
	{
		if(ibmsg_dispatch_event_loop(&event_loop))
		{
			fprintf(stderr, APPLICATION_NAME": error: something went wrong while working in the event loop\n");
		}
		if (connection.status == IBMSG_ERROR)
		{
			fprintf(stderr, APPLICATION_NAME": error: could not disconnect\n");
		}
	}


	/* Teardown */
	if(ibmsg_destroy_event_loop(&event_loop))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not destroy event loop\n");
	}

    return EXIT_SUCCESS;
}
