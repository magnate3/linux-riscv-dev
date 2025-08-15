#include <stdlib.h>
#include <stdio.h>
#include "ibmsg.h"

#define APPLICATION_NAME "ibmsg-rdma-recv"
#define MAX_CONNECTIONS (64)

ibmsg_socket* connection;


void
connection_established(ibmsg_socket* connection)
{
	printf("connection established\n");
}

void
connection_request(ibmsg_connection_request* request)
{
	connection = malloc(sizeof(ibmsg_socket));
	ibmsg_accept(request, connection);
}

void
message_received(ibmsg_socket* connection, ibmsg_buffer* msg)
{
	printf("message received\n");
	ibmsg_free_msg(msg);
}


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
	event_loop.connection_established = connection_established;
	event_loop.connection_request = connection_request;
	event_loop.message_received = message_received;


	/* Listen */
	ibmsg_socket socket;
	if(ibmsg_listen(&event_loop, &socket, ip, port, MAX_CONNECTIONS))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not listen\n");
		exit(EXIT_FAILURE);
	}


	/* Event loop */
	while(1)
	{
		if(ibmsg_dispatch_event_loop(&event_loop))
		{
			fprintf(stderr, APPLICATION_NAME": error: something went wrong while working in the event loop\n");
			break;
		}
	}


	/* Teardown */
	free(connection);
	if(ibmsg_disconnect(&socket))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not disconnect socket\n");
	}
	if(ibmsg_destroy_event_loop(&event_loop))
	{
		fprintf(stderr, APPLICATION_NAME": error: could not destroy event loop\n");
	}

    return EXIT_SUCCESS;
}
