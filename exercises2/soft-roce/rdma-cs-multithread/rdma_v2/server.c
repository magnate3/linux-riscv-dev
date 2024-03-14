/**
 * @file server.c
 * @author Austin Pohlmann
 * @brief A RDMA server
 * This server uses pthreads for each client conection, as well as a listener thread and the main thread for server administration.
 */
 #include "rdma_cs.h"

/**
 *@brief Determines if a client's memory region is open or closed to other clients
 */
enum client_status{
	OPEN,	/**< Memory region is open for other clients to use */
	CLOSED	/**< Memory region is closed to other clients */
};

/**
 *@brief Linked list node containing information on all running threads
 */
struct pnode {
	pthread_t id;			/**< The thread id */
	unsigned short type;	/**< The purpose of the thread */
	struct pnode *next;		/**< A pointer to the next node in the list */
} *tlist_head;

/**
 *@brief Linked list node containing information on all connected clients
 */
struct cnode {
	struct rdma_cm_id *id;		/**< The communication manager id */
	pthread_t tid;				/**< The thread id */
	unsigned long cid;			/**< The numerical id */
	uint32_t rkey;				/**< The rkey of the server-side memory region */
	uint64_t remote_addr;		/**< The address of the server-side memory region */
	size_t length;				/**< The length of the server-side memory region */
	enum client_status status;	/**< The status of the server-side memory region */
	struct cnode *next;			/**< A pointer to the next node in the list */
} *clist_head;

/**
 * @brief Semaphore for synchronizing the manipulation of the client list
 */
sem_t clist_sem;
/**
 * @brief Semaphore for synchronizing the manipulation of the thread list
 */
sem_t tlist_sem;
/**
 * @brief The port number that the server will be bound to
 */
short port;
/**
 * @brief The current number of connected clients
 */
unsigned long clients = 0, idnum = 0;
/**
 * @brief The file pointer of the log file
 */
FILE *log_p;

void binding_of_isaac(struct rdma_cm_id *, short);
void *hey_listen(void *);
void *secret_agent(void *);
void add_thread(struct pnode);
void add_client(struct cnode);
void remove_thread(pthread_t);
void remove_client(pthread_t);
void set_status(pthread_t, enum client_status);
void remote_remove(pthread_t);
void remote_add(pthread_t);

int main(int argc, char **argv){
	// Create log directory
	struct stat st = {0};
	if (stat("server_logs", &st) == -1) {
	    mkdir("server_logs", 0700);
	}
	// Create log file
	char filename[100];
	time_t rawtime;
	struct tm *timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strcat(filename, asctime(timeinfo));
	sprintf(filename,"%s%d-%d-%d-%d:%d.log", SERVER_LOG_PATH, timeinfo->tm_year + 1900,timeinfo->tm_mon + 1,timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min);
	int i;
	log_p = fopen(filename , "w");
	fprintf(log_p, "Server started on: %s\n", asctime(timeinfo));
	// Get port from arguments
	if(argc == 2)
		port = atoi(argv[1]);
	else
		port = 0;
	// Create event channel
	struct rdma_event_channel *event_channel = rdma_create_event_channel();
	if(event_channel == NULL)
		stop_it("rdma_create_event_channel()", errno, log_p);
	// Create the ID
	struct rdma_cm_id *cm_id;
	if(rdma_create_id(event_channel, &cm_id, "qwerty", RDMA_PS_TCP))
		stop_it("rdma_create_id()", errno, log_p);
	// Bind to the port
	binding_of_isaac(cm_id, port);
	// Store the pthread's id and purpose
	tlist_head = malloc(sizeof(struct pnode));
	memset(tlist_head, 0, sizeof(struct pnode));
	tlist_head->type =0;
	// Initialize the client and thread list access semaphores
	sem_init(&clist_sem, 0, 1);
	sem_init(&tlist_sem, 0, 1);
	// Spawn listener thread
	if(pthread_create(&tlist_head->id, NULL, hey_listen, cm_id))
		stop_it("pthread_create()", errno, log_p);
	int opcode;
	int num;
	struct cnode *client_list;
	struct pnode *threads;
	// Handle server side operations
	while(1){
		// Print the menu
		printf("---------------\n"
			"| RDMA server |\n"
			"------------------------------------------\n"
			"1) Shut down                             |\n"
			"2) View connected clients                |\n"
			"3) Disconnect a client                   |\n"
			"4) Shut down when all clients disconnect |\n"
			"> ");
		scanf("%d", &opcode);
		if(opcode==1){
			// Disconnect all clients and shut down
			printf("Shutting down server...\n");
			client_list = clist_head;
			sem_wait(&tlist_sem);
			pthread_cancel(tlist_head->id);
			while(client_list != NULL){
				rdma_send_op(client_list->id, DISCONNECT, log_p);
				get_completion(client_list->id, SEND, 1, log_p);
				printf("Client %lu has been successfully disconnected.\n", client_list->cid);
				sem_wait(&clist_sem);
				pthread_cancel(client_list->tid);
				client_list = client_list->next;
				sem_post(&clist_sem);
			}
			break;
		} else if (opcode==2){
			// Print a list of the connected clients
			if(clients > 0){
				sem_wait(&clist_sem);
				client_list = clist_head;
				for(i=0;i<clients; i++){
					printf("---Client id: %lu\n"
						"Client MR length: %llu bytes\n"
						"Client MR status: %s\n",
						client_list->cid,
						(unsigned long long)client_list->length,
						client_list->status == OPEN ? "open" : "closed");
					client_list = client_list->next;
				}
				sem_post(&clist_sem);
			} else {
				printf("There are curently no connected clients :(\n");
			}
		} else if (opcode == 0){
			// Print a list of all open threads
			struct pnode * threads;
			sem_wait(&tlist_sem);
			for(threads = tlist_head; threads != NULL; threads = threads->next){
				printf("---Thread id: %llx\nType: %u\n",
					(unsigned long long)threads->id, threads->type);
			}
			sem_post(&tlist_sem);
		} else if (opcode == 3) {
			// Disconnect a single connected client
			sem_wait(&clist_sem);
			client_list = clist_head;
			printf("Enter client ID: ");
			scanf("%d", &num);
			while(1){
				if (client_list == NULL){
					printf("Client not found.\n");
					break;
				} else if (client_list->cid == num){
					rdma_send_op(client_list->id, DISCONNECT, log_p);
					get_completion(client_list->id, SEND, 1, log_p);
					printf("Client has been successfully disconnected.\n");
					break;
				} else {
					client_list=client_list->next;
				}
			}
			sem_post(&clist_sem);
		} else if (opcode == 4){
			// Wait for all clients to disconnect before shutting down
			printf("Waiting for clients to disconnect...\n");
			sem_wait(&tlist_sem);
			threads = tlist_head;
			pthread_cancel(threads->id);
			threads = threads->next;
			sem_post(&tlist_sem);
			while(threads != NULL){
				pthread_join(threads->id, NULL);
				threads = threads->next;
			}
			break;
		}
	}
	fclose(log_p);
	return 0;
}

/**
 * @brief Bind to a specified port.
 *
 * If @p port is 0, a random free port will be chosen
 * @return @c NULL
 * @param cm_id The server's communication manager id
 * @param port the port to bind to
 */
void binding_of_isaac(struct rdma_cm_id *cm_id, short port){
	struct sockaddr_in sin;
	memset(&sin, 0, sizeof(struct sockaddr_in));
	sin.sin_family = AF_INET;
	sin.sin_port = htons(port);
	if(rdma_bind_addr(cm_id, (struct sockaddr *)&sin))
		stop_it("rdma_bind_addr()", errno, log_p);
	fprintf(log_p, "RDMA device bound to port %u.\n", ntohs(rdma_get_src_port(cm_id)));
}

/**
 * @brief The funtion for the listener thread.
 *
 * Listens for incoming connection requests and passes the relevant information to the creation of a new secret_agent() thread.
 * @return @c NULL
 * @param cmid the @c struct @c rdma_cm_id of the server cast to be a @c void @c *
 */
void *hey_listen(void *cmid){
	// Standard initializing
	struct rdma_cm_id *cm_id = cmid;
	struct rdma_event_channel *ec = cm_id->channel;
	struct pnode tlist;
	struct cnode clist;
	while(1){
		// Listen for connection requests
		fprintf(log_p, "Listening for connection requests...\n");
		if(rdma_listen(cm_id, 1))
			stop_it("rdma_listen()", errno, log_p);
		// Make an ID specific to the client that connected
		clist.id = cm_event(ec, RDMA_CM_EVENT_CONNECT_REQUEST, log_p);
		clist.length = SERVER_MR_SIZE;
		clist.status = CLOSED;
		idnum++;
		clist.cid = idnum;
		cm_event(ec, RDMA_CM_EVENT_ESTABLISHED, log_p);
		// Spawn an agent thread for the new conenction
		sem_wait(&tlist_sem);
		tlist.type = 1;
		if(pthread_create(&tlist.id, NULL, secret_agent, clist.id)){
			stop_it("pthread_create()", errno, log_p);
		}
		sem_post(&tlist_sem);
		add_thread(tlist);
		clist.tid = tlist.id;
		add_client(clist);
		// Remake the cm_id
		if(rdma_destroy_id(cm_id))
			stop_it("rdma_destroy_id()", errno, log_p);
		ec = rdma_create_event_channel();
		if(rdma_create_id(ec, &cm_id, "qwerty", RDMA_PS_TCP))
			stop_it("rdma_create_id()", errno, log_p);
		// Bind to the port
		binding_of_isaac(cm_id, port);
		// Rinse and repeat
	}
	return 0;
}

/**
 * @brief The function for the agent threads.
 *
 * Handles a single client connection
 * @return @c NULL
 * @param id the @c struct @c rdma_cm_id of the connection to the client cast to be a @c void @c *
 */
void *secret_agent(void *id){
	struct rdma_cm_id *cm_id = id;
	struct ibv_mr *mr = ibv_reg_mr(cm_id->qp->pd, malloc(SERVER_MR_SIZE),SERVER_MR_SIZE,
	 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
	if(mr == NULL)
		stop_it("ibv_reg_mr()", errno, log_p);
	sem_wait(&clist_sem);
	struct cnode * node = clist_head;
	while(node != NULL){
		if(node->tid == pthread_self()){
			node->rkey = mr->rkey;
			node->remote_addr = (uint64_t)mr->addr;
			break;
		}
		node = node->next;
	}
	sem_post(&clist_sem);
	// Exchange addresses and rkeys with the client
	uint32_t rkey;
	uint64_t remote_addr;
	swap_info(cm_id, mr, &rkey, &remote_addr, NULL, log_p);
	// The real good
	uint32_t opcode;
	while(1){
		rdma_recv(cm_id, mr, log_p);
		opcode = get_completion(cm_id, RECV, 1, log_p);
		if(opcode == DISCONNECT){
			fprintf(log_p, "Client issued a disconnect.\n");
			rdma_send_op(cm_id, 0, log_p);
			get_completion(cm_id, SEND, 1, log_p);
			break;
		} else if (opcode == OPEN_MR){
			remote_add(pthread_self());
			set_status(pthread_self(), OPEN);
		} else if (opcode == CLOSE_MR){
			remote_remove(pthread_self());
			set_status(pthread_self(), CLOSED);
		} 
	}
	// Disconnect and remove client from lists
	remove_thread(pthread_self());
	remote_remove(pthread_self());
	remove_client(pthread_self());
	obliterate(NULL, cm_id, mr, cm_id->channel, log_p);
	return 0;
}
/**
 * @brief Add a node to the thread list.
 *
 * @return @c NULL
 * @param node the node to add to the list
 */
void add_thread(struct pnode node){
	struct pnode *current;
	current = tlist_head;
	sem_wait(&tlist_sem);
	while(current->next != NULL)
		current = current->next;
	current->next = malloc(sizeof(struct pnode));
	current = current->next;
	memset(current, 0, sizeof(*current));
	current->id = node.id;
	current->type = node.type;
	sem_post(&tlist_sem);
}
/**
 * @brief Add a node to the client list.
 *
 * @return @c NULL
 * @param node the node to add to the list
 */
void add_client(struct cnode node){
	struct cnode *current;
	sem_wait(&clist_sem);
	clients++;
	if(clist_head == NULL){
		clist_head = malloc(sizeof(struct cnode));
		current = clist_head;
	} else {
		current = clist_head;
		while(current->next != NULL)
			current = current->next;
		current->next = malloc(sizeof(struct cnode));
		current = current->next;
	}
	memset(current, 0, sizeof(*current));
	current->id = node.id;
	current->tid = node.tid;
	current->cid = node.cid;
	current->rkey = node.rkey;
	current->remote_addr = node.remote_addr;
	current->status = node.status;
	current->length = node.length;
	current->next = NULL;
	sem_post(&clist_sem);
}
/**
 * @brief Remove a node from the thread list.
 *
 * @return @c NULL
 * @param id the id of the node to remove from the list
 */
void remove_thread(pthread_t id){
	struct pnode *un, *deux;
	sem_wait(&tlist_sem);
	un = tlist_head;
	deux = un->next;
	while(deux != NULL){
		if(deux->id == id){
			un->next = deux->next;
			free(deux);
			break;
		} else {
			un = deux;
			deux = deux->next;
		}
		
	}
	sem_post(&tlist_sem);
}
/**
 * @brief Remove a node from the client list.
 *
 * @return @c NULL
 * @param id the thread id of the node to remove from the list
 */
void remove_client(pthread_t id){
	struct cnode *ichi;
	struct cnode *ni;
	sem_wait(&clist_sem);
	ichi = clist_head;
	if(clist_head->tid == id){
		if(clist_head->next == NULL){
			free(clist_head);
			clist_head = NULL;
		} else {
			clist_head = clist_head->next;
			free(ichi);
		}
	} else {
		ni = ichi->next;
		while(ni != NULL){
			if(ni->tid == id){
				ichi->next = ni->next;
				free(ni);
				break;
			} else {
				ichi = ni;
				ni = ni->next;
			}
		}
	}
	clients--;
	sem_post(&clist_sem);
}
/**
 * @brief Change the status of a client's memory region
 *
 * @return @c NULL
 * @param id the thread id of the client
 * @param status the new status
 */
void set_status(pthread_t id, enum client_status status){
	struct cnode *node;
	sem_wait(&clist_sem);
	node = clist_head;
	while(node != NULL){
		if(node->tid == id){
			node->status = status;
			sem_post(&clist_sem);
			return;
		}
		node = node->next;
	}
	sem_post(&clist_sem);
}

/**
 * @brief Inform all clients when another client's memory region closes(but only if it was previously open).
 *
 * @return @c NULL
 * @param id the thread ID of the client that closed their memory region
 */
void remote_remove(pthread_t id){
	struct cnode *node = clist_head;
	struct cnode *client;
	sem_wait(&clist_sem);
	while(node != NULL){
		if(node->tid == id){
			client = node;
			break;
		}
		node = node->next;
	}
	if (client->status == CLOSED){
		sem_post(&clist_sem);
		return;
	}
	node = clist_head;
	while(node != NULL){
		if(node->tid == id){
			node = node->next;
			continue;
		} else {
			rdma_send_op(node->id, REMOVE_CLIENT, log_p);
			get_completion(node->id, SEND, 1, log_p);
			rdma_send_op(node->id, client->cid, log_p);
			get_completion(node->id, SEND, 1, log_p);
		}
		node = node->next;
	}
	sem_post(&clist_sem);
}
/**
 * @brief Inform all clients when another client's memory region opens.
 *
 * @return @c NULL
 * @param id the thread ID of the client that opened their memory region
 */
void remote_add(pthread_t id){
	struct cnode *node = clist_head;
	struct cnode *client;
	struct client client_data;
	sem_wait(&clist_sem);
	while(node != NULL){
		if(node->tid == id){
			client = node;
			break;
		}
		node = node->next;
	}
	if (client->status == OPEN){
		sem_post(&clist_sem);
		return;
	}
	node = clist_head;
	while(node != NULL){
		if(node->tid == id){
			node = node->next;
			continue;
		} else {
			client_data.rkey = client->rkey;
			client_data.remote_addr = client->remote_addr;
			client_data.cid = client->cid;
			client_data.length = client->length;
			rdma_send_op(node->id, ADD_CLIENT, log_p);
			get_completion(node->id, SEND, 1, log_p);
			rdma_post_send(node->id," ", &client_data, sizeof(client_data), 0, IBV_SEND_INLINE | IBV_SEND_SIGNALED);
			get_completion(node->id, SEND, 1, log_p);
		}
		node = node->next;
	}
	sem_post(&clist_sem);
}