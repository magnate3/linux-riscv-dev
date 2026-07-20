#ifndef CONN_H
#define CONN_H

#define _GNU_SOURCE

#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>

struct conn_list;

struct conn_node
{
    int id; /* connection ID */
    int sockfd; /* socket */
    pthread_t thread;   /* thread */
    bool busy;  /* whether the connection is receiving data */
    bool connected; /* whether the connection is established */
    struct conn_node *next; /* pointer to next node */
    struct conn_list *list; /* pointer to parent list */
    pthread_mutex_t lock;
};

struct conn_list
{
    int index;  /* server index */
    char ip[20];    /* server IP address */
    unsigned short port;   /* server port number */
    struct conn_node *head; /* pointer to head node */
    struct conn_node *tail; /* pointer to tail node */
    unsigned int len;   /* total number of nodes */
    unsigned int available_len; /* total number of available nodes */
    unsigned int flow_finished; /* total number of flows finished */
    pthread_mutex_t lock;
};


/* initialize functions */
int init_socket(struct conn_list *list);
bool init_conn_node(struct conn_node *node, int id, struct conn_list *list);
bool init_conn_list(struct conn_list *list, int index, char *ip, unsigned short port);

bool reinit_conn_node(struct conn_node *node);

/* insert several nodes to the tail of the linked list */
bool insert_conn_list(struct conn_list *list, int num);

/* search the first available connection (busy==false) in the list. */
struct conn_node *search_conn_list(struct conn_list *list);

/* search N available connections in the list */
struct conn_node **search_n_conn_list(struct conn_list *list, unsigned int num);

/* wait for all threads in the linked list to finish */
void wait_conn_list(struct conn_list *list);

/* clear all the nodes in the linked list */
void clear_conn_list(struct conn_list *list);

/* print information of the linked list */
void print_conn_list(struct conn_list *list);

/* remove a node from the list - Musa */
bool remove_from_conn_list(struct conn_list *list, struct conn_node *node);

/* musa */
void print_conn_node(struct conn_node *node);

/* musa */
void remove_conn_node_at_head(struct conn_list* list);
// musa
struct conn_node *get_tail_conn_node(struct conn_list *list);

#endif
