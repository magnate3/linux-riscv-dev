/*
 * A simple IB example ... 
 *
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>

#include "event_loop.h"

/*
 * A simple set of routines for event handling ...
 *
 * Users can register fd's they are interested in getting events on as well as
 * functions to call when those events occur.
 *
 * We are primarily interested in when they become readable ...
 */

/*
 * Add and event source along with a callback function. This requires an fd.
 * It is targeted at Infiniband CM events and ibv completion channels, since
 * they are tied to fds.
 */

#define LIST_HEAD(list_name) \
	struct list_head_struct list_name = { &(list_name), &(list_name) };

/*
 * List manip stuff based on Linux Kernel list.h, but seems standard.
 * Circular doubly linked list.
 */
struct list_head_struct {
	struct list_head_struct *next, *prev;
};

static void my_list_add(struct list_head_struct *new,
			struct list_head_struct *prev,
			struct list_head_struct *next)
{
	next->prev = new;
	new->next = next;
	new->prev = prev;
	prev->next = new;
}

#define LIST_ADD_TAIL(list_head, new) my_list_add(new, (list_head)->prev, list_head)

struct event_entry_struct {
	struct list_head_struct list_ent;
	int32_t fd;
	uint32_t events;
	void *ctx;
	event_dispatch_fn_t event_dispatch_fn;
};

/* 
 * Add an event source ... We link all of them together and call epoll_ctl
 * to add the new event source to the list we are listening to.
 */
static int epollfd = -1;
LIST_HEAD(epoll_event_list);

int poll_add_event_source(int32_t fd, 
			  uint32_t events, 
			  void *context, 
			  event_dispatch_fn_t event_dispatch_fn)
{
	int res = -1;
	struct event_entry_struct *event;
	struct epoll_event epoll_event;

	/*
	 * Have we created the epollfd yet?
	 */
	if (epollfd < 0) {
		epollfd = epoll_create(20); /* Num is ignored anyway */
		if (epollfd < 0) {
			fprintf(stderr, "Could not create pollfd: %s\n", 
				strerror(errno));
			return epollfd;
		}
	}

	event = (void *)malloc(sizeof(struct event_entry_struct));
	if (!event) {
		fprintf(stderr, "Unable to allocate space for event: %s\n",
			strerror(errno));
		return -1;
	}

	event->fd = fd;
	event->events = events;
	event->ctx = context;
	event->event_dispatch_fn = event_dispatch_fn;

	LIST_ADD_TAIL(&epoll_event_list, &event->list_ent);

	epoll_event.events = events;
	epoll_event.data.ptr = event;

	res = epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &epoll_event);
	if (res < 0) {
		fprintf(stderr, "Unable to add epoll event: %s\n",
			strerror(errno));
		return -1;
	} 

	return res;
}

/*
 * Poll for events. Call the appropriate dispatch function when we get them.
 * Might not play well in the presence of signals.
 */
#define MAX_EVENTS 10
static int exit_loop = 0;   /* Will be set to exit the loop */
int event_loop_and_dispatch(void)
{
	struct epoll_event event, events[MAX_EVENTS];
	int res = -1, i = 0;
	uint32_t hnd_res = 0;
	struct event_entry_struct *ev;
	event_dispatch_fn_t fn = 0;

	while (!exit_loop) {
		/* Wait forever for events */
		res = epoll_wait(epollfd, events, MAX_EVENTS, -1); 
		if (res < 0) {
			fprintf(stderr, "Unable to poll for events: %s\n",
				strerror(errno));
			return res;
		}

		/*
		 * Process the events
		 */
		for (i = 0; i < res; i++) {
			ev = (struct event_entry_struct *)events[i].data.ptr;
			fn = ev->event_dispatch_fn;
			hnd_res = fn(ev->fd, events[i].events, ev->ctx);
			if (hnd_res) {
				exit_loop = 1;
				break;
			}
		}
	}

	/*
	 * We exited the loop ... too bad ...
	 */

	return 0;
}

