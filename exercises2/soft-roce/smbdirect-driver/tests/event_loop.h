#ifndef __EVENT_POLL_H__
#define __EVENT_POLL_H__

typedef uint32_t (*event_dispatch_fn_t)(int32_t fd, uint32_t events, void *ctx);

int poll_add_event_source(int32_t fd, 
			  uint32_t events, 
			  void *context, 
			  event_dispatch_fn_t event_dispatch_fn);

int event_loop_and_dispatch(void);
#endif
