#include "spdk/nvme.h"
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/thread.h"
#include "spdk_internal/event.h"



#include "spdk/log.h"
#include "spdk/string.h"

#include "spdk/sock.h"
#include "spdk/net.h"

#define ACCEPT_TIMEOUT_US 1000
#define CLOSE_TIMEOUT_US 1000000
#define BUFFER_SIZE 1024
#define ADDR_STR_LEN INET6_ADDRSTRLEN
static bool g_is_running;

static char *g_host;
static char *g_sock_impl_name;
static int g_port;
static bool g_is_server;
static bool g_verbose;
/*
 * We'll use this struct to gather housekeeping hello_context to pass between
 * our events and callbacks.
 */
struct hello_context_t {
	bool is_server;
	char *host;
	char *sock_impl_name;
	int port;

	bool verbose;
	int bytes_in;
	int bytes_out;

	struct spdk_sock *sock;

	struct spdk_sock_group *group;
	struct spdk_poller *poller_in;
	struct spdk_poller *poller_out;
	struct spdk_poller *time_out;

	int rc;
};
struct hello_context_t * g_group_ctx = NULL;
struct spdk_sock_group * nvmf_tcp_get_optimal_poll_group(void);
static int
hello_sock_close_timeout_poll(void *arg)
{
        //uint32_t total_core = spdk_env_get_core_count();
	uint32_t i;
	struct hello_context_t *ctx = arg;
	struct hello_context_t *ctx2;
	SPDK_NOTICELOG("Connection closed\n");

	spdk_poller_unregister(&ctx->time_out);
	spdk_poller_unregister(&ctx->poller_in);
	spdk_sock_close(&ctx->sock);
	spdk_sock_group_close(&ctx->group);

        SPDK_ENV_FOREACH_CORE(i) {
	     if(ctx == (g_group_ctx + i)){
		     continue;
	      }
              ctx2 = g_group_ctx + i;
	      //listen socket
	      //spdk_sock_close(&ctx2->sock);
	      spdk_sock_group_close(&ctx2->group);
	}
	spdk_app_stop(ctx->rc);
	return 0;
}

static int
hello_sock_quit(struct hello_context_t *ctx, int rc)
{
	ctx->rc = rc;
	spdk_poller_unregister(&ctx->poller_out);
	if (!ctx->time_out) {
		ctx->time_out = SPDK_POLLER_REGISTER(hello_sock_close_timeout_poll, ctx,
						     CLOSE_TIMEOUT_US);
	}
	return 0;
}
static void
hello_sock_cb(void *arg, struct spdk_sock_group *group, struct spdk_sock *sock)
{
	ssize_t n;
	char buf[BUFFER_SIZE];
	struct iovec iov;
	struct hello_context_t *ctx = arg;
        uint32_t  current_core;
        current_core = spdk_env_get_current_core();
        SPDK_NOTICELOG("%s current core : %u\n",__func__, current_core);

	n = spdk_sock_recv(sock, buf, sizeof(buf));
	if (n < 0) {
		if (errno == EAGAIN || errno == EWOULDBLOCK) {
			SPDK_ERRLOG("spdk_sock_recv() failed, errno %d: %s\n",
				    errno, spdk_strerror(errno));
			return;
		}

		SPDK_ERRLOG("spdk_sock_recv() failed, errno %d: %s\n",
			    errno, spdk_strerror(errno));
	}

	if (n > 0) {
		ctx->bytes_in += n;
		iov.iov_base = buf;
		iov.iov_len = n;
		n = spdk_sock_writev(sock, &iov, 1);
		if (n > 0) {
			ctx->bytes_out += n;
		}
		return;
	}

	/* Connection closed */
	SPDK_NOTICELOG("Connection closed\n");
	spdk_sock_group_remove_sock(group, sock);
	spdk_sock_close(&sock);
}
struct spdk_sock_group * nvmf_tcp_get_optimal_poll_group(void)
{
	int index = rand()%spdk_env_get_core_count();
	SPDK_NOTICELOG("**************** add client to group %d\n", index);
	return (g_group_ctx + index)->group;
	//return ctx->group;
}
static int
hello_sock_accept_poll(void *arg)
{
	struct hello_context_t *ctx = arg;
	struct spdk_sock *sock;
	int rc;
	int count = 0;
	char saddr[ADDR_STR_LEN], caddr[ADDR_STR_LEN];
	uint16_t cport, sport;

	if (!g_is_running) {
		hello_sock_quit(ctx, 0);
		return 0;
	}

	while (1) {
		sock = spdk_sock_accept(ctx->sock);
		if (sock != NULL) {
			rc = spdk_sock_getaddr(sock, saddr, sizeof(saddr), &sport, caddr, sizeof(caddr), &cport);
			if (rc < 0) {
				SPDK_ERRLOG("Cannot get connection addresses\n");
				spdk_sock_close(&ctx->sock);
				return -1;
			}

			SPDK_NOTICELOG("Accepting a new connection from (%s, %hu) to (%s, %hu)\n",
				       caddr, cport, saddr, sport);

			rc = spdk_sock_group_add_sock(nvmf_tcp_get_optimal_poll_group(), sock,
						      hello_sock_cb, ctx);

			if (rc < 0) {
				spdk_sock_close(&sock);
				SPDK_ERRLOG("failed\n");
				break;
			}

			count++;
		} else {
			if (errno != EAGAIN && errno != EWOULDBLOCK) {
				SPDK_ERRLOG("accept error(%d): %s\n", errno, spdk_strerror(errno));
			}
			break;
		}
	}

	return count;
}
/*
 *
 */
static int
hello_sock_group_poll(void *arg)
{
	struct hello_context_t *ctx = arg;
	int rc;

	rc = spdk_sock_group_poll(ctx->group);
	if (rc < 0) {
		SPDK_ERRLOG("Failed to poll sock_group=%p\n", ctx->group);
	}

	return -1;
}

static int
hello_sock_listen(struct hello_context_t *ctx)
{
	ctx->host = g_host;
	ctx->sock_impl_name = g_sock_impl_name;
	ctx->port = g_port;

	
	SPDK_NOTICELOG("Listening connection on %s:%d with sock_impl(%s)\n", ctx->host, ctx->port,
		       ctx->sock_impl_name);
	ctx->sock = spdk_sock_listen(ctx->host, ctx->port, ctx->sock_impl_name);
	if (ctx->sock == NULL) {
		SPDK_ERRLOG("Cannot create server socket\n");
		return -1;
	}


	/*
	 * Create sock group for server socket
	 */
	ctx->group = spdk_sock_group_create(NULL);

	g_is_running = true;

	/*
	 * Start acceptor and group poller
	 */
	ctx->poller_in = SPDK_POLLER_REGISTER(hello_sock_accept_poll, ctx,
					      ACCEPT_TIMEOUT_US);
	ctx->poller_out = SPDK_POLLER_REGISTER(hello_sock_group_poll, ctx, 0);

	return 0;
}
/*
 static struct spdk_nvme_transport_poll_group *
 nvme_tcp_poll_group_create(void)
 {
      group->sock_group = spdk_sock_group_create(group);
 }
 */
static void nvmf_transport_poll_group_create(struct hello_context_t *ctx)
{
     int rc = 0;
     uint32_t  current_core;
     current_core = spdk_env_get_current_core();
     SPDK_NOTICELOG("%s current core : %u\n",__func__, current_core);
     rc = hello_sock_listen(ctx); 
     if (rc) {
		spdk_app_stop(-1);
		return;
	}
}
static void
nvmf_tgt_create_poll_group(void * arg)
{
     struct hello_context_t *ctx = arg;
     uint32_t  current_core;
     current_core = spdk_env_get_current_core();
     SPDK_NOTICELOG("%s current core : %u\n",__func__, current_core);
     ctx->group = spdk_sock_group_create(NULL);
     ctx->poller_out = SPDK_POLLER_REGISTER(hello_sock_group_poll, ctx, 0);
}


static void
nvmf_tgt_create_poll_groups(void)
{
      uint32_t  current_core;
      struct spdk_cpuset tmp_cpumask = {};
      uint32_t i;
      char thread_name[32];
      struct spdk_thread *thread;
      current_core = spdk_env_get_current_core();
      SPDK_ENV_FOREACH_CORE(i) {
          if (i != current_core) {
#if 1
	                       spdk_cpuset_zero(&tmp_cpumask);
			       spdk_cpuset_set_cpu(&tmp_cpumask, i, true);
			       snprintf(thread_name, sizeof(thread_name), "nvmf_tgt_poll_group_%u", i);
			       thread = spdk_thread_create(thread_name, &tmp_cpumask);
			       assert(thread != NULL);
			       spdk_thread_send_msg(thread, nvmf_tgt_create_poll_group, g_group_ctx + i);
#endif
	 }
      }
}
/* Main program after the hello is started */
static void hello_start(void *arg) {
       
#if 0
     uint32_t  i,current_core;
     struct spdk_lw_thread *lw_thread;
     // lw_thread = spdk_thread_get_ctx(thread);
     struct spdk_thread *thread ;
     struct spdk_reactor *reactor;
     current_core = spdk_env_get_current_core();
     SPDK_NOTICELOG("current core : %u\n", current_core);
     SPDK_ENV_FOREACH_CORE(i) {
          if (i != current_core) {
		 reactor = spdk_reactor_get(i);
		 if (reactor == NULL) {
		             continue;
		  }
		  TAILQ_FOREACH(lw_thread, &reactor->threads, link) {

                          thread = spdk_thread_get_from_ctx(lw_thread);
			  spdk_thread_send_msg(thread, msg_func, NULL) ;
		   }
	  }
     }
 #else
     uint32_t  current_core,total_core;
     total_core = spdk_env_get_core_count();
     current_core = spdk_env_get_current_core();
     SPDK_NOTICELOG("%s current core : %u, total_core : %u \n",__func__, current_core,total_core);
     g_group_ctx= (struct hello_context_t *)calloc(total_core, sizeof(struct hello_context_t));
     nvmf_transport_poll_group_create(g_group_ctx + current_core);
     nvmf_tgt_create_poll_groups();
 #endif
}

/* cleanup to do after termination of app */
static void cleanup(void) {

	if(g_group_ctx){
	    free(g_group_ctx);
	}
}

static void
hello_sock_usage(void)
{
	printf(" -H host_addr  host address\n");
	printf(" -P port       port number\n");
	printf(" -N sock_impl  socket implementation, e.g., -N posix or -N uring\n");
	printf(" -S            start in server mode\n");
	printf(" -V            print out additional informations\n");
}

/*
 * This function is called to parse the parameters that are specific to this application
 */
static int hello_sock_parse_arg(int ch, char *arg)
{
	switch (ch) {
	case 'H':
		g_host = arg;
		break;
	case 'N':
		g_sock_impl_name = arg;
		break;
	case 'P':
		g_port = spdk_strtol(arg, 10);
		if (g_port < 0) {
			fprintf(stderr, "Invalid port ID\n");
			return g_port;
		}
		break;
	case 'S':
		g_is_server = 1;
		break;
	case 'V':
		g_verbose = true;
		break;
	default:
		return -EINVAL;
	}
	return 0;
}

static void
hello_sock_shutdown_cb(void)
{
	g_is_running = false;
}
int main(int argc, char **argv) {
    int rc=0;
    struct spdk_app_opts opts = {};
    /* Use all four cores, 0xF = 0b1111 */
    /* Initialize the app event framework */
    spdk_app_opts_init(&opts, sizeof(opts));
    opts.reactor_mask = "0x3F";
    opts.name = "hello_sock";
    opts.shutdown_cb = hello_sock_shutdown_cb;

	if ((rc = spdk_app_parse_args(argc, argv, &opts, "H:N:P:SV", NULL, hello_sock_parse_arg,
				      hello_sock_usage)) != SPDK_APP_PARSE_ARGS_SUCCESS) {
		exit(rc);
	}
    SPDK_NOTICELOG("Total cores available: %d\n", spdk_env_get_core_count());
    /* Start the event app framework */
    if (spdk_app_start(&opts, hello_start, NULL)) {
        fprintf(stderr, "Failed to start app framework\n");
        return 1;
    }
    printf("app program successfully terminated\n");
    cleanup();

    spdk_app_fini();
    return 0;
}
