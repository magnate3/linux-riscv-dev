#include <stdio.h>
#include <spdk/bdev.h>
#include <spdk/thread.h>
#include <spdk/queue.h>
#include "spdk/thread.h"
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/likely.h"
struct nvmf_lw_thread {
	TAILQ_ENTRY(nvmf_lw_thread) link;
	bool resched;
};

struct nvmf_reactor {
	uint32_t core;

	struct spdk_ring		*threads;
	TAILQ_ENTRY(nvmf_reactor)	link;
};
static struct nvmf_reactor *g_main_reactor = NULL;
TAILQ_HEAD(, nvmf_reactor) g_reactors = TAILQ_HEAD_INITIALIZER(g_reactors);
static struct nvmf_reactor *g_next_reactor = NULL;
static struct spdk_thread *g_init_thread = NULL;
static struct spdk_thread *g_fini_thread = NULL;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_reactors_exit = false;
static int nvmf_schedule_spdk_thread(struct spdk_thread *thread);
static struct spdk_poller * g_poller = NULL;
static int g_time_in_sec = 5;
static int poller_stop(void *arg)
{

        printf("hello poller\n");
#if 0
	spdk_poller_unregister(&g_poller);
#endif
        return SPDK_POLLER_BUSY;
}
// refer to accel_perf_start(void *arg1)
static void hello_reader(void *arg1)
{
    printf("hello reader \n");
}
static void hello_start(void *arg1)
{
    struct spdk_thread* thread = spdk_thread_create("first_reader_thread", NULL);
    struct nvmf_lw_thread *lw_thread;
    lw_thread = spdk_thread_get_ctx(thread);
    assert(lw_thread != NULL);
    memset(lw_thread, 0, sizeof(*lw_thread));
    spdk_ring_enqueue(g_main_reactor->threads, (void **)&lw_thread, 1, NULL);
    spdk_thread_send_msg(thread, hello_reader, NULL);
    g_poller = SPDK_POLLER_REGISTER(poller_stop, NULL, g_time_in_sec * 1000000ULL);
    printf("hello start \n");
#if 0
    // will cause coredump
    spdk_app_stop(0);
#endif
}
// refer to ./examples/nvmf/nvmf/nvmf.c


static void
nvmf_request_spdk_thread_reschedule(struct spdk_thread *thread)
{
	struct nvmf_lw_thread *lw_thread;

	assert(thread == spdk_get_thread());

	lw_thread = spdk_thread_get_ctx(thread);

	assert(lw_thread != NULL);

	lw_thread->resched = true;
}
static int
nvmf_reactor_thread_op(struct spdk_thread *thread, enum spdk_thread_op op)
{
	switch (op) {
	case SPDK_THREAD_OP_NEW:
                printf("%s new thread =%p   \n",__func__, thread);
		return nvmf_schedule_spdk_thread(thread); 
	case SPDK_THREAD_OP_RESCHED:
                printf("%s new reshedule thread \n",__func__);
		nvmf_request_spdk_thread_reschedule(thread);
		return 0;
	default:
		return -ENOTSUP;
	}
}
static bool
nvmf_reactor_thread_op_supported(enum spdk_thread_op op)
{
	        switch (op) {
                case SPDK_THREAD_OP_NEW:
                case SPDK_THREAD_OP_RESCHED:
	                return true;
	        default:
                return false;
	        }
}

static int
nvmf_reactor_run(void *arg)
{
	struct nvmf_reactor *nvmf_reactor = arg;
	struct nvmf_lw_thread *lw_thread;
	struct spdk_thread *thread;

	/* run all the lightweight threads in this nvmf_reactor by FIFO. */
	do {
		if (spdk_ring_dequeue(nvmf_reactor->threads, (void **)&lw_thread, 1)) {
			thread = spdk_thread_get_from_ctx(lw_thread);

			spdk_thread_poll(thread, 0, 0);

			if (spdk_unlikely(spdk_thread_is_exited(thread) &&
					  spdk_thread_is_idle(thread))) {
				spdk_thread_destroy(thread);
			} else if (spdk_unlikely(lw_thread->resched)) {
				lw_thread->resched = false;
				nvmf_schedule_spdk_thread(thread);
			} else {
				spdk_ring_enqueue(nvmf_reactor->threads, (void **)&lw_thread, 1, NULL);
			}
		}
	} while (!g_reactors_exit);

	/* free all the lightweight threads */
	while (spdk_ring_dequeue(nvmf_reactor->threads, (void **)&lw_thread, 1)) {
		thread = spdk_thread_get_from_ctx(lw_thread);
		spdk_set_thread(thread);

		if (spdk_thread_is_exited(thread)) {
			spdk_thread_destroy(thread);
		} else {
			/* This thread is not exited yet, and may need to communicate with other threads
			 * to be exited. So mark it as exiting, and check again after traversing other threads.
			 */
			spdk_thread_exit(thread);
			spdk_thread_poll(thread, 0, 0);
			spdk_ring_enqueue(nvmf_reactor->threads, (void **)&lw_thread, 1, NULL);
		}
	}

	return 0;
}
static int
nvmf_schedule_spdk_thread(struct spdk_thread *thread)
{
	struct nvmf_reactor *nvmf_reactor;
	struct nvmf_lw_thread *lw_thread;
	struct spdk_cpuset *cpumask;
	uint32_t i;

	/* Lightweight threads may have a requested cpumask.
	 * This is a request only - the scheduler does not have to honor it.
	 * For this scheduler implementation, each reactor is pinned to
	 * a particular core so honoring the request is reasonably easy.
	 */
	cpumask = spdk_thread_get_cpumask(thread);

	lw_thread = spdk_thread_get_ctx(thread);
	assert(lw_thread != NULL);
	memset(lw_thread, 0, sizeof(*lw_thread));

	/* assign lightweight threads to nvmf reactor(core)
	 * Here we use the mutex.The way the actual SPDK event framework
	 * solves this is by using internal rings for messages between reactors
	 */
	pthread_mutex_lock(&g_mutex);
	for (i = 0; i < spdk_env_get_core_count(); i++) {
		if (g_next_reactor == NULL) {
			g_next_reactor = TAILQ_FIRST(&g_reactors);
		}
		nvmf_reactor = g_next_reactor;
		g_next_reactor = TAILQ_NEXT(g_next_reactor, link);

		/* each spdk_thread has the core affinity */
		if (spdk_cpuset_get_cpu(cpumask, nvmf_reactor->core)) {
			spdk_ring_enqueue(nvmf_reactor->threads, (void **)&lw_thread, 1, NULL);
			break;
		}
	}
	pthread_mutex_unlock(&g_mutex);

	if (i == spdk_env_get_core_count()) {
		fprintf(stderr, "failed to schedule spdk thread\n");
		return -1;
	}
	return 0;
}

static int
nvmf_init_threads(void)
{
	int rc;
	uint32_t i;
	char thread_name[32];
	struct nvmf_reactor *nvmf_reactor;
	struct spdk_cpuset cpumask;
	uint32_t main_core = spdk_env_get_current_core();

	/* Whenever SPDK creates a new lightweight thread it will call
	 * nvmf_schedule_spdk_thread asking for the application to begin
	 * polling it via spdk_thread_poll(). Each lightweight thread in
	 * SPDK optionally allocates extra memory to be used by the application
	 * framework. The size of the extra memory allocated is the second parameter.
	 */
	spdk_thread_lib_init_ext(nvmf_reactor_thread_op, nvmf_reactor_thread_op_supported,
				 sizeof(struct nvmf_lw_thread));

	/* Spawn one system thread per CPU core. The system thread is called a reactor.
	 * SPDK will spawn lightweight threads that must be mapped to reactors in
	 * nvmf_schedule_spdk_thread. Using a single system thread per CPU core is a
	 * choice unique to this application. SPDK itself does not require this specific
	 * threading model. For example, another viable threading model would be
	 * dynamically scheduling the lightweight threads onto a thread pool using a
	 * work queue.
	 */
	SPDK_ENV_FOREACH_CORE(i) {
		nvmf_reactor = calloc(1, sizeof(struct nvmf_reactor));
		if (!nvmf_reactor) {
			fprintf(stderr, "failed to alloc nvmf reactor\n");
			rc = -ENOMEM;
			goto err_exit;
		}

		nvmf_reactor->core = i;

		nvmf_reactor->threads = spdk_ring_create(SPDK_RING_TYPE_MP_SC, 1024, SPDK_ENV_SOCKET_ID_ANY);
		if (!nvmf_reactor->threads) {
			fprintf(stderr, "failed to alloc ring\n");
			free(nvmf_reactor);
			rc = -ENOMEM;
			goto err_exit;
		}

		TAILQ_INSERT_TAIL(&g_reactors, nvmf_reactor, link);

		if (i == main_core) {
			g_main_reactor = nvmf_reactor;
			g_next_reactor = g_main_reactor;
		} else {
			rc = spdk_env_thread_launch_pinned(i,
							   nvmf_reactor_run,
							   nvmf_reactor);
			if (rc) {
				fprintf(stderr, "failed to pin reactor launch\n");
				goto err_exit;
			}
		}
	}

	/* Spawn a lightweight thread only on the current core to manage this application. */
	spdk_cpuset_zero(&cpumask);
	spdk_cpuset_set_cpu(&cpumask, main_core, true);
	snprintf(thread_name, sizeof(thread_name), "nvmf_main_thread");
	g_init_thread = spdk_thread_create(thread_name, &cpumask);
        printf("%s g_init_thread  =%p   \n",__func__, g_init_thread);
	if (!g_init_thread) {
		fprintf(stderr, "failed to create spdk thread\n");
		return -1;
	}

	fprintf(stdout, "nvmf threads initlize successfully\n");
	return 0;

err_exit:
	return rc;
}

static void
nvmf_destroy_threads(void)
{
	struct nvmf_reactor *nvmf_reactor, *tmp;

	TAILQ_FOREACH_SAFE(nvmf_reactor, &g_reactors, link, tmp) {
		spdk_ring_free(nvmf_reactor->threads);
		free(nvmf_reactor);
	}

	pthread_mutex_destroy(&g_mutex);
	spdk_thread_lib_fini();
	fprintf(stdout, "nvmf threads destroy successfully\n");
}
int main(int argc, char **argv)
{
    int rc;
    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "reactor-test-v2";
    if (spdk_env_init(&opts) < 0) {
	   fprintf(stderr, "unable to initialize SPDK env\n");
	   return -EINVAL;
    }
      /* Initialize the threads */
    rc = nvmf_init_threads();
    assert(rc == 0);
    spdk_thread_send_msg(g_init_thread, hello_start, NULL);
    nvmf_reactor_run(g_main_reactor);

    printf("main thread wait all \n");
    spdk_env_thread_wait_all();
    nvmf_destroy_threads();
    spdk_env_fini();
    return 0;
}
