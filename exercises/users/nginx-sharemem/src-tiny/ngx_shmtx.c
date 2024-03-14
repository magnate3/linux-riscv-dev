
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */

#include "ngx_shmtx.h"
#include "config.h"

static void ngx_shmtx_wakeup(ngx_shmtx_t *mtx);

extern ngx_pid_t ngx_pid;


ngx_int_t
ngx_shmtx_create(ngx_shmtx_t *mtx, ngx_shmtx_sh_t *addr, u_char *name)
{
    init_ncpu();
//    init_pid();
    mtx->lock = &addr->lock;

    if (mtx->spin == (ngx_uint_t) -1) {
        return NGX_OK;
    }

    mtx->spin = 2048;


    mtx->wait = &addr->wait;

    if (sem_init(&mtx->sem, 1, 0) == -1) {
//        ngx_log_error(NGX_LOG_ALERT, ngx_cycle->log, ngx_errno,
//                      "sem_init() failed");
    } else {
        mtx->semaphore = 1;
    }

    return NGX_OK;
}


void
ngx_shmtx_destroy(ngx_shmtx_t *mtx)
{
    if (mtx->semaphore) {
        if (sem_destroy(&mtx->sem) == -1) {
//            ngx_log_error(NGX_LOG_ALERT, ngx_cycle->log, ngx_errno,
//                          "sem_destroy() failed");
        }
    }
}


ngx_uint_t
ngx_shmtx_trylock(ngx_shmtx_t *mtx)
{
    return (*mtx->lock == 0 && ngx_atomic_cmp_set(mtx->lock, 0, ngx_pid));
}


void
ngx_shmtx_lock(ngx_shmtx_t *mtx)
{
    ngx_uint_t         i, n;

//    ngx_log_debug0(NGX_LOG_DEBUG_CORE, ngx_cycle->log, 0, "shmtx lock");

    for ( ;; ) {

        if (*mtx->lock == 0 && ngx_atomic_cmp_set(mtx->lock, 0, ngx_pid)) {
            return;
        }

        if (ngx_ncpu > 1) {

            for (n = 1; n < mtx->spin; n <<= 1) {

                for (i = 0; i < n; i++) {
                    ngx_cpu_pause();
                }

                if (*mtx->lock == 0
                    && ngx_atomic_cmp_set(mtx->lock, 0, ngx_pid))
                {
                    return;
                }
            }
        }


        if (mtx->semaphore) {
            (void) ngx_atomic_fetch_add(mtx->wait, 1);

            if (*mtx->lock == 0 && ngx_atomic_cmp_set(mtx->lock, 0, ngx_pid)) {
                (void) ngx_atomic_fetch_add(mtx->wait, -1);
                return;
            }

//            ngx_log_debug1(NGX_LOG_DEBUG_CORE, ngx_cycle->log, 0,
//                           "shmtx wait %uA", *mtx->wait);

            while (sem_wait(&mtx->sem) == -1) {
                int  err;

                err = errno;

                if (err != NGX_EINTR) {
//                    ngx_log_error(NGX_LOG_ALERT, ngx_cycle->log, err,
//                                  "sem_wait() failed while waiting on shmtx");
                    break;
                }
            }

//            ngx_log_debug0(NGX_LOG_DEBUG_CORE, ngx_cycle->log, 0,
//                           "shmtx awoke");

            continue;
        }


        ngx_sched_yield();
    }
}


void
ngx_shmtx_unlock(ngx_shmtx_t *mtx)
{
    if (mtx->spin != (ngx_uint_t) -1) {
//        ngx_log_debug0(NGX_LOG_DEBUG_CORE, ngx_cycle->log, 0, "shmtx unlock");
    }

    if (ngx_atomic_cmp_set(mtx->lock, ngx_pid, 0)) {
        ngx_shmtx_wakeup(mtx);
    }
}


ngx_uint_t
ngx_shmtx_force_unlock(ngx_shmtx_t *mtx, ngx_pid_t pid)
{
//    ngx_log_debug0(NGX_LOG_DEBUG_CORE, ngx_cycle->log, 0,
//                   "shmtx forced unlock");

    if (ngx_atomic_cmp_set(mtx->lock, pid, 0)) {
        ngx_shmtx_wakeup(mtx);
        return 1;
    }

    return 0;
}


static void
ngx_shmtx_wakeup(ngx_shmtx_t *mtx)
{
    ngx_atomic_uint_t  wait;

    if (!mtx->semaphore) {
        return;
    }

    for ( ;; ) {

        wait = *mtx->wait;

        if ((ngx_atomic_int_t) wait <= 0) {
            return;
        }

        if (ngx_atomic_cmp_set(mtx->wait, wait, wait - 1)) {
            break;
        }
    }

//    ngx_log_debug1(NGX_LOG_DEBUG_CORE, ngx_cycle->log, 0,
//                   "shmtx wake %uA", wait);

    if (sem_post(&mtx->sem) == -1) {
//        ngx_log_error(NGX_LOG_ALERT, ngx_cycle->log, ngx_errno,
//                      "sem_post() failed while wake shmtx");
    }

}

void
ngx_spinlock(ngx_atomic_t *lock, ngx_atomic_int_t value, ngx_uint_t spin)
{
    ngx_uint_t  i, n;

    for ( ;; ) {

        if (*lock == 0 && ngx_atomic_cmp_set(lock, 0, value)) {
            return;
        }

        if (ngx_ncpu > 1) {

            for (n = 1; n < spin; n <<= 1) {

                for (i = 0; i < n; i++) {
                    ngx_cpu_pause();
                }

                if (*lock == 0 && ngx_atomic_cmp_set(lock, 0, value)) {
                    return;
                }
            }
        }

        ngx_sched_yield();
    }
}