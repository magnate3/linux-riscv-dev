

# 中断模式


# thread_send_msg_notification
```
static inline int
thread_send_msg_notification(const struct spdk_thread *target_thread)
{
        uint64_t notify = 1;
        int rc;

        /* Not necessary to do notification if interrupt facility is not enabled */
        if (spdk_likely(!spdk_interrupt_mode_is_enabled())) {
                return 0;
        }

        /* When each spdk_thread can switch between poll and interrupt mode dynamically,
         * after sending thread msg, it is necessary to check whether target thread runs in
         * interrupt mode and then decide whether do event notification.
         */
        if (spdk_unlikely(target_thread->in_interrupt)) {
                rc = write(target_thread->msg_fd, &notify, sizeof(notify));
                if (rc < 0) {
                        SPDK_ERRLOG("failed to notify msg_queue: %s.\n", spdk_strerror(errno));
                        return -EIO;
                }
        }

        return 0;
}
```