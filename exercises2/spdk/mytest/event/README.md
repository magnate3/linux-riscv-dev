
#   thread_send_msg_notification

```C
int
spdk_thread_send_critical_msg(struct spdk_thread *thread, spdk_msg_fn fn)
{
	spdk_msg_fn expected = NULL;

	if (!__atomic_compare_exchange_n(&thread->critical_msg, &expected, fn, false, __ATOMIC_SEQ_CST,
					 __ATOMIC_SEQ_CST)) {
		return -EIO;
	}

	return thread_send_msg_notification(thread);
}
```

```C
int
spdk_thread_send_msg(const struct spdk_thread *thread, spdk_msg_fn fn, void *ctx)
{


	return thread_send_msg_notification(thread);
}
```