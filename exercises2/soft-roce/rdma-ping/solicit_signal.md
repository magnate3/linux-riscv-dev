#  RDMA cq event机制-ibv_req_notify_cq


cq_event机制的使用可以减少程序执行过程中对CPU的消耗，因为CQEs的获取如果通过poll cq循环来实现会持续占用core，而cq_event机制则可以使得core被释放，等CQEs进入CQ才会产生event，程序可以监控此event，往下执行。ibv_req_notify_cq()函数将对特定cq使能cq_event监控机制，后续该CQ的所有CQE的产生都是通过用户等待event产生而被知悉。其函数原型为：

```
/**
* ibv_req_notify_cq - Request completion notification on a CQ. An
* event will be added to the completion channel associated with the
* CQ when an entry is added to the CQ.
* @cq: The completion queue to request notification for.
* @solicited_only: If non-zero, an event will be generated only for
* the next solicited CQ entry. If zero, any CQ entry, solicited or
* not, will generate an event.
*/
```
```
static inline int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only)
```
signaled机制是控制发送端CQE产生的。通常一个QP连续发送多个WR，为了避免每次都进行poll cq校验CQE，则可以指定某WR的send_flags带有IBV_SEND_SIGNALLED标志，则最终只有这些WR会产生CQE，进行校验即可，这是一种提升程序性能的方式，因为中间未预期CQE的WQE如果send/recv的结果校验错误，也会最终体现在我们预期的CQEs发生错误。本文将signaled机制和cq_event机制整体讨论。因为这两个机制一起使用可以达到提升程序性能的目的，减少CPU consumption！

具体解释如下：   
 
##  send端       
先说CQE的产生：

受到wr.send_flags中的IBV_SEND_SIGNALLED标志控制，在QP创建时
+ sq_sig_all = 0的情况下，只有IBV_SEND_SIGNALED标志置位才会产生cqe     
+ 在sq_sig_all = 1的情况下，所有WQEs的完成都会产生cqe, 不会受IBV_SEND_SIGNALED标志的影响;  

+ 再说event的产生：

如果不使用cq_event_notify()机制, 则预期不会产生 event；    

如果使能了ibv_request_cq_notify(solicited_only = 0)机制, 则预期带有IBV_SNED_SOLICITED flag的WR产生的unsuccessful send CQE才会产生event；不带有IBV_SEND_SOLICITED flag的WR 产生的无论正确的还是错误的send CQE都会产生event；    

如果使能了ibv_request_cq_notify(solicited_only = 1)机制，则预期带有IBV_SNED_SOLICITED flag的WR产生的unsuccessful send CQE才会产生event。   

##  recv端
会正常产生CQE。不会受IBV_SEND_SIGNALED标志的影响;    



如果不使用cq_event_notify()机制, 则预期不会产生event；   

+ 如果使用cq_event_notify(solicited_only = 0)机制，任意recv cqe 会产生cq_event；   

+ 如果使用cq_event_notify(solicited_only = 1)机制, 只有incoming    的带有IBV_SEND_SOLICITED标志的wr产生的CQE才能产生cq_event。         

————这些机制必须跟业务流程结合起来使用！比如recv端预期cq_event，如果使用cq_event_notify(solicited_only = 1)机制，则流程中所有不带IBV_SEND_SOLICITED标志的RR产生的CQE将不会产生event，走正常的循环poll cq流程去获取CQE即可。cq_event机制使能之后并不是说必须得使用ibv_get_cq_event()去获得CQE，只是提供了一种可以等待event产生再去获取CQE的机制，可以减少poll cq循环执行对core的占用，减轻CPU负担。        



以上总结来自Dotan的博客中的解释：ibv_req_notify_cq()       
```
There are two types of Completion Events that can be requested:

1.Solicited Completion Event - Occurs when an incoming Send or RDMA Write with Immediate Data message with the Solicited Event indicator set (i.e. the remote side posted a Send Request with IBV_SEND_SOLICITED set in send_flags) causes a generation of a successful Receive Work Completion or any unsuccessful (either Send of Receive) Work Completion to be added to the CQ.

————IBV_SEND_SOLICITED 非零

2.Unsolicited Completion Event - occurs when any Work Completion is added to the CQ, whether it is a Send or Receive Work Completion and whether is it successful or unsuccessful Work Completion.

————IBV_SEND_SOLICITED 为零
```



```
int process_work_completion_event(struct ibv_comp_channel *completion_channel,
                                  struct ibv_wc *wc, int expected_wc)
{
        struct ibv_cq *cq_ptr = NULL;
        void *context = NULL; /* User-defined CQ context, N/A here */
        int ret = 0;
        int total_wc = 0; /* Number of WC elements we've processed so far */

        /* Blocks and waits for the next IO completion event */
        ret = ibv_get_cq_event(
                completion_channel, /* IO Completion Channel */
                &cq_ptr, /* Which CQ has activity, should match same CQ we created */
                &context /* User context for CQ, which we didn't set */
        );
        if (ret) {
                fprintf(stderr, "Failed to get CQ event: %s\n",
                        strerror(errno));
		return -errno;
        }

        /* Immediately request more notifications */
        ret = ibv_req_notify_cq(cq_ptr, 0);
        if (ret) {
                fprintf(stderr, "Failed to request notifications for CQ events: %s\n",
                        strerror(errno));
		return -errno;
        }

        /* Since we've received a CQ notification, we now need to process
         * expected_wc WC elements. ibv_poll_cq() can return 0 or more WC
         * elements, or errno in the case of failure to poll.
         */
        do {
                ret = ibv_poll_cq(
                        cq_ptr, /* The CQ we got a notification for */
                        expected_wc - total_wc, /* Remaining WC elements */
                        wc + total_wc
                );
                if (ret < 0) {
                        /* ret is errno, in case of failure */
                        fprintf(stderr, "Failed to poll the CQ for a WC event: %s\n",
                                strerror(ret));
		        return -ret;
                }
                total_wc += ret;
        } while (total_wc < expected_wc);

        /* Now that we've gotten expected_wc WC elements, we need to check each
         * one's status.
         */
        for (int i = 0; i < total_wc; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                        fprintf(stderr, "Failed status %s (%d) for wr_id %d\n",
		                ibv_wc_status_str(wc[i].status),
		                wc[i].status, (int)wc[i].wr_id);
	                return -1;
                }
                printf("Work Request %d status: %s\n", (int)wc[i].wr_id,
                       ibv_wc_status_str(wc[i].status));
        }

        /* Finally, ACK the CQ event. We only got 1 CQ event notification for
         * n WR elements; this is not the number of WC elements we got/expected.
         */
        ibv_ack_cq_events(cq_ptr, 1);

        return total_wc;
}
```