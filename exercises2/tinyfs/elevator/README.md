
#  禁用调度器的处理

[block多队列分析 - 3. 读文件过程](https://blog.csdn.net/jasonactions/article/details/116662524)   
```
  blk_mq_sched_insert_requests
|- -blk_mq_try_issue_list_directly（禁用调度器的处理）
|- -dd_insert_requests（使能调度器的处理）
|- -blk_mq_run_hw_queue
```



#  elv_register
```
static struct elevator_type iosched_deadline = {
        .ops.sq = {
                .elevator_merge_fn =            deadline_merge,
                .elevator_merged_fn =           deadline_merged_request,
                .elevator_merge_req_fn =        deadline_merged_requests,
                .elevator_dispatch_fn =         deadline_dispatch_requests,
                .elevator_add_req_fn =          deadline_add_request,
                .elevator_former_req_fn =       elv_rb_former_request,
                .elevator_latter_req_fn =       elv_rb_latter_request,
                .elevator_init_fn =             deadline_init_queue,
                .elevator_exit_fn =             deadline_exit_queue,
        },

        .elevator_attrs = deadline_attrs,
        .elevator_name = "deadline",
        .elevator_owner = THIS_MODULE,
};

static int __init deadline_init(void)
{
        return elv_register(&iosched_deadline);
}
```