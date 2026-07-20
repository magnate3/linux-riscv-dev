

# dump_stack
```
static int rxe_post_send(struct ib_qp *ibqp, struct ib_send_wr *wr,
                         struct ib_send_wr **bad_wr)
{
        struct rxe_qp *qp = to_rqp(ibqp);
        dump_stack();
        if (unlikely(!qp->valid)) {
                *bad_wr = wr;
                return -EINVAL;
        }

        if (unlikely(qp->req.state < QP_STATE_READY)) {
                *bad_wr = wr;
                return -EINVAL;
        }

        if (qp->is_user) {
                /* Utilize process context to do protocol processing */
                rxe_run_task(&qp->req.task, 0);
                return 0;
        } else
                return rxe_post_send_kernel(qp, wr, bad_wr);
}
```

```
[501031.783755] Call trace:
[501031.786278] [<ffff000008089e14>] dump_backtrace+0x0/0x23c
[501031.791739] [<ffff00000808a074>] show_stack+0x24/0x2c
[501031.796852] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[501031.801973] [<ffff000000c27a38>] rxe_post_send+0x3c/0x534 [rdma_rxe]
[501031.808389] [<ffff000003457ac0>] ib_uverbs_post_send+0x5d8/0x600 [ib_uverbs]
[501031.815496] [<ffff000003450bd0>] ib_uverbs_write+0x1b8/0x430 [ib_uverbs]
[501031.822251] [<ffff0000082b2620>] __vfs_write+0x58/0x180
[501031.827538] [<ffff0000082b2960>] vfs_write+0xb0/0x1a8
```