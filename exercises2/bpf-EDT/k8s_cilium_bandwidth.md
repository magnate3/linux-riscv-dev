# intro

TBF（Token Bucket Filter）是个令牌桶，所有连接/流量都要经过单个队列排队处理,在设计上存在的问题：
1. TBF qdisc 所有 CPU 共享一个锁（著名的 qdisc root lock），因此存在锁竞争；流量越大锁开销越大；
2. veth pair 是单队列（single queue）虚拟网络设备，因此物理网卡的 多队列（multi queue，不同 CPU 处理不同 queue，并发）优势到了这里就没用了， 大家还是要走到同一个队列才能进到 pod；
3. 在入向排队是不合适的（no-go），会占用大量系统资源和缓冲区开销（bufferbloat）。

出向（egress）限速存在的问题：
1. Pod egress 对应 veth 主机端的 ingress，ingress 是不能做整形的，因此加了一个 ifb 设备；
2. 所有从 veth 出来的流量会被重定向到 ifb 设备，通过 ifb TBF qdisc 设置容器限速。
3. 原来只需要在物理网卡排队（一般都会设置一个默认 qdisc，例如 pfifo_fast/fq_codel/noqueue），现在又多了一层 ifb 设备排队，缓冲区膨胀（bufferbloat）；
4. 与 ingress 一样，存在 root qdisc lock 竞争，所有 CPU 共享；
5. 干扰 TCP Small Queues (TSQ) 正常工作；TSQ 作用是减少 bufferbloat， 工作机制是觉察到发出去的包还没有被有效处理之后就减少发包；ifb 使得包都缓存在 qdisc 中， 使 TSQ 误以为这些包都已经发出去了，实际上还在主机内。
6. 延迟显著增加：每个 pod 原来只需要 2 个网络设备，现在需要 3 个，增加了大量 queueing 逻辑。

edt两点核心转变：
1. 每个包（skb）打上一个最早离开时间（Earliest Departure Time, EDT），也就是最早可以发送的时间戳；
2. 用时间轮调度器（timing-wheel scheduler）替换原来的出向缓冲队列（qdisc queue）

Cilium 的 bandwidth manager:
1. 基于 eBPF+EDT，实现了无锁 的 pod 限速功能；
2. 在物理网卡（或 bond 设备）而不是 veth 上限速，避免了 bufferbloat，也不会扰乱 TCP TSQ 功能。
3. 不需要进入协议栈，Cilium 的 BPF host routing 功能，使得 FIB lookup 等过程完全在 TC eBPF 层完成，并且能直接转发到网络设备。
4. 在物理网卡（或 bond 设备）上添加 MQ/FQ，实现时间轮调度。


## 工作原理

1. Cilium attach 到宿主机的物理网卡（或 bond 设备），在 BPF 程序中为每个包设置 timestamp， 然后通过 earliest departure time 在 fq 中实现限速。容器限速是在物理网卡上做的，而不是在每个 pod 的 veth 设备上。这跟之前基于 ifb 的限速方案有很大不同。
   1. BPF 程序：管理（计算和设置） skb 的 departure timestamp；
   2. TC qdisc (multi-queue) 发包调度；
   3. 物理网卡的队列。
   4. 如果宿主机使用了 bond，那么根据 bond 实现方式的不同，FQ 的数量会不一样， 可通过 tc -s -d qdisc show dev {bond} 查看实际状态。
   5. Linux bond 默认支持多队列（multi-queue），会默认创建 16 个 queue， 每个 queue 对应一个 FQ，挂在一个 MQ 下面；
   6. OVS bond 不支持 MQ，因此只有一个 FQ（v2.3 等老版本行为，新版本不清楚）。
   7. bond 设备的 TXQ 数量，可以通过 ls /sys/class/net/{dev}/queues/ 查看。 物理网卡的 TXQ 数量也可以通过以上命令看，但 ethtool -l {dev} 看到的信息更多，包括了最大支持的数量和实际启用的数量。
2. egress 限速工作流程：
   1. Pod egress 流量从容器进入宿主机，此时会发生 netns 切换，但 socket 信息 skb->sk 不会丢失；
   2. Host 端 veth 上的 BPF 标记（marking）包的 aggregate（queue_mapping），见 Cilium 代码(lib/edt.h)；
   3. 物理网卡上的 BPF 程序根据 aggregate 设置的限速参数，设置每个包的时间戳 skb->tstamp；
   4. FQ+MQ 基本实现了一个 timing-wheel 调度器，根据 skb->tstamp 调度发包。
   5. bpf map 存储 aggregate 信息

```C
THROTTLE_MAP
struct edt_id aggregate;
struct edt_info *info;

edt_set_aggregate(ctx, LXC_ID)
   ctx->queue_mapping = aggregate

edt_sched_departure()
   info = map_lookup_elem(&THROTTLE_MAP, &aggregate);
   now = ktime_get_ns();
   t = ctx->tstamp;
   delay = ((__u64)ctx_wire_len(ctx)) * NSEC_PER_SEC / info->bps; // 数据包根据长度可以delay的时间
	t_next = READ_ONCE(info->t_last) + delay; // 计算最晚发送时间
   t_next - now >= info->t_horizon_drop // 如果比较晚，就丢掉drop
   ctx->tstamp = t_next; // 设置一个最晚时间，应该是给后续的to_netdev/to_overlay使用
```

## EDT 还能支持 BBR

完整 BBR 设计可参考 (论文) BBR：[基于拥塞（而非丢包）的拥塞控制（ACM, 2017）](https://arthurchiao.art/blog/better-bandwidth-management-with-ebpf-zh/%7B%20%%20link%20_posts/2022-01-02-bbr-paper-zh.md%20%%7D)。

### bbr对比cubic

### 跨 netns 时，skb->tstamp 被重置。

BBR + FQ 机制上是能协同工作的；但是，内核在 skb 离开 pod netns 时，将 skb 的时间戳清掉了，导致包进入 host netns 之后没有时间戳，FQ 无法工作。

几种时间规范：https://www.cl.cam.ac.uk/~mgk25/posix-clocks.html

对于包的时间戳 skb->tstamp，内核根据包的方向（RX/TX）不同而使用的两种时钟源：
- Ingress 使用 CLOCK_TAI (TAI: international atomic time)
- Egress 使用 CLOCK_MONOTONIC（也是 FQ 使用的时钟类型）
如果不重置，将包从 RX 转发到 TX 会导致包在 FQ 中被丢弃，因为 [超过 FQ 的 drop horizon](https://github.com/torvalds/linux/blob/v5.10/net/sched/sch_fq.c#L463)。 FQ horizon [默认是 10s](https://github.com/torvalds/linux/blob/v5.10/net/sched/sch_fq.c#L950)。

horizon 是 FQ 的一个配置项，表示一个时间长度， 在 [net_sched: sch_fq: add horizon attribute](https://github.com/torvalds/linux/commit/39d010504e6b) 引入，
```text
QUIC servers would like to use SO_TXTIME, without having CAP_NET_ADMIN,
to efficiently pace UDP packets.

As far as sch_fq is concerned, we need to add safety checks, so
that a buggy application does not fill the qdisc with packets
having delivery time far in the future.

This patch adds a configurable horizon (default: 10 seconds),
and a configurable policy when a packet is beyond the horizon
at enqueue() time:
- either drop the packet (default policy)
- or cap its delivery time to the horizon.
```
简单来说，如果一个包的时间戳离现在太远，就直接将这个包 丢弃，或者将其改为一个上限值（cap），以便节省队列空间；否则，这种 包太多的话，队列可能会被塞满，导致时间戳比较近的包都无法正常处理。 内核代码如下：
```c
static bool fq_packet_beyond_horizon(const struct sk_buff *skb, const struct fq_sched_data *q)
{
    return unlikely((s64)skb->tstamp > (s64)(q->ktime_cache + q->horizon));
}
```

另外，现在给定一个包，我们无法判断它用的是哪种 timestamp，因此只能用这种 reset 方式。

### skb->tstamp 统一到同一种时钟吗？

其实最开始，TCP EDT 用的也是 CLOCK_TAI 时钟。 但有人在[邮件列表](https://lore.kernel.org/netdev/2185d09d-90e1-81ef-7c7f-346eeb951bf4@gmail.com/) 里反馈说，某些特殊的嵌入式设备上重启会导致时钟漂移 50 多年。所以后来 EDT 又回到了 monotonic 时钟，而我们必须跨 netns 时 reset。

我们做了个原型验证，新加一个 bit skb->tstamp_base 来解决这个问题，

- 0 表示使用的 TAI，
- 1 表示使用的 MONO，
然后，
- TX/RX 通过 skb_set_tstamp_{mono,tai}(skb, ktime) helper 来获取这个值，
- fq_enqueue() 先检查 timestamp 类型，如果不是 MONO，就 reset skb->tstamp
此外，
- 转发逻辑中所有 skb->tstamp = 0 都可以删掉了
- skb_mstamp_ns union 也可能删掉了
- 在 RX 方向，net_timestamp_check() 必须推迟到 tc ingress 之后执行

我们和 Facebook 的朋友合作，已经解决了这个问题，在跨 netns 时保留时间戳， patch 并合并到了 kernel 5.18+。 因此 BBR+EDT 可以工作了，



## cilium bandwidth manager

- 启用本地路由 (Native Routing)
- 完全替换 KubeProxy
- IP 地址伪装 (Masquerading) 切换为基于 eBPF 的模式
- Kubernetes NodePort 实现在 DSR(Direct Server Return) 模式下运行
- 绕过 iptables 连接跟踪 (Bypass iptables Connection Tracking)
- 主机路由 (Host Routing) 切换为基于 BPF 的模式 (需要 Linux Kernel >= 5.10)
- 启用 IPv6 BIG TCP (需要 Linux Kernel >= 5.19, 支持的 NICs: mlx4, mlx5)
- 修改 MTU 为巨型帧 (jumbo frames) （需要网络条件允许）
- 启用带宽管理器 (Bandwidth Manager) (需要 Kernel >= 5.1)
- 启用 Pod 的 BBR 拥塞控制 (需要 Kernel >= 5.18)
- 启用 XDP 加速 （需要 支持本地 XDP 驱动程序）
- (高级用户可选)调整 eBPF Map Size
- Linux Kernel 优化和升级: CONFIG_PREEMPT_NONE=y
- 停止 irqbalance，将网卡中断引脚指向特定 CPU

### 配置
```yaml
bandwidthManager.enabled=true
bandwidthManager.bbr=true
```

```shell
helm upgrade cilium cilium/cilium --version 1.13.4 \
  --namespace kube-system \
  --reuse-values \
  --set bandwidthManager.enabled=true 
  --set bandwidthManager.bbr=true
```

```shell
$ kubectl -n kube-system exec ds/cilium -- cilium status | grep BandwidthManager
BandwidthManager:       EDT with BPF [BBR] [eth0]
```