# Packet Scheduling

PS目标是,有一系列input的packets, 如何进行重组然后进行dequeue
核心：enqueue和dequeue的操作，中间需要维护的数据结构
目标：吞吐量(低成本下实现高吞吐，effective)、公平性(efficient)；例如: stronger delay guarantees, burst-friendly fairness, starvation-free priorization of short flows
使用的工具：纯软件、硬件加速

## Project归纳

工具: NetBench, NS-3/NS-2 (流量模型可以参照pFarbic等)
方向:

1. PIFO本质是order, virtual clock可以实现work-conserving, 是否有可能实现混用: 把order变到virtual-clock里, 解决PIFO的work-conserving缺陷? **本质上是Logical Clock和Real Clock之间的区别?** 需要深入阅读以下Logical Clock的知识和Lamport的TLA语言设计
2. 层级性的调度如何解决?

针对1的一种设计是: 前面使用PIFO，后面接着使用TimeWheel数据结构(使用list实现便于插入操作); 从而实现order和time两种要求。

1. 可以使用两个TW, 一个用来维护PIFO结构, 一个用来维护on-time的结构。对于on-time的queue, 时间性的要求更强, 需要更快地发送出去。带来的优势是，对于on-time的pkt良好, 但会存在对于PIFO类型的报文(见缝插针地进行调度)的饥饿问题。这里就可以和Loom的设计比较性能差异(因为Loom需要对pkt进行re-order操作)
2. 此外, 设计的时候不应该per-packet进行，而是per-flow地进行。每到处理1个flow的时候, 就从对应flow中进行dequeue操作(像Loom这样的设计可能带来flow内的pkts乱序问题: 需要实验验证乱序问题带来的较大的影响)

| 项目 | 会议 | 算法说明 | 实现方式 | 使用工具 | 优势 | 劣势 | 可能的后续方向 |

### 针对SP-PIFO的改进

因为SP-PIFO本质是把原来的order映射到更小的range的order(这个映射是动态调整的), 其实我们可以在此基础上增加固定的highest order来降低reversation? 也可以通过让**每个queue通过的pkt数量近似相等来实现最小的total reverse**?

## 文献阅读

文献阅读放在了OneNote: 网络研究/流量控制/已有项目代码下，主要阅读的paper有

### SP-PIFO

- 会议: NSDI'20
- 算法说明: 使用多FIFO队列来模拟PIFO
- 实现方式: 软件-Java(基于NetBench)来比较schedule schemes, 对比FIFO、gradient-based algorithm(rank排序效果、inversion的数量), 进而分析出SP-PIFO的设计空间; 硬件-P4(基于MiniNet)来进行实际实验(200LOC-$P4_{16}$),对比了LSTF、STFQ、FIFO+, 目标是: 最小化FCTs和提高fairness。为了比较FCT, 使用的traffic model是pFabric web application和data mining(流到达是遵循Poisson分布), 其中pFabric是基于remaining flow sizes来最小化FCT(要使用PIFO或SP-PIFO), 同时比较了传统TCP和DCTCP算法。 为了比较fairness across flows, 在PIFO/SP-PIFO之上实现STFQ(Start-Time Fair Queueing), 进而分析不同flow sizes和queues数量下的performance, 比对的基线是AFQ。
- 使用工具: Moongen(流量产生)、pFabric、P4等
- 优点: 减少PIFO开销, 其他的各项实验本质上是PIFO的好处(比如pFabric减小FCT等)
- 缺点: 继承了PIFO的缺点(不能rate-limit egress throughput,进而不能实现non-work-conserving scheduling algorithm; 不能直接实现hierarchical scheduling; 但是可以通过多个PIFO模拟hierarchical)。 SP-PIFO可能存在queue数量和accuracy之间的tradeoff(目前switch支持32 queues/port)。 可能存在对抗流量作为attack
- 可能的后续方向: 多个PIFO模拟hierarchical, 在PIFO文章中有提到可以通过recirculate packet(多次access queues)来实现, 但是如何减少对性能的影响未能解决。Hierarchical的核心问题在于，插入一个新报文后，所有pkt的顺序会发生变化（而且存在不定的变化结果！）；需要研究hierarchical结构对原始的pkt queue的影响(在CalQueue文章中也提到了这一点: PIFO使得pkt在enqueue之后relative order不能变化, 使得PIFO不能实现pFabric的starvation prevention technique)
- TODO: 研究**LSTF**、STFQ、FIFO+、AFQ、FDPA、PIAS、EDF, PIFO不能实现non-work-conserving scheduling?

### CalQueue

- 会议: NSDI'20
- 算法说明: 解决PIFO的动态性问题(需要动态调节priority)
- 针对问题: 很多schedule算法(如WFQ)翻译成PIFO的order时会发现映射成rank出现"infinite"
- 实现方式: 使用Physical Calendar Queue(Queue向前移动的时候要根据时间前进)可以implement work-conserving schemes(如EDF)和non-work(如Leaky Bucket Filter、Jitter-EDD、Stop-and-Go)，而Logical Calendar Queue对应work-conserving(LSTF、WFQ、SRPT)。文章里说明了如何用CQ来实现WFQ、EDF和LBF。
- 使用工具: CalQueue结构；使用3个case study验证性能(deadline-aware的co-flow/flow调度、FQ的variant--burstiness容忍进而可以在fairness和FCT之间平衡、pFabric的变种--通过逐渐增加all入队pkt的priority可以防止长流starve)。流量模型复用了pFabric(Sigcomm'13)的data mining workload。使用mptcp-htsim simulator来仿真pkt执行情况。
- 优点: 使用多个队列CalQueue解决单个queue不足的问题, 和MLFQ有些类似。可以理解成Carousel和PIFO的组合。
- 缺点: 存在feasiblity vs. accuracy的tradeoff(单个queue内存在inversion); 显著存在的问题是单次enqueue的range受限(不能任意插入某个rank的pkt----使用ACL切换的方法可以解决这个问题)
- 可能的后续方向: 现代schedule, 粗粒度 queue-level priority, 或者细粒度packet-level(PIFO)。文章里有提到，可以再使用SP-PIFO来reduce inversion; 处理bucket不足可以使用一个单独的queue然后在合适的时候(get close to their service time)做recirculate, 也可以使用hierarchical结构来增加queue数量。可以使用ACL规则的排列插入，来实现更好的更新算法！(一个queue就可以实现多bucket的情况, 但是算法复杂度仍为log(N))。 **PIFO本质是order, virtual clock可以实现work-conserving, 是否有可能实现混用: 把order变到virtual-clock里, 解决PIFO的work-conserving缺陷?** **此外, hierarchical问题解决?** **SP-PIFO中的theoretical差别: 充分模拟PIFO和性能之间(多个queue)的trade-off** **SP-PIFO存在对抗流量: SP-PIFO本质是对rank distribution的模拟, 存在对抗性流量** **Facilitate(促进) PIFO，由于硬件改进带来的可以在enqueue时更精准地预测unpifoness**
- TODO: 研究**LSTF(Least Slack Time First)**、STFQ、FIFO+、AFQ、FDPA、PIAS、EDF， Calendar Queues

### Loom

- 会议: NSDI'19
- 算法说明: 1. 使用DAG表示Policy abstraction; 2. Programmable Hierarchical PS; 3. OS interface
- 针对问题: 纯software的PS开销过大, 使用NIC加速有imperfect的问题; 需要定义NIC PS和OS/NIC interface。其中**DAG Policy Abstraction**是重点内容, 定义了两种node: scheduling(order)和shaping(rate-limit)；要求: 把shaping去掉后scheduling能组成tree；shaping node可能是并行的shaping nodes的nested set(嵌套集合: 类似俄罗斯套娃、类似线段树结构)；node的parent是scheduling, 那么它只能有一个parent，如果是shaping、则可以有多个parents。在进入DAG结构前, 所有的metadata都要准备好。
设计新的NIC来把schedule从OS放到NIC上(支持hierarchical)，
- 实现方式: 设计新的scheduling hierarchy, 新的OS/NIC interface。 shaping的时候可以
- 工具: 设计基于BESS和Domino。Domino: 现有的scheduling tree的compiler; prototype使用BESS(通过修改BESS kernel driver来实现OS/NIC的Loom的interface); 此外Loom还修改了PIFO的C++实现。使用iperf3(测量throughput)和sockperf(测量端到端latency)，使用Spark with the TeraSort benchmark来perform a 25GB shufffle, 使用50ms的窗口来计算throughput。使用CloudLab(cloudlab.us)来进行send data between two servers.
- 优点:
- 缺点:
- 可能的后续方向: policy有两种, work conserving的scheduling(确定pkt的relative order)和rate-limiting的shaping(确定pkt的time)。 network-wide的policy across a cluster of servers是一个方向。**Network层面的policy尚不清楚怎么处理(目前的policies都集中在单个node上)。** **如果DAG过大，或者算法过于复杂，就会拒绝策略部署到NIC上: 因此,这里如何进行扁平化操作显得非常重要!**
-TODO: 研究NSDI'11的Sharing the data center network(50); NSDI'13 EyeQ(30); Sigcomm'11 Chatty tenants and the cloud network sharing problem (10); Sigcomm'15 BwE: Flexible, hierarchical bandwidth allocation for WAN distributed computing(32); Sigcomm'15 Silo: Predictable message latency in the Cloud(29); Sigcomm'12 FairCLoud: Sharing the network in cloud computing (41); Sigcomm'07 Cloud control with distributed rate limiting (44).
PSPAT:Software Packet scheduling at hardware speed(46)。

### Eiffel

- 会议: NDSI'19
- 算法说明: 观察到packet rank基本有一个specific range, 多数packets有同样的rank, 这使得bucket-based priority queue非常有效。 定义bitmap-hierarchical结构(利用FindFirstSet新指令来加速)
- 针对问题: 网络需要10k数量级的rate limiter而网卡只支持10-128queues(在Introduction里的介绍); 多平台支持等需要我们使用Software来实现PS
- 实现方式: Qdisc(Linux Kernel中进行实现), https://github.com/saeed/eiffel_linux/tree/working_ffs-based_qdisc
- TODO: OpenQueue(39)、hClock(19)
- PIEO:
- Carousel:
- PIFO:
- AFQ:

### OpenQueue

- 会议: ICNP'17

### hClock

- 会议: EuroSys'13
这篇文章非常重要, 里面详细阐述了Hierarchical PS的相关设计思想和说明!
定义了tag和Clock(Reservation, Limit, Shared)来控制层级性的scheduler的调度: 有点像把HTB变成clock控制

- CBQ算法
- Virtual Clock说明
- HFSC算法
- HTB算法

### Universal Packet Scheduling

- 会议: NSDI'16
- 目标: UPS的定义在于, 理论上可以replay any schedule, 实际上是可以achieve不同的performance objectives(fairness, tail latency, FCT等, 而且是能够replay已知的最好的scheduling算法)。

## 文件夹说明

### ./code

文件夹下是已有项目代码
./code/sp-pifo: NSDI'20-SPPIFO项目源码

## Software Packet Scheduling

Fair Queueing
Carousel
Eiffel

在pFabric上做test

## Hardware Packet Scheduling

FPGA
SmartNIC
P4
