Linux SVA特性分析
=================

-v0.1 2019.9.15 Sherlock init
-v0.2 2019.9.18 Sherlock update数据结构和动态分析
-v0.3 2019.10.13 Sherlock add性能分析
-v0.4 2019.10.14 Sherlock add数据结构说明
-v0.5 2019.10.31 Sherlock add iommu_sva_set_ops, iommu_sva_get_pasid

简介: 最近Linux社区在上传SVA特性(share virtual memory), 简单讲这个特性可以达到
这样的功能，就是可以是设备直接使用进程里的虚拟地址。如果把设备操作映射到用户态，
就可以在用户态叫设备直接使用进程虚拟地址。设备可以使用进程虚拟地址做DMA，如果
设备功能强大，比如可以执行一段指令，设备甚至可以直接执行相应进程虚拟地址上的
指令。目前SVA特性正处于上传阶段，这个特性的补丁涉及到IOMMU, PCI, 内存管理(MM),
SMMU, VFIO, 虚拟化，DT, ACPI等方面的修改，所以补丁比较分散，不过好在这个特性的
上传者把相关补丁收集在了一个分支上，即https://jpbrucker.net/sva/ 的sva/current
分支上。想要了解SVA的整体，可以直接查看这个分支上的各个提交。需要说明是，这个
分支的base是5.2-rc7, 在5.3中这个分支上的少量补丁已经合入Linux主线，如果想用最
新内核测试SVA需要做一定的适配工作。本文的分析以base是5.2-rc7的sva/current分支
上的补丁为基础。

使用场景介绍
------------------

  如上，SVA特性可以做到进程虚拟地址在进程和设备之间共享。最直观的使用场景就是在
  用户态做DMA。
```
             
    +------------------+     +----------------+
    |   process        |     |  process       |       ...
    |                  |     |                |
    | va = malloc();   |     |                |
    | set dma dst: va  |     |                |
    |      +---------+ |     |                |          用户态
    +------+ mmap io +-+     +----------------+
 <---------+         +------------------------------------------> 
           +---------+
                        内核

 <--------------------------------------------------------------> 
                +--------------+
                |  DMA dst: va |
                |              |
                |  device      |
                +--------------+
```
  如上图所示，在SVA的支持下，我们可以在用户态进程malloc一段内存，然后把得到va
  直接配置给设备的DMA，然后启动DMA向va对应的虚拟地址写数据，当然也可以从va对应
  的虚拟地址上往设备读数据。这里我们把设备DMA相关的寄存器先mmap到用户态，这样
  DMA操作在用户态就可以完成。

  可以注意到，SVA可以支持功能很大一部分取决于设备的功能，SVA就是提供一个进程和
  设备一致的虚拟地址空间，其他的设备想怎么用都可以。如上，如果设备做的足够强，
  设备完全可以执行va上对应的代码。

  可以看到，设备完全可以把自身的资源分配给不同的进程同时使用。

  为了满足上面的使用场景，SVA特性需要硬件支持IOMMU以及设备发起缺页。在下一节
  先介绍硬件，再基于此分析SVA的软件实现。

硬件基础介绍
------------------

  本文以ARM64体系结构为基础分析，在ARM64下，IOMMU指的就是SMMU。对于设备，ARM64
  下有平台设备和PCI设备。整体的硬件示意图如下，图中也画出了硬件工作时相关的内存
  里的数据结构。
```
               +-----+
               | CPU |
               +--+--+
                  |             
                  v             
               +-----+           +---------------------------------------------+
               | MMU |-----------+------------------------------------+        |
               +--+--+           | DDR                                |        |
                  |              |                                    |        |
                  v              |                                    |        |
  system bus ------------------> |                                    |        |
                  ^              |                                    v        |
                  |   SID/SSID   |      +-----+     +----+      +------------+ |
                  |       +------+----> | STE |---->| CD |----->| page table | |
                  |       |      |      +-----+     +----+      +------------+ |
                  |       |      |       ...        | CD |----->| page table | |
                  |       |      |                  +----+      +------------+ |
          IRQs    |       |      |                  | .. |      | ..         | |
                ^ |^  ^   |      |                  +----+      +------------+ |
                | ||  |   |      |                  | CD |----->| page table | |
      +---------+-++--+---+---+  |                  +----+      +------------+ |
      | SMMU    |  |  |   |   |  |                                             |
      |         |  |  |       |  |     +-------------+                         |
      |  +------+--+-CMD Q ---+--+---->| CMD queue   |                         |
      +--v--+   |  |          |  |     +-------------+                         |
      | PRI |---> PRI Q ------+--+---->| PRI queue   |                         |
      +-----+   |             |  |     +-------------+                         |
      | ATS |  EVENT Q -------+--+---->| EVENT queue |                         |
      +-----+-----------------+  |     +-------------+                         |
        ^ |     ^            ^   +---------------------------------------------+
        | |     |            |
        | v     | BDF/PASID  +--------------+
      +---------+-------------+             | 
      |         RP            |             | 
      +-----------------------+             | 
        ^ |       ^                         | 
        | v       |  BDF/PASID              | 
      +-+----+----+-+---------+  +-----+----+----------+
      | ATC  |      |         |  |     |               |                       
      +------+      |         |  | DMA |               |                       
      | PRI  |      |  EP     |  |     |               |                       
      +------+ DMA  |         |  +-----+  platform dev |                       
      +-------------+         |  |                     |                       
      +-----------------------+  +---------------------+
```
   基于上一节中提到的使用场景, 我们梳理硬件中的逻辑关系。调用malloc后，其实只是
   拿到了一个虚拟地址，内核并没有为申请的地址空间分配实际的物理内存，直到访问这
   块地址空间时引发缺页，内核在缺页流程里分配实际的物理内存，然后建立虚拟地址到
   物理内存的映射。这个过程需要MMU的参与。设想SVA的场景中，先malloc得到va, 然后
   把这个va传给设备，配置设备DMA去访问该地址空间，这时内核并没有为va分配实际的
   物理内存，所以设备一侧的访问流程必然需要进行类似的缺页请求。支持设备侧缺页
   请求的硬件设备就是上面所示的SMMU，其中对于PCI设备，还需要ATS、PRI硬件特性支持。
   平台设备需要SMMU stall mode支持(使用event queue)。PCI设备和平台设备都需要
   PASID特性的支持。

   如上图所示，引入SVA后，MMU和SMMU使用相同的进程页表, SMMU使用STE表和CD表管理
   当前SMMU下设备的页表，其中STE表用来区分设备，CD表用来区分同一个设备上分配给
   不同进程使用的硬件资源所对应的进程页表。STE表和CD表都需要SMMU驱动预先分配好。

   SMMU内部使用command queue，event queue，pri queue做基本的事件管理。当有相应
   硬件事件发生时，硬件把相应的描述符写入event queue或者pri queue, 然后上报中断。
   软件使用command queue下发相应的命令操作硬件。

   PCI设备和平台设备的硬件缺页流程有一些差异，下面分别介绍。对于PCI设备，ATS, 
   PRI和PASID的概念同时存在于PCIe和SMMU规范中。对于ATS的介绍可以参考这里:
   https://blog.csdn.net/scarecrow_byr/article/details/74276940。简单讲，ATS特性
   由设备侧的ATC和SMMU侧的ATS同时支持，其目的是在设备中缓存va对应的pa，设备随后
   使用pa做内存访问时无需经过SMMU页表转换，可以提高性能。PRI(page request
   interface)也是需要设备和SMMU一起工作，PCIe设备可以发出缺页请求，SMMU硬件在解
   析到缺页请求后可以直接将缺页请求写入PRI queueu, 软件在建立好页表后，可以通过
   CMD queue发送PRI response给PCIe设备。具体的ATS和PRI的实现是硬件相关的，目前
   市面上还没有实现这两个硬件特性的PCIe设备，但是我们可以设想一下ATS和PRI的硬件
   实现，最好的实现应该是软件透明的，也就是软件配置给设备DMA的访问地址是va, 软件
   控制DMA发起后，硬件先发起ATC请求，从SMMU请求该va对应的pa，如果SMMU里已经有va
   到pa的映射，那么设备可以得到pa，然后设备再用pa发起一次内存访问，该访问将直接
   访问对应pa地址，不在SMMU做地址翻译，如果SMMU没有va到pa的映射, 那么设备得到
   这个消息后会继续向SMMU发PRI请求，设备得到从SMMU来的PRI response后发送内存访问
   请求，该请求就可以在SMMU中翻译得到pa, 最终访问到物理内存。

   PRI请求是基于PCIe协议的, 平台设备无法用PRI发起缺页请求。实际上，平台设备是无法
   靠自身发起缺页请求的，SMMU用stall模式支持平台设备的缺页，当一个平台设备的内存
   访问请求到达SMMU后，如果SMMU里没有为va做到pa的映射，硬件会给SMMU的event queue
   里写一个信息，SMMU的event queue中断处理里可以做缺页处理，然后软件给SMMU发送
   RESUME CMD继续stall的请求，如果出错，软件可以通过SMMU CMD回信息给设备。实际上，
   SMMU使用event queue来处理各种错误异常，这里的stall模式是借用了event queue来处
   理缺页。

   可以注意到PRI和stall模式完成缺页的区别是，PRI缺页的时候并不是在IO实际发生的
   时候，因为如果PRI response表示PRI请求失败，硬件完全可以不发起后续的IO操作。
   而stall模式，完全发生在IO请求的途中。所以，他被叫做stall模式。

数据结构
--------------
```
   struct device
           +-> struct iommu_param
                   +-> struct iommu_fault_param
                           +-> handler(struct iommu_fault, void *)
                           +-> faults list
                   +-> struct iopf_device_param
                        +-> struct iopf_queue
                                +-> work queue
                                +-> devices list
                        +-> wait queue
                   +-> struct iommu_sva_param
```
  在引起缺页的外设的device里，需要添加缺页相关的数据结构。handler是缺页要执行
  的函数，具体见下面动态流程分析。iopf_queue在smmu驱动初始化时添加，这里iopf_queue
  可能是eventq对应的，也可能是priq对应的，iopf_queue是smmu里的概念，所以同一个
  iopf_queue会服务同一个smmu下的所有外设, 上面的devices list链表就是用来收集这个
  smmu下的外设。
```
  struct iommu_bond
        +-> struct iommu_sva
                +-> device
                +-> struct iommu_sva_ops
                        +-> iommu_mmu_exit_handle ?
        +-> struct io_mm
                +-> pasid
                +-> device list
                +-> mm
                +-> mmu_notifier
                +-> iommu_ops
                        +-> attach
                        +-> dettach
                        +-> invalidat
                        +-> release
        +-> mm list
        +-> device list
        +-> wait queue
```
  下面的图引用自JPB的补丁，该图描述的是建立好的静态数据结构之间的关系，以及IOMMU
  (e.g. SMMU的STE和CD表在这种数据结构下的具体配置情况)表格的配置。用这个图可以
  很好说明iommu_bond中的各个数据结构的意义以及之间的关系。

  iommu_bond这个结构并不对外, 用下面的iommu_sva_bind_device/unbind接口时，函数
  参数都是iommu_sva。使用SVA的设备可以把一个设备的一些资源和一个进程地址空间绑定，
  这种绑定关系是灵活的，比如可以一个设备上的不同资源和不用的进程地址空间绑定
  (bond 1, bond 2), 还可以同一个设备上的资源都绑定在一个进程的地址空间上(bond 3,
  bond 4)。从进程地址空间的角度看，一个进程地址空间可能和多个设备资源绑定。

  iommu_bond指的就是一个绑定，io_mm指的是绑定了外设资源的一个进程地址空间。
  io_pgtables是指内核dma接口申请内存的页表。
```
              ___________________________
             |  IOMMU domain A           |
             |  ________________         |
             | |  IOMMU group   |        +------- io_pgtables
             | |                |        |
             | |   dev 00:00.0 ----+------- bond 1 --- io_mm X
             | |________________|   \    |
             |                       '----- bond 2 ---.
             |___________________________|             \
              ___________________________               \
             |  IOMMU domain B           |             io_mm Y
             |  ________________         |             / /
             | |  IOMMU group   |        |            / /
             | |                |        |           / /
             | |   dev 00:01.0 ------------ bond 3 -' /
             | |   dev 00:01.1 ------------ bond 4 --'
             | |________________|        |
             |                           +------- io_pgtables
             |___________________________|

                                PASID tables
                                 of domain A
                              .->+--------+
                             / 0 |        |-------> io_pgtable
                            /    +--------+
            Device tables  /   1 |        |-------> pgd X
              +--------+  /      +--------+
      00:00.0 |      A |-'     2 |        |--.
              +--------+         +--------+   \
              :        :       3 |        |    \
              +--------+         +--------+     --> pgd Y
      00:01.0 |      B |--.                    /
              +--------+   \                  |
      00:01.1 |      B |----+   PASID tables  |
              +--------+     \   of domain B  |
                              '->+--------+   |
                               0 |        |-- | --> io_pgtable
                                 +--------+   |
                               1 |        |   |
                                 +--------+   |
                               2 |        |---'
                                 +--------+
                               3 |        |
                                 +--------+
```

  相关接口:
```
    - iommu_dev_enable_feature: 准备和sva相关的smmu中的管理结构, 该接口可以在
                                    设备驱动里调用，用来使能sva的功能
      +-> arm_smmu_dev_enable_feature
          +-> arm_smmu_dev_enable_sva
            +-> iommu_sva_enable
              +-> iommu_register_device_fault_handler(dev, iommu_queue_iopf, dev)
                  /* 动态部分将执行iommu_queue_iopf */
                  把iommu_queue_iopf赋值给iommu_fault_param里的handler                  
            +-> iopf_queue_add_device(struct iopf_queue, dev)
                把相应的iopf_queue赋值给iopf_device_param里的iopf_queue, 这里有
                pri对应的iopf_queue或者是stall mode对应的iopf_queue。初始化
                iopf_device_param里的wait queue

                对应iopf_queue的初始化在在smmu驱动probe流程中: e.g.
                arm_smmu_init_queues
                  +-> smmu->prq.iopf = iopf_queue_alloc
                                           +-> alloc_workqueue
                                             分配以及初始化iopf_queue里的工作队列
            +-> arm_smmu_enable_pri
                调用PCI函数是能EP设备的PRI功能

    - iommu_sva_bind_device: 将设备和mm绑定, 该接口可以在设备驱动里调用，把一个
                             设备和mm绑定在一起。返回struct iommu_sva *
      +-> iommu_sva_bind_group
        +-> iommu_group_do_bind_dev
          +-> arm_smmu_sva_bind
            +-> arm_smmu_alloc_shared_cd(mm)
                分配相应的CD表项，并且把CD表项里的页表地址指向mm里保存的进程页表
                地址。这个函数主要配置SMMU的硬件
            +-> iommu_sva_bind_generic(dev, mm, cd, &arm_smmu_mm_ops, drvdata)
              +-> io_mm_alloc
                  分配io_mm以及初始化其中的数据域段, 向mm注册io_mm的notifier
                  /* to do: mm发生变化的时候通知io_mm */
                  +-> mmu_notifier_register
              +-> io_mm_attach
                +-> init_waitqueue_head
                    初始化iommu_bond里的等待队列mm_exit_wq /* to do: 作用 */
                +-> io_mm->ops->attach(bond->sva.dev, io_mm->pasid, io_mm->ctx)
                    调用e.g.SMMU arm_smmu_mm_ops里的attach函数
                    +-> arm_smmu_mm_attach
                      +-> __arm_smmu_write_ctx_desc
                        下发SMMU command使能CD配置。可以看到arm_smmu_mm_ops里的
                        这一组回调函数基本都是下发SMMU命令控制CD/ATC/TLB相关的
                        配置

    - iommu_sva_unbind_device
        +-> iopf_queue_flush_dev
        +-> iommu_unbind_locked(to_iommu_bond(handle))
        这里的handle是一个struct iommu_sva
```
    - iommu_sva_set_ops(iommu_sva, iommu_sva_ops)

      这个接口把iommu_sva_ops传递给iommu_sva, iommu_sva_ops包含mm_exit回调。
      在上面的iommu_sva_bind_generic里会调用mmu_notifier_register给当前的mm里
      注册一个notifier，在mm发生改变的时候，mm子系统可以触发notifier里的回调
      函数。当前的代码里，是在notifier的release回调里去调用iommu_sva里保存的
      mm_exit回调函数。

      SVA特性使得设备可以看到进程虚拟地址空间，这样在进程虚拟地址空间销毁的时候
      应该调用设备驱动提供的函数停止设备继续访问进程虚拟地址空间。这里iommu_sva_set_ops
      就是把设备驱动的回调函数注册给进程mm。

      注意上面的mmu_notifier_register注册的iommu_mmu_notifier_ops回调里。release
      只在进程异常时调用到，用户态进程正常退出时并不会调用。在进程正常退出时，
      如何保证设备停止访问将要释放的进程地址空间，这里还有疑问。

      进程退出的调用链是:
      kernel/exit.c:
      do_exit
        +-> exit_mm
	  +-> mmput
	    +-> exit_mmap
	      +-> mmu_notifier_release

    - iommu_sva_get_pasid

      这个接口返回返回smmu_sva对应的pasid数值，设备驱动需要把pasid配置给与这个
      smmu_sva相关的硬件资源。

  需要使用SVA特性的社区驱动在调用上面的接口后，可以建立起静态的数据结构。

动态分析
--------------

  - 缺页流程

    当一个PRI或者是一个stall event上报后, 软件会在缺页流程里建立页表，然后控制
    SMMU给设备返送reponse信息。我们可以从SMMU PRI queue或者是event queue的中断
    处理流程入手跟踪: e.g.PRI中断流程
```
    devm_request_threaded_irq(..., arm_smmu_priq_thread, ...)
    arm_smmu_priq_thread
      +-> arm_smmu_handle_ppr
        +-> iommu_report_device_fault
          +-> iommu_fault_param->handler
            +-> iommu_queue_iopf /* 初始化参见上面第2部分 */
              +-> iopf_group = kzalloc
              +-> list_add(faults list in group, fault)
              +-> INIT_WORK(&group->work, iopf_handle_group)
              +-> queue_work(iopf_param->queue->wq, &group->work)
              这段代码创建缺页的group，并把当前的缺页请求挂入group里的链表，然后
              创建一个任务，并调度这个任务运行

              在工作队列线程中:
              +-> iopf_handle_group
                +-> iopf_handle_single
                  +-> handle_mm_fault
                      这里会最终申请内存并建立页表

        +-> arm_smmu_page_response
            软件执行完缺页流程后，软件控制SMMU向设备回响应。
```
  - Invalid流程

    当软件释放申请的内存时，SMMU中关于这些内存的tlb以及设备ATC里都要Invalid。
    进程mm变动的时候，调用注册的io_mm notifier完成相关的tlb、atc的invalid。

    (to do: ...)

性能分析
--------------

  以UACCE(https://lkml.org/lkml/2019/10/9/634)和ZIP压缩解压缩引擎为测试case，
  在SVA情况下。测试结果为:

  1. 因为使用SVA可以做到对用户态malloc的内存直接访问，ZIP性能得到很大提升。
     可以从原来的1900MB/s提升到5000+MB/s

  2. 从测试结果来看, 大量的SMMU tlb miss不会造成系统新能下降。或者说在把系统里
     原来的内存拷贝去掉后, 换成SVA，性能上的增加远大于smmu tlb miss的开销。
     测试中使用的smmu的L1 tlb entry个数是512， L2 tlb entry个数是2048，tlb粒度是
     4KB, 我们假设就算这个smmu的所有tlb都给ZIP用(L2 tlb是这个smmu下面所接外设共享
     的), 那么所有tlb所覆盖的内存范围也就512 × 4K + 2014 × 4K = 10MB。测试中，
     我们使用100M的测试数据，每次压缩4K数据，直到把这100M的数据都压缩完。可以看
     到就算前一个10M的数据都可以tlb命中，那么在下一个10M的数据处理中，每个4K开始
     总会有一个tlb不命中。(该外设的burst大小是128 Byte, 所以在处理一个4K数据时，
     最多只有第一次128 Byte的读写会有tlb miss，后续的tlb都是hit的)

  3. 如果在IO流程中有设备缺页，性能会有很大的下降，至少会下降到没有缺页流程的
     1/6。在测试中，对于压缩的源数据，在准备数据的时候会把数据写到内存中，这个
     动作会引发CPU侧的系统缺页，为了把目的数据内存也提前在CPU侧缺页，我们先用
     memset把目的数据内存都写0。实际测试的时候可以发现，对于一次测试(如上所说的，
     把100MB数据以4K大小为单位压缩完), 在第一次测试的时候性能总是很差(执行时间
     很长), 如果我们，再以同样的数据多测试几次，并且统计每一次执行的时间，会发现
     除了第一次外，以后每次的执行时间都很短，这样每次的性能都很好。

     一开始，怀疑是smmu的tlb命中率有关，因为根据上面的分析，页表应该在开始测试
     之前就在CPU侧建立好了。但是，根据上面的分析可以知道，即使在一次测试里smmu
     的tlb也会大量的刷新，每次测试的情况也是一样的，不会出现后面测试的tlb miss
     会减少。实际中smmu pmcg工具测试的结果也表示，第一次测试中的tlb hit和后面测试
     的tlb hit(平均值)没有差异，但是第一次的性能依然很差。
     (smmu pmcg的用法可以具体参考: https://blog.csdn.net/scarecrow_byr/article/details/101673228)

     于是怀疑还是缺页的问题，在smmu缺页处理里加了打印，发现第一次测试的时候，每
     次都会执行到smmu缺页处理里。怀疑是上面目的地址使用memset的方式无法引起缺页，
     在memset之前，把目的地址随便写入一些值，然后再用memset清0，再次测试发现
     第一次测试是不会有设备层缺页了，第一次测试的性能也和后面的一样。

虚拟化
------------

 https://blog.csdn.net/scarecrow_byr/article/details/104606571
