 

<center><h1>Homa算法开发记录</h1></center>



## 1  发送方逻辑

按照窗口大小发送携带标记的 `udp` 数据包，需要修改 `bps` 速率控制发送。



### 1.1  窗口相关

- 在 `main` 函数中有相应窗口的设置：
  - `clientHelper` 对象初始化的时候，有设定 `win`，此时值为 `maxbdp`，最大带宽时延积。





## 代码阅读

### SetAttribute

在 `ns3` 中，一般会使用类似的语句，创建对象工厂 `ObjectFactory`，用字符串的方式设置属性和对应的值。

```cpp
// 这时是将"DataRate"和字符串值"100Gbps"暂存到qbb工厂的配置列表中
qbb.SetDeviceAttribute("DataRate", StringValue("100Gbps"));

// 后续的操作如下，这是才是真正创建一个d来具体将配置创建为真正的对象
NetDeviceContainer d = qbb.Install(snode, dnode);

// 在Install方法中：
Ptr<QbbNetDevice> devA = m_deviceFactory.Create<QbbNetDevice>();
```



还有例如 `clientHelper`，也是类似的

```cpp
RdmaClientHelper clientHelper(
    pg, serverAddress[src], serverAddress[dst], sport, dport, target_len,
    has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
    global_t == 1 ? maxRtt : pairRtt[n.Get(src)][n.Get(dst)]);
clientHelper.SetAttribute("StatFlowID", IntegerValue(flow_input.idx)); // 设置了9个变量，进入工厂

// 这个时候真正install，创建app实例，这个start时间和stop时间与Simulator::Now()是同步的
ApplicationContainer appCon = clientHelper.Install(n.Get(src));  // SRC
appCon.Start(Seconds(Time(0)));
appCon.Stop(Seconds(100.0));
```



### topology_file

默认的 `topology_file` 是 `leaf_spine_128_100G_OS2.txt`，存储在 `config` 文件夹下。

```cpp
// 由 topof 打开
// txt文件内容

// 第一行分别是 节点数node_num 交换机数switch_num 链路数link_num
144 16 192
     
// 第二行分别是 switch_num 个 交换机号
128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143

// 接下来的每一行（共计192行） 5个变量分别是
// 源src 目的dst 速率data_rate 链路延迟link_delay 错误速率error_rate
0 128 100Gbps 1000ns 0
1 128 100Gbps 1000ns 0
2 128 100Gbps 1000ns 0
......
```



### flow_file与从flow的执行流程

`flow_file` 存储在 `config` 文件夹下。

```cpp
// 第一行是流的数量
3
// 后续的每一行 5个变量分别是
// src dst pg maxPacketCount start_time
77 0 3 4287 2.000000003
102 42 3 5328 2.000000237
94 91 3 1956 2.000000438
```



流的读入在代码上也是有安排的：

```cpp
// 1. 首先在前面读入了 flow_num，这里首次调用 ReadFlowInput 函数。
if (flow_num > 0) {
    ReadFlowInput();
    // 2. 把“在 0+0 秒的时候调度 ScheduleFlowInputs 函数”这个任务加入调度队列中
    Simulator::Schedule(Seconds(0), &ScheduleFlowInputs, flow_input_stream);
}

// 3. 把 0+2.0 秒的事件加入到调度队列中
// 这个事件是一个递归事件，每 100us 检查一次是否执行完毕，不断插入
Simulator::Schedule(Seconds(flowgen_start_time),&stop_simulation_middle);

// 4. 把 0+12.5 秒加入调度队列，到这个时候，强制停止，不再继续模拟。
Simulator::Stop(Seconds(flowgen_stop_time + 10.0));

// 5. 开始从 0 秒模拟
Simulator::Run();

// 6. 先执行 0 秒的 ScheduleFlowInputs 函数
// 一上来，ScheduleFlowInputs 函数的 while 循环不成立，所以直接执行下面的 if 条件
// 第一个流的启动时间是 2.000000003，在 0+2.000000003 秒调度 ScheduleFlowInputs 函数
if (flow_input.idx < flow_num) 
    Simulator::Schedule(Seconds(flow_input.start_time) - Simulator::Now(), &ScheduleFlowInputs, infile);

// 7. 这个时候调度 while 循环就进来了
while (flow_input.idx < flow_num && Seconds(flow_input.start_time) == Simulator::Now())
...
// 8. while循环有执行ReadFlowInput函数
ReadFlowInput();

// 9. 继续递进到 2.000000237，执行 ScheduleFlowInputs 函数
if (flow_input.idx < flow_num) 
    Simulator::Schedule(Seconds(flow_input.start_time) - Simulator::Now(), &ScheduleFlowInputs, infile);

// 10. 接下来执行其他即可

```



借此机会说说以后的执行过程：

```cpp
// 1. 工厂 install 成了一个appCon
ApplicationContainer appCon = clientHelper.Install(n.Get(src));

// 2. 调用Start方法，这里创建了一个指向Application的指针
appCon.Start(Seconds(Time(0)));

// 3.Application 中定义了如下虚函数
virtual void StartApplication (void);

// 4. 而 RdmaClient 是 Application 的派生类
class RdmaClient : public Application

// 5. RdmaClient 类中有 StartApplication 的虚函数的声明以及实现
virtual void StartApplication (void);

// 6. rdma->AddQueuePair方法开始创建qp流程
void RdmaClient::StartApplication (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  // get RDMA driver and add up queue pair
  Ptr<Node> node = GetNode();
  Ptr<RdmaDriver> rdma = node->GetObject<RdmaDriver>();
  rdma->AddQueuePair(m_size, m_pg, m_sip, m_dip, m_sport, m_dport, m_win, m_baseRtt, m_flow_id);
}

// 7. 调用 RdmaHw::AddQueuePair 方法开始进入 conweave 实现的流程
void RdmaHw::AddQueuePair(uint64_t size, uint16_t pg, Ipv4Address sip, Ipv4Address dip,
                          uint16_t sport, uint16_t dport, uint32_t win, uint64_t baseRtt,
                          int32_t flow_id) {
    // create qp
    Ptr<RdmaQueuePair> qp = CreateObject<RdmaQueuePair>(pg, sip, dip, sport, dport);
    qp->SetSize(size); // size就是传输数据的字节长度
    qp->SetWin(win);
    qp->SetBaseRtt(baseRtt);
    qp->SetVarWin(m_var_win);
    qp->SetFlowId(flow_id);
    qp->SetTimeout(m_waitAckTimeout);

    if (m_irn) {
        qp->irn.m_enabled = m_irn;
        qp->irn.m_bdp = m_irn_bdp;
        qp->irn.m_rtoLow = m_irn_rtoLow;
        qp->irn.m_rtoHigh = m_irn_rtoHigh;
    }

    // add qp
    uint32_t nic_idx = GetNicIdxOfQp(qp); // 通过哈希选中一个网卡
    m_nic[nic_idx].qpGrp->AddQp(qp);
    uint64_t key = GetQpKey(dip.Get(), sport, dport, pg);
    m_qpMap[key] = qp; // qp是一个指针

    // set init variables
    DataRate m_bps = m_nic[nic_idx].dev->GetDataRate();
    qp->m_rate = m_bps;
    qp->m_max_rate = m_bps;
    if (m_cc_mode == 1) {
        qp->mlx.m_targetRate = m_bps;
    } else if (m_cc_mode == 3) {
        qp->hp.m_curRate = m_bps;
        if (m_multipleRate) {
            for (uint32_t i = 0; i < IntHeader::maxHop; i++) qp->hp.hopState[i].Rc = m_bps;
        }
    } else if (m_cc_mode == 7) {
        qp->tmly.m_curRate = m_bps;
    } 
    
    // Notify Nic
    m_nic[nic_idx].dev->NewQp(qp); // 真正通知网卡设备，添加QP
}

```



`ReadFlowInput` 函数是向全局变量 `flowf` 中读入上述的5个变量

```cpp
void ReadFlowInput() {
    if (flow_input.idx < flow_num) {
        flowf >> flow_input.src >> flow_input.dst >> flow_input.pg >> flow_input.maxPacketCount >>
            flow_input.start_time;
        assert(n.Get(flow_input.src)->GetNodeType() == 0 &&
               n.Get(flow_input.dst)->GetNodeType() == 0);
    } else {
        std::cout << "*** input flow is over the prefixed number -- flow number : " << flow_num
                  << std::endl;
        std::cout << "*** flow_input.idx : " << flow_input.idx << std::endl;
        std::cout << "*** THIS IS THE LAST FLOW TO SEND :) " << std::endl;
    }
}
```



### Schedule函数

`schedule` 函数的执行情况：

```cpp
// 当程序执行到这儿时，只是将事件加入队列，实际执行发生在Simulator::Run()开始之后。
Simulator::Schedule(Seconds(0), &ScheduleFlowInputs, flow_input_stream);
```

注意：schedule函数是在当前模拟时间的基础上加上一段时间执行，这里是 `0` 秒。 



`schedule` 函数有非常多的模板，支持传入不同个数参数。

```cpp
// 1个参数的Schedule
template <typename U1, typename T1>
EventId Simulator::Schedule (Time const &time, void (*f)(U1), T1 a1)
{
  return DoSchedule (time, MakeEvent (f, a1));
}

// 2个参数的Schedule
template <typename U1, typename U2, 
          typename T1, typename T2>
EventId Simulator::Schedule (Time const &time, void (*f)(U1,U2), T1 a1, T2 a2)
{
  return DoSchedule (time, MakeEvent (f, a1, a2));
}

// 等等等
```



### BPS相关

目前还是没看懂到底怎么传过去的，但是知道 `m_bps` 就是 `100G`。

