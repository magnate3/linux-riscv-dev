

# qos

```
root@ubuntux86:# g++ -O3 -std=c++11 main2.cpp -o vct_fairness_chart -lpthread -I ./concurrentqueue/
root@ubuntux86:# ./vct_fairness_chart 
========================================================
开始测试：基于开源无锁队列的虚拟时间轮调度 (适配 4MB KVCache)
并发 Worker 数: 4, 总任务量: 8000000
========================================================
测试完成！
总耗时: 0.626726 秒
成功处理任务总量: 8000000 个
吞吐量 (Throughput): 12.7648 M ops/sec (百万次/秒)
单次虚拟时间映射与调度平均延迟: 78.3407 ns (纳秒)
========================================================
root@ubuntux86:# 
```
Jain's Fairness Index    
```
g++ -O3 -std=c++11 main3.cpp -o vct_fairness_chart -lpthread -I ./concurrentqueue/
./vct_fairness_chart 
========================================================
实时监控：单机多线程 Jain's Fairness 公平性曲线抓取测试
并发 Worker 数: 4, 总任务量: 16000000
========================================================
[时间步] 吞吐:   15899K | Jain's Fairness: 1.0000 [=============================>]

测试完成结算报告：
总耗时: 1.3176 秒
吞吐量 (Throughput): 12.1429 M ops/sec
各数据流最终分发计数明细：
  -> Flow [0]: 1600000 个包
  -> Flow [1]: 1600000 个包
  -> Flow [2]: 1600000 个包
  -> Flow [3]: 1600000 个包
  -> Flow [4]: 1600000 个包
  -> Flow [5]: 1600000 个包
  -> Flow [6]: 1600000 个包
  -> Flow [7]: 1600000 个包
  -> Flow [8]: 1600000 个包
  -> Flow [9]: 1600000 个包
终审 Jain's Fairness Index = 1.0000 (完美公平为 1.0)
========================================================
```

# 多路平衡树


传统的 Linux CFS 调度器使用的是二叉平衡树（红黑树）来维护虚拟时间。但在面对超大规模并发（如数据中心数万个流、或是硬核网络芯片线速转发）时，二叉树面临严重挑战：    

•  树高限制速度    

硬件不友好：在芯片（ASIC/FPGA）或多核 CPU 缓存中，高扇出的多路树（如 4 路、8 路、16 路 B+ 树或 PIFO 树）更加扁平。        

• 缓存命中率高：多路树一个节点包含多个键值，可以完美填满一个 CPU Cache Line（缓存行），大幅减少内存时延。   

• Linux 6.6 之后引入的 EEVDF（Earliest Eligible Virtual Deadline First）算法引入了延迟与带宽解耦的概念。
   


# HPCC


+ mix/config.txt 做出以下两处微调：         
1. ACK_HIGH_PRIO 0 改为 ACK_HIGH_PRIO 1：      
	• 原因：在原版的 HPCC 仿真中，如果该项为 0，高优先级队列（即你刚才修改的 EnqueueHighPrioQ 和 qIndex == -1 分支）可能不会被激活或者不处理普通数据包。将其设置为 1，可以强制仿真网卡（QbbNetDevice）将所有控制信号、ACK 包以及我们期望插队的老鼠流全部送入你编写的虚拟时间轮进行 Rank 排序。    
2. BUFFER_SIZE 32（保持默认或根据需要调大）：        
	• 原因：目前的缓冲区大小设置是符合标准 RDMA 交换机配比的。由于 4MB 大块极易占满缓存并触发 PFC 拥塞控制（配置文件第一行的 USE_DYNAMIC_PFC_THRESHOLD 1 说明开启了动态 PFC），我们的虚拟时间轮可以在缓冲区饱和前，微秒级地优先吐出老鼠流，这极利于你在结果中观察到时延的巨大改善。  

## 编译

```
git clone  https://github.com/alibaba-edu/High-Precision-Congestion-Control.git
cd simulation/
./waf clean
CC='gcc-5' CXX='g++-5' ./waf configure  -d optimized 
./waf build
```	
	
##  mix/topology.txt

第一行 66 1 65 蕴含了极为关键的信息：    
• 66：代表该网络拓扑中的总节点数（Total Nodes = 66）。    
• 1：代表网络中的交换机总数（Switches = 1）。在这个经典的单星型/大胖树单核心拓扑中，节点 0 就是那台核心交换机。     
• 65：代表该拓扑中可用的主机/服务器总数（Hosts = 65）。
紧接着的连线配置（如 0 1 100Gbps 0.001ms 0）代表核心交换机 0 与其余 65 台服务器（节点 ID 为 1 到 65）一一相连。    
• 主机的合法合法 ID 区间：1 到 65。    
• 交换机的 ID：0。     
	
## 产生混合场景流量(10% 是 4MB 块，90% 是 4KB 小包)

+ 指标   

老鼠流的 P99/P99.9 尾部时延    
观察大象流是否吃满网卡 Goodput   

AliStorage2019.txt（阿里真实存储业务分布）。该分布天生具备“少量大象流贡献 90% 吞吐，大量老鼠流贡献 90% 连接”的帕累托特性 [AliStorage2019.txt]     
根据topology.txt中host

```
python3 traffic_gen.py -n 65 -c AliStorage2019.txt -l 0.6 -t 4 -o ../simulation/mix/flow.txt
```

## 加速跑



+ mix/config.txt，将 ENABLE_TRACE 1 改为 ENABLE_TRACE 0;命令行启动时，去掉 --trace flow 参数  


+ mix/config.txt，将这一行 SIMULATOR_STOP_TIME 4 改为 SIMULATOR_STOP_TIME 2.5：

(注：由于流量生成脚本是从 2.0 秒开始发包的，所以改成 2.5 说明在 100G 高压下全力倾泻 0.5 秒，足够生成大量 FCT 数据。)





```
 python run.py --cc dcqcn  --bw 100 --topo topology --hpai 50
NS_LOG="QbbNetDevice=info" python run.py --cc dcqcn --bw 100 --topo topology --hpai 50

```

## gdb


```
ps aux | grep scratch/third
gdb -p 
```

```
(gdb) bt
#0  0x00007f5b0460b149 in ns3::RdmaQueuePairGroup::Get(unsigned int) () at ../src/point-to-point/model/rdma-queue-pair.cc:208
#1  0x00007f5b045f7a65 in ns3::RdmaEgressQueue::GetNextQindex (this=0x15f12a0, paused=paused@entry=0x15f1238) at ../src/point-to-point/model/qbb-net-device.cc:140
#2  0x00007f5b045f7d2e in ns3::QbbNetDevice::DequeueAndTransmit (this=0x15f0ff0) at ../src/point-to-point/model/qbb-net-device.cc:332
#3  0x00007f5b045f747f in ns3::QbbNetDevice::TransmitComplete (this=0x15f0ff0) at ../src/point-to-point/model/qbb-net-device.cc:321
#4  0x00007f5b03cc4843 in ns3::DefaultSimulatorImpl::ProcessOneEvent (this=this@entry=0x6e9f00) at ../src/core/model/default-simulator-impl.cc:142
#5  0x00007f5b03cc48be in ns3::DefaultSimulatorImpl::Run (this=0x6e9f00) at ../src/core/model/default-simulator-impl.cc:199
#6  0x000000000040c4a9 in main (argc=<optimized out>, argv=<optimized out>) at ../scratch/third.cc:1017
(gdb)
```


### 死锁判断


```
(gdb) p ns3::Simulator::Now()
$1 = 2105
(gdb) c
(让程序继续跑几秒，再次 Ctrl+C 中断)
(gdb) p ns3::Simulator::Now()
$2 = 2142

```


```
(gdb)  p ns3::Simulator::Now().GetSeconds()
Cannot evaluate function -- may be inlined
(gdb)  p ns3::Simulator::Now()
$1 = {static g_markingTimes = 0x0, m_data = 3368531968}
(gdb) c
Continuing.
^C
Program received signal SIGINT, Interrupt.
0x00007fd5b08baf89 in ns3::Packet::AddHeader (this=0x10f08dab0, header=...) at ../src/network/model/packet.cc:264
264     }
(gdb)  p ns3::Simulator::Now()
$2 = {static g_markingTimes = 0x0, m_data = 3383908466}
(gdb) c
Continuing.
```

如果发现打印出来的 Simulator::Now() 永远不改变，但这个函数在疯狂疯狂输出，说明你陷入了死循环
```
NS_LOG_INFO("[Qbb-Debug-Entry] Time: " << Simulator::Now().GetNanoSeconds() 
					<< " ns | Broadcom Q TotalPackets: " << m_queue->GetTotalPackets() );
```

## 比较

```
./waf clean

rm -rf build/
./waf configure -d optimized --enable-static
./waf build
```


```
diff -s mix/fct_topology_flow_fifo.txt mix/fct_topology_flow_vct.txt
```


```
du -sh ../simulation/mix/fct_topology_flow_fifo.txt 
124M    ../simulation/mix/fct_topology_flow_fifo.txt
```

# ns3 源码


```
        bool QbbNetDevice::SwitchSend (uint32_t qIndex, Ptr<Packet> packet, CustomHeader &ch){
                m_macTxTrace(packet);
                m_traceEnqueue(packet, qIndex);
                m_queue->Enqueue(packet, qIndex);
                DequeueAndTransmit();
                return true;
        }
		    bool QbbNetDevice::Send(Ptr<Packet> packet, const Address &dest, uint16_t protocolNumber)
        {
                NS_ASSERT_MSG(false, "QbbNetDevice::Send not implemented yet\n");
                return false;
        }
     void
                QbbNetDevice::Receive(Ptr<Packet> packet)
        {
                NS_LOG_FUNCTION(this << packet);
                if (!m_linkUp){
                        m_traceDrop(packet, 0);
                        return;
                }

                if (m_receiveErrorModel && m_receiveErrorModel->IsCorrupt(packet))
                {
                        // 
                        // If we have an error model and it indicates that it is time to lose a
                        // corrupted packet, don't forward this packet up, let it go.
                        //
                        m_phyRxDropTrace(packet);
                        return;
                }

                m_macRxTrace(packet);
                CustomHeader ch(CustomHeader::L2_Header | CustomHeader::L3_Header | CustomHeader::L4_Header);
                ch.getInt = 1; // parse INT header
                packet->PeekHeader(ch);
                if (ch.l3Prot == 0xFE){ // PFC
                        if (!m_qbbEnabled) return;
                        unsigned qIndex = ch.pfc.qIndex;
                        if (ch.pfc.time > 0){
                                m_tracePfc(1);
                                m_paused[qIndex] = true;
                        }else{
                                m_tracePfc(0);
                                Resume(qIndex);
                        }
                }else { // non-PFC packets (data, ACK, NACK, CNP...)
                        if (m_node->GetNodeType() > 0){ // switch
                                packet->AddPacketTag(FlowIdTag(m_ifIndex));
                                m_node->SwitchReceiveFromDevice(this, packet, ch);
                        }else { // NIC
                                // send to RdmaHw
                                int ret = m_rdmaReceiveCb(packet, ch);
                                // TODO we may based on the ret do something
                        }
                }
                return;
        }
```

1. 服务器端（NIC）不走此路径：在 Receive 中，如果是服务器收到包，直接递交给上层硬件回调 m_rdmaReceiveCb(packet, ch)，这也是为什么你原先 DequeueAndTransmit 的 GetNodeType() == 0 分支完全保留原厂默认网卡逻辑的原因。

2. 交换机（Switch）发包核心在 SwitchSend：当交换机内部完成路由/交换查找后，会统一调用当前出接口 QbbNetDevice 的 SwitchSend 函数。原厂的做法是直接把包塞进博通队列并尝试触发发送：m_queue->Enqueue(packet, qIndex); DequeueAndTransmit();。

因此，要在当前架构下最完美、最优雅地实现你的自定义网络包调度器，你应当把入队拦截点放在 SwitchSend，把出队调度点放在修改后的 DequeueAndTransmit。





network/utils/broadcom-egress-queue.h
```
namespace ns3 {

	NS_OBJECT_ENSURE_REGISTERED(BEgressQueue);

	TypeId BEgressQueue::GetTypeId(void)
	{
		static TypeId tid = TypeId("ns3::BEgressQueue")
			.SetParent<Queue>()
			.AddConstructor<BEgressQueue>()
			.AddAttribute("MaxBytes",
				"The maximum number of bytes accepted by this BEgressQueue.",
				DoubleValue(1000.0 * 1024 * 1024),
				MakeDoubleAccessor(&BEgressQueue::m_maxBytes),
				MakeDoubleChecker<double>())
			.AddTraceSource ("BeqEnqueue", "Enqueue a packet in the BEgressQueue. Multiple queue",
					MakeTraceSourceAccessor (&BEgressQueue::m_traceBeqEnqueue))
			.AddTraceSource ("BeqDequeue", "Dequeue a packet in the BEgressQueue. Multiple queue",
					MakeTraceSourceAccessor (&BEgressQueue::m_traceBeqDequeue))
			;

		return tid;
	}

	BEgressQueue::BEgressQueue() :
		Queue()
	{
		NS_LOG_FUNCTION_NOARGS();
		m_bytesInQueueTotal = 0;
		m_rrlast = 0;
		for (uint32_t i = 0; i < fCnt; i++)
		{
			m_bytesInQueue[i] = 0;
			m_queues.push_back(CreateObject<DropTailQueue>());
		}
	}

	BEgressQueue::~BEgressQueue()
	{
		NS_LOG_FUNCTION_NOARGS();
	}

	bool
		BEgressQueue::DoEnqueue(Ptr<Packet> p, uint32_t qIndex)
	{
		NS_LOG_FUNCTION(this << p);

		if (m_bytesInQueueTotal + p->GetSize() < m_maxBytes)  //infinite queue
		{
			m_queues[qIndex]->Enqueue(p);
			m_bytesInQueueTotal += p->GetSize();
			m_bytesInQueue[qIndex] += p->GetSize();
		}
		else
		{
			return false;
		}
		return true;
	}

	Ptr<Packet>
		BEgressQueue::DoDequeueRR(bool paused[]) //this is for switch only
	{
		NS_LOG_FUNCTION(this);

		if (m_bytesInQueueTotal == 0)
		{
			NS_LOG_LOGIC("Queue empty");
			return 0;
		}
		bool found = false;
		uint32_t qIndex;

		if (m_queues[0]->GetNPackets() > 0) //0 is the highest priority
		{
			found = true;
			qIndex = 0;
		}
		else
		{
			if (!found)
			{
				for (qIndex = 1; qIndex <= qCnt; qIndex++)
				{
					if (!paused[(qIndex + m_rrlast) % qCnt] && m_queues[(qIndex + m_rrlast) % qCnt]->GetNPackets() > 0)  //round robin
					{
						found = true;
						break;
					}
				}
				qIndex = (qIndex + m_rrlast) % qCnt;
			}
		}
		if (found)
		{
			Ptr<Packet> p = m_queues[qIndex]->Dequeue();
			m_traceBeqDequeue(p, qIndex);
			m_bytesInQueueTotal -= p->GetSize();
			m_bytesInQueue[qIndex] -= p->GetSize();
			if (qIndex != 0)
			{
				m_rrlast = qIndex;
			}
			m_qlast = qIndex;
			NS_LOG_LOGIC("Popped " << p);
			NS_LOG_LOGIC("Number bytes " << m_bytesInQueueTotal);
			return p;
		}
		NS_LOG_LOGIC("Nothing can be sent");
		return 0;
	}

	bool
		BEgressQueue::Enqueue(Ptr<Packet> p, uint32_t qIndex)
	{
		NS_LOG_FUNCTION(this << p);
		//
		// If DoEnqueue fails, Queue::Drop is called by the subclass
		//
		bool retval = DoEnqueue(p, qIndex);
		if (retval)
		{
			NS_LOG_LOGIC("m_traceEnqueue (p)");
			m_traceEnqueue(p);
			m_traceBeqEnqueue(p, qIndex);

			uint32_t size = p->GetSize();
			m_nBytes += size;
			m_nTotalReceivedBytes += size;

			m_nPackets++;
			m_nTotalReceivedPackets++;
		}
		return retval;
	}

	Ptr<Packet>
		BEgressQueue::DequeueRR(bool paused[])
	{
		NS_LOG_FUNCTION(this);
		Ptr<Packet> packet = DoDequeueRR(paused);
		if (packet != 0)
		{
			NS_ASSERT(m_nBytes >= packet->GetSize());
			NS_ASSERT(m_nPackets > 0);
			m_nBytes -= packet->GetSize();
			m_nPackets--;
			NS_LOG_LOGIC("m_traceDequeue (packet)");
			m_traceDequeue(packet);
		}
		return packet;
	}

	bool
		BEgressQueue::DoEnqueue(Ptr<Packet> p)	//for compatiability
	{
		std::cout << "Warning: Call Broadcom queues without priority\n";
		uint32_t qIndex = 0;
		NS_LOG_FUNCTION(this << p);
		if (m_bytesInQueueTotal + p->GetSize() < m_maxBytes)
		{
			m_queues[qIndex]->Enqueue(p);
			m_bytesInQueueTotal += p->GetSize();
			m_bytesInQueue[qIndex] += p->GetSize();
		}
		else
		{
			return false;

		}
		return true;
	}


	Ptr<Packet>
		BEgressQueue::DoDequeue(void)
	{
		NS_ASSERT_MSG(false, "BEgressQueue::DoDequeue not implemented");
		return 0;
	}


	Ptr<const Packet>
		BEgressQueue::DoPeek(void) const	//DoPeek doesn't work for multiple queues!!
	{
		std::cout << "Warning: Call Broadcom queues without priority\n";
		NS_LOG_FUNCTION(this);
		if (m_bytesInQueueTotal == 0)
		{
			NS_LOG_LOGIC("Queue empty");
			return 0;
		}
		NS_LOG_LOGIC("Number bytes " << m_bytesInQueue);
		return m_queues[0]->Peek();
	}

	uint32_t
		BEgressQueue::GetNBytes(uint32_t qIndex) const
	{
		return m_bytesInQueue[qIndex];
	}


	uint32_t
		BEgressQueue::GetNBytesTotal() const
	{
		return m_bytesInQueueTotal;
	}

	uint32_t
		BEgressQueue::GetLastQueue()
	{
		return m_qlast;
	}

}
```