# TCP BBR {#h1-tcp-bbr}

TCP BBR 已经在 Youtube 服务器和 Google 跨数据中心的内部广域网（B4）上部署。

* 论文：[http://queue.acm.org/detail.cfm?id=3022184](http://queue.acm.org/detail.cfm?id=3022184)
* Linux内核：4.9+[commit](http://git.kernel.org/cgit/linux/kernel/git/davem/net-next.git/commit/?id=0f8782ea14974ce992618b55f0c041ef43ed0b78)

* 即时速率的计算：该带宽是bbr一切计算的基准，bw=应答数据/应答这些数据所用时间（只关注数据的大小，不关注数据的含义） 
  ![images](./assets/network-virtualnet-linuxnet-tcp-bbr1.png)
* RTT跟踪：系统会跟踪当前为止最小RTT
* bbr pipe状态机：STARTUP，DRAIN，PROBE\_BW，PROBE\_RTT  
  ![images](./assets/network-virtualnet-linuxnet-tcp-bbr2.png)  
  ![images](./assets/network-virtualnet-linuxnet-tcp-bbr3.png)

  * bbr处在各个状态时的增益系数：

    * 结果输出：pacing rate（规定cwnd指示的一窗数据的数据包之间，以多大的时间间隔发送出去）和cwnd
      * pacing rate怎么计算？很简单，就是是使用时间窗口内\(默认10轮采样\)最大BW。上一次采样的即时BW，用它来在可能的情况下更新时间窗口内的BW采样值集合。这次能否按照这个时间窗口内最大BW发送数据呢？这样看当前的增益系数的值，设为G，那么BW\*G就是pacing rate的值，是不是很简单呢？！
        * 至于说cwnd的计算可能要稍微复杂一点，但是也是可以理解的，我们知道，cwnd其实描述了一条网络管道\(rwnd描述了接收端缓冲区\)，因此cwnd其实就是这个管道的容量，也就是BDP！
        * BW我们已经有了，缺少的是D，也就是RTT，不过别忘了，bbr一直在持续搜集最小的RTT值，注意，bbr并没有采用什么移动指数平均算法来“猜测”RTT\(我用猜测而不是预测的原因是，猜测的结果往往更加不可信！\)，而是直接冒泡采集最小的RTT\(注意这个RTT是TCP系统层面移动指数平均的结果，即SRTT，但brr并不会对此结果再次做平均！\)。我们用这个最小RTT干什么呢？ 当前是计算BDP了！这里bbr取的RTT就是这个最小RTT。最小RTT表示一个曾经达到的最佳RTT，既然曾经达到过，说明这是客观的可以再次达到的RTT，这样有益于网络管道利用率最大化！
        * 我们采用BDP\*G’就算出了cwnd，这里的G’是cwnd的增益系数，与带宽增益系数含义一样，根据bbr的状态机来获取！

    bbr确实忽略了Recovery等非Open的拥塞状态:![images](./assets/network-virtualnet-linuxnet-tcp-bbr4.png)

## BBR算法 {#h2-bbr-}

**TCP BBR 致力于解决两个问题：**

1. **在有一定丢包率的网络链路上充分利用带宽。**
2. **降低网络链路上的 buffer 占用率，从而降低延迟。**

TCP 拥塞控制的目标是最大化利用网络上瓶颈链路的带宽: 网络内尚未被确认收到的数据包数量 = 网络链路上能容纳的数据包数量 = 链路带宽 × 往返延迟。

TCP 维护一个发送窗口，估计当前网络链路上能容纳的数据包数量，希望在有数据可发的情况下，回来一个确认包就发出一个数据包，总是保持发送窗口那么多个包在网络中流动。标准 TCP 中的拥塞控制算法也类似：不断增加发送窗口，直到发现开始丢包。这就是所谓的 ”加性增，乘性减”，也就是当收到一个确认消息的时候慢慢增加发送窗口，当确认一个包丢掉的时候较快地减小发送窗口。

标准 TCP 的这种做法有两个问题：

* 首先，假定网络中的丢包都是由于拥塞导致（网络设备的缓冲区放不下了，只好丢掉一些数据包）。事实上网络中有可能存在传输错误导致的丢包，基于丢包的拥塞控制算法并不能区分拥塞丢包和错误丢包。在数据中心内部，错误丢包率在十万分之一（1e-5）的量级；在广域网上，错误丢包率一般要高得多。更重要的是，“加性增，乘性减” 的拥塞控制算法要能正常工作，错误丢包率需要与发送窗口的平方成反比。数据中心内的延迟一般是 10-100 微秒，带宽 10-40 Gbps，乘起来得到稳定的发送窗口为 12.5 KB 到 500 KB。而广域网上的带宽可能是 100 Mbps，延迟 100 毫秒，乘起来得到稳定的发送窗口为 10 MB。广域网上的发送窗口比数据中心网络高 1-2 个数量级，错误丢包率就需要低 2-4 个数量级才能正常工作。因此标准 TCP 在有一定错误丢包率的长肥管道（long-fat pipe，即延迟高、带宽大的链路）上只会收敛到一个很小的发送窗口。这就是很多时候客户端和服务器都有很大带宽，运营商核心网络也没占满，但下载速度很慢，甚至下载到一半就没速度了的一个原因。
* 其次，网络中会有一些 buffer，就像输液管里中间膨大的部分，用于吸收网络中的流量波动。由于标准 TCP 是通过 “灌满水管” 的方式来估算发送窗口的，在连接的开始阶段，buffer 会被倾向于占满。后续 buffer 的占用会逐渐减少，但是并不会完全消失。客户端估计的水管容积（发送窗口大小）总是略大于水管中除去膨大部分的容积。这个问题被称为 bufferbloat（缓冲区膨胀）。

缓冲区膨胀有两个危害：

* 增加网络延迟。buffer 里面的东西越多，要等的时间就越长嘛。
* 共享网络瓶颈的连接较多时，可能导致缓冲区被填满而丢包。很多人把这种丢包认为是发生了网络拥塞，实则不然。

![images](./assets/network-virtualnet-linuxnet-tcp-bbr5.png)

往返延迟随时间的变化。红线：标准 TCP（可见周期性的延迟变化，以及 buffer 几乎总是被填满）；绿线：TCP BBR

有很多论文提出在网络设备上把当前缓冲区大小的信息反馈给终端，比如在数据中心广泛应用的 ECN（Explicit Congestion Notification）。然而广域网上网络设备众多，更新换代困难，需要网络设备介入的方案很难大范围部署。

TCP BBR 是怎样解决以上两个问题的呢？

* 既然不容易区分拥塞丢包和错误丢包，TCP BBR 就干脆不考虑丢包。
* 既然灌满水管的方式容易造成缓冲区膨胀，TCP BBR 就分别估计带宽和延迟，而不是直接估计水管的容积。

TCP BBR 解决带宽和延迟无法同时测准的方法是：**交替测量带宽和延迟；用一段时间内的带宽极大值和延迟极小值作为估计值**。

在连接刚建立的时候，TCP BBR 采用类似标准 TCP 的慢启动，指数增长发送速率。然而标准 TCP 遇到任何一个丢包就会立即进入拥塞避免阶段，它的本意是填满水管之后进入拥塞避免，然而（1）如果链路的错误丢包率较高，没等到水管填满就放弃了；（2）如果网络里有 buffer，总要把 buffer 填满了才会放弃。

TCP BBR 则是根据收到的确认包，发现有效带宽不再增长时，就进入拥塞避免阶段。（1）链路的错误丢包率对 BBR 没有影响；（2）当发送速率增长到开始占用 buffer 的时候，有效带宽不再增长，BBR 就及时放弃了（事实上放弃的时候占的是 3 倍带宽 × 延迟，后面会把多出来的 2 倍 buffer 清掉）。

![images](./assets/network-virtualnet-linuxnet-tcp-bb6.png)

发送窗口与往返延迟和有效带宽的关系。BBR 会在左右两侧的拐点之间停下，基于丢包的标准 TCP 会在右侧拐点停下

在慢启动过程中，由于 buffer 在前期几乎没被占用，延迟的最小值就是延迟的初始估计；慢启动结束时的最大有效带宽就是带宽的初始估计。

慢启动结束后，为了把多占用的 2 倍带宽 × 延迟消耗掉，BBR 将进入排空（drain）阶段，指数降低发送速率，此时 buffer 里的包就被慢慢排空，直到往返延迟不再降低。

![images](./assets/network-virtualnet-linuxnet-tcp-bbr7.png)

TCP BBR（绿线）与标准 TCP（红线）有效带宽和往返延迟的比较

排空阶段结束后，BBR 进入稳定运行状态，交替探测带宽和延迟。由于网络带宽的变化比延迟的变化更频繁，BBR 稳定状态的绝大多数时间处于带宽探测阶段。带宽探测阶段是一个正反馈系统：定期尝试增加发包速率，如果收到确认的速率也增加了，就进一步增加发包速率。

具体来说，以每 8 个往返延迟为周期，在第一个往返的时间里，BBR 尝试增加发包速率 1/4（即以估计带宽的 5/4 速度发送）。在第二个往返的时间里，为了把前一个往返多发出来的包排空，BBR 在估计带宽的基础上降低 1/4 作为发包速率。剩下 6 个往返的时间里，BBR 使用估计的带宽发包。

当网络带宽增长一倍的时候，每个周期估计带宽会增长 1/4，每个周期为 8 个往返延迟。其中向上的尖峰是尝试增加发包速率 1/4，向下的尖峰是降低发包速率 1/4（排空阶段），后面 6 个往返延迟，使用更新后的估计带宽。3 个周期，即 24 个往返延迟后，估计带宽达到增长后的网络带宽。

![images](./assets/network-virtualnet-linuxnet-tcp-bbr8.png)

网络带宽增长一倍时的行为。绿线为网络中包的数量，蓝线为延迟

当网络带宽降低一半的时候，多出来的包占用了 buffer，导致网络中包的延迟显著增加（下图蓝线），有效带宽降低一半。延迟是使用极小值作为估计，增加的实际延迟不会反映到估计延迟（除非在延迟探测阶段，下面会讲）。带宽的估计则是使用一段滑动窗口时间内的极大值，当之前的估计值超时（移出滑动窗口）之后，降低一半后的有效带宽就会变成估计带宽。估计带宽减半后，发送窗口减半，发送端没有窗口无法发包，buffer 被逐渐排空。

  
![images](./assets/network-virtualnet-linuxnet-tcp-bbr9.png)

当带宽增加一倍时，BBR 仅用 1.5 秒就收敛了；而当带宽降低一半时，BBR 需要 4 秒才能收敛。前者由于带宽增长是指数级的；后者主要是由于带宽估计采用滑动窗口内的极大值，需要一定时间有效带宽的下降才能反馈到带宽估计中。

当网络带宽保持不变的时候，稳定状态下的 TCP BBR 是下图这样的：（我们前面看到过这张图）可见每 8 个往返延迟为周期的延迟细微变化。

![images](./assets/network-virtualnet-linuxnet-tcp-bbr11.png)

往返延迟随时间的变化。红线：标准 TCP；绿线：TCP BBR

上面介绍了 BBR 稳定状态下的带宽探测阶段，那么什么时候探测延迟呢？在带宽探测阶段中，估计延迟始终是使用极小值，如果实际延迟真的增加了怎么办？TCP BBR 每过 10 秒，如果估计延迟没有改变（也就是没有发现一个更低的延迟），就进入延迟探测阶段。延迟探测阶段持续的时间仅为 200 毫秒（或一个往返延迟，如果后者更大），这段时间里发送窗口固定为 4 个包，也就是几乎不发包。这段时间内测得的最小延迟作为新的延迟估计。也就是说，大约有 2% 的时间 BBR 用极低的发包速率来测量延迟。

TCP BBR 还使用 pacing 的方法降低发包时的 burstiness，减少突然传输的一串包导致缓冲区膨胀。

下面我们来看 TCP BBR 的效果如何。

首先看 BBR 试图解决的第一个问题：在有随机丢包情况下的吞吐量。如下图所示，只要有万分之一的丢包率，标准 TCP 的带宽就只剩 30%；千分之一丢包率时只剩 10%；有百分之一的丢包率时几乎就卡住了。而 TCP BBR 在丢包率 5% 以下几乎没有带宽损失，在丢包率 15% 的时候仍有 75% 带宽。

![images](./assets/network-virtualnet-linuxnet-tcp-bbr121.png)

100 Mbps，100ms 下的丢包率和有效带宽（红线：标准 TCP，绿线：TCP BBR）

异地数据中心间跨广域网的传输往往是高带宽、高延迟的，且有一定丢包率，TCP BBR 可以显著提高传输速度。这也是中国科大 LUG HTTP 代理服务器和 Google 广域网（B4）部署 TCP BBR 的主要原因。

再来看 BBR 试图解决的第二个问题：降低延迟，减少缓冲区膨胀。如下图所示，标准 TCP 倾向于把缓冲区填满，缓冲区越大，延迟就越高。当用户的网络接入速度很慢时，这个延迟可能超过操作系统连接建立的超时时间，导致连接建立失败。使用 TCP BBR 就可以避免这个问题。

![images](./assets/network-virutalnet-linuxnet-tcp-bbr122.png)

缓冲区大小与延迟的关系（红线：标准 TCP，绿线：TCP BBR）

Youtube 部署了 TCP BBR 之后，全球范围的中位数延迟降低了 53%（也就是快了一倍），发展中国家的中位数延迟降低了 80%（也就是快了 4 倍）。从下图可见，延迟越高的用户，采用 TCP BBR 后的延迟下降比例越高，原来需要 10 秒的现在只要 2 秒了。如果您的网站需要让用 GPRS 或者慢速 WiFi 接入网络的用户也能流畅访问，不妨试试 TCP BBR。

![images](./assets/network-virtualnet-linuxnet-tcp-bbr123.png)

标准 TCP 与 TCP BBR 的往返延迟中位数之比

综上，TCP BBR 不再使用丢包作为拥塞的信号，也不使用 “加性增，乘性减” 来维护发送窗口大小，而是分别估计极大带宽和极小延迟，把它们的乘积作为发送窗口大小。

BBR 的连接开始阶段由慢启动、排空两阶段构成。为了解决带宽和延迟不易同时测准的问题，BBR 在连接稳定后交替探测带宽和延迟，其中探测带宽阶段占绝大部分时间，通过正反馈和周期性的带宽增益尝试来快速响应可用带宽变化；偶尔的探测延迟阶段发包速率很慢，用于测准延迟。

BBR 解决了两个问题：  
在有一定丢包率的网络链路上充分利用带宽。非常适合高延迟、高带宽的网络链路。  
降低网络链路上的 buffer 占用率，从而降低延迟。非常适合慢速接入网络的用户。

大概是由于 ACM queue 的篇幅限制和目标读者，这篇论文并没有讨论（仅有拥塞丢包情况下）TCP BBR 与标准 TCP 的公平性。也没有讨论 BBR 与现有拥塞控制算法的比较，如基于往返延迟的（如 Vegas）、综合丢包和延迟因素的（如 Compound TCP）、基于网络设备提供拥塞信息的（如 ECN）、网络设备采用新调度策略的（如 CoDel）。期待 Google 发表更详细的论文，也期待各位同行报告 TCP BBR 在实验或生产环境中的性能。

## 开启BBR \(linux kernel 4.9+\) {#h2--bbr-linux-kernel-4-9-}

NOTE: BBR_must_be used with the fq qdisc \(“man tc-fq”\) with pacing  
enabled, since pacing is integral to the BBR design and  
implementation. BBR without pacing would not function properly, and  
may incur unnecessary high packet loss rates.

```
# upgrade kernel
rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
rpm -Uvh http://www.elrepo.org/elrepo-release-7.0-2.el7.elrepo.noarch.rpm
yum --enablerepo=elrepo-kernel install kernel-ml -y
grub2-set-default 0
reboot
# enable bbr
echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf
sysctl -p
```

## Ubuntu 升级内核的方法 {#h2-ubuntu-}

```
wget http://kernel.ubuntu.com/~kernel-ppa/mainline/v4.9/linux-image-4.9.0-040900-generic_4.9.0-040900.201612111631_amd64.deb
dpkg -i linux-image-4.9.0*.deb
update-grub
reboot
```



