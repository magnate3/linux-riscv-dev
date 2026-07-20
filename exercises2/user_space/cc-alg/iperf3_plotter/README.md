> #  netem

> ## netem模拟丢包设置
```
# 发送的报文有 50% 的丢包率
tc qdisc change dev eth0 root netem loss 50%

# 发送的报文有 0.3% ~ 25% 的丢包率
tc qdisc change dev eth0 root netem loss 0.3% 25%
```


> ##  netem延迟设置
 
 
DELAY := delay TIME [ JITTER [ CORRELATION ]]]
    [ distribution { uniform | normal | pareto |  paretonormal } ]     
TIME：延迟的时间   
JITTER：抖动，增加一个随机时间长度，让延迟时间出现在某个范围    
CORRELATION：相关，下一个报文延迟时间和上一个报文的相关系数    
distribution：分布，延迟的分布模式，可以选择的值有 uniform、normal、pareto 和 paretonormal   
 
```
# eth0 网卡延迟增加100ms 
tc qdisc add dev eth0 root netem delay 100ms

# 报文延迟的时间在 100ms ± 20ms 之间（90ms - 110ms）
tc qdisc add dev eth0 root netem delay 100ms 20ms

# 因为网络状况是平滑变化的，短时间里相邻报文的延迟应该是近似的而不是完全随机的。这个值是个百分比，如果为 100%，就退化到固定延迟的情况；如果是 0% 则退化到随机延迟的情况
tc qdisc change dev eth0 root netem delay 100ms 20ms 50%

# distribution 参数来限制它的延迟分布模型。比如让报文延迟时间满足正态分布
tc qdisc change dev eth0 root netem delay 100ms 20ms distribution normal
```
> ##  netem模拟丢包设置
```
# 发送的报文有 50% 的丢包率
tc qdisc change dev eth0 root netem loss 50%

# 发送的报文有 0.3% ~ 25% 的丢包率
tc qdisc change dev eth0 root netem loss 0.3% 25%
丢包也支持 state（4-state Markov 模型） 和 gemodel（Gilbert-Elliot 丢包模型） 两种模型的丢包配置。不过相对比较复杂，这里我们就不再详细描述。
```

> ##  netem模拟报文重复\损坏设置
```
# 随机产生 50% 重复的包
# tc qdisc change dev eth0 root netem loss 50%  # 原错误命令
tc qdisc change dev eth0 root netem duplicate 50%

# 随机产生 2% 损坏的报文（在报文的随机位置造成一个比特的错误）
tc qdisc change dev eth0 root netem corrupt 2%
```

> ## netem模拟包乱序
网络传输并不能保证顺序，传输层 TCP 会对报文进行重组保证顺序，所以报文乱序对应用的影响比上面的几种问题要小。
 
```
# 固定的每隔一定数量的报文就乱序一次
tc qdisc change dev eth0 root netem reorder 50% gap 3 delay 100ms
# 使用概率来选择乱序的报文
tc qdisc change dev eth0 root netem reorder 50% 15% delay 300ms
```


#  iperf3


> ## make   
```
root@ubuntux86:# make
apt-get -y install iperf3 jq gnuplot
Reading package lists... Done
Building dependency tree       
Reading state information... Done
gnuplot is already the newest version (5.2.8+dfsg1-2).
iperf3 is already the newest version (3.7-3).
jq is already the newest version (1.6-1ubuntu0.20.04.1).
0 upgraded, 0 newly installed, 0 to remove and 596 not upgraded.
cp preprocessor.sh /usr/bin
cp plot_iperf.sh /usr/bin
cp plot_* /usr/bin
cp fairness.sh /usr/bin
root@ubuntux86:# 
```


```
iperf3 -c 10.22.116.221  -C bbr -p 5202  -J -P p -t 60 > my_test.json
```

> ## iperf3测试命令

```
iperf3 -c $server_ip -p 49800 -i 0.1 -M 1400 -t 100 -P6 -w16M -M1400 -J > iperf3.json
其中 
-c表示运行在客户端模式，server_ip是iperf服务端地址。 
-p是端口号。 
-i是report的间隔，单位秒，0.1表示100ms统计一次。 
-t 100表示测试100秒。 
-P6表示同时使用6个stream来测试。提高这个参数，可以提高Throughput。 
-M 1400表示tcp_mss是1400。 
-w16M 表示window大小是16MB。提高这个参数，可以提高Throughput。 
不加-u(udp)表示测试使用tcp。 
-J表示输出json格式。
```

JSON统计信息   

```
iperf输出的iperf3.json由start，interval和end这3大部分组成。 
start里的是各stream的信息，比如本地ip，端口，远端的server的ip，端口。这些信息可以后续用于wireshark的分析。 
interval里有各stream在某些间隔的信息，比如retransmits是重传次数，snd_cwnd是拥塞窗口，rtt是延迟，pmtu是路径的mtu，可以看到是否有比较小的mtu存在。 
end里有各stream的统计，总的retransmits，max_snd_cwnd，mean_rtt，sum_sent和sum_received的bits_per_second等。以及发送方，接收方的流控机制，是cubic还是bbr。
```

数据分析   

首先统计出iperf输出的内容，主要是snd_cwnd和rtt的中位数、最小值、最大值。其中snd_cwnd是发送端的拥塞窗口，和TCP receive window中较小的一个，决定了发送方的send window。     



> ### 吞吐量的计算

```
sustained throughput = window size / RTT 
先算出某个stream的TP: 
2.8MB/0.094s = 29MBps = 238Mbps 
再把各stream都算出来，或者乘起来，得到最终的结果。
```

> ### Bytes In Flight的计算

BDP(bandwidth delay product)是~10Gbps * 0.094s = 0.94Gb = 118MB。所以这条链路上，不能超过118MB，receiver buffer也要大于等于118MB。   

> ### 提高吞吐量的方法

根据公式sustained throughput = window size / RTT，可以知道，我们可以提高window size或者减少RTT，来提高吞吐量。    

> ### window size的调整

```
尽可能避免丢包，丢包会导致慢启动，会影响窗口大小。当然也可能是快速重传。
根据带宽和时延来调整拥塞控制算法，比如使用BBR(Bottleneck Bandwidth and RTT)代替CUBIC算法。
# 查看支持哪些拥塞控制算法
cat /proc/sys/net/ipv4/tcp_allowed_congestion_control
# 看现在配置了什么算法
sysctl net.ipv4.tcp_congestion_control
# 修改为BBR
sysctl net.ipv4.tcp_congestion_control=bbr
# 可以在某些队列增加丢包来测试
tc qdisc replace dev eth0 root netem loss 2% latency 10ms
如果老版本不支持bbr，需要自己编译https://github.com/google/bbr，加载tcp_bbr.ko

修改初始拥塞窗口ICW(initial congestion window) 
拥塞窗口是在发送端控制的。
ss -nli | fgrep cwnd
cubic cwnd:10
输出是10的话，说明已经是10个MSS了。 
```

改变的方法是：
```
# 找到现有配置的路由
ip route
192.168.1.0/24 dev eth0  proto kernel  scope link  src 192.168.1.100  metric 1
169.254.0.0/16 dev eth0  scope link  metric 1000
default via 192.168.1.1 dev eth0  proto static
# ip route change后面复制整行路由，在最后加上initcwnd 10即可
# 比如复制现有的default那条配置
ip route change default via 192.168.1.1 dev eth0 proto static initcwnd 10
# 或只修改某条路由的配置
ip route change 192.168.1.0/24 dev eth0  proto kernel  scope link  src 192.168.1.100  metric 1
修改缓存队列算法


# 看某个队列是否有丢包
tc -s qdisc ls
qdisc pfifo_fast 0: dev etha01 parent :4 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
Sent 155418 bytes 1310 pkt (dropped 0, overlimits 0 requeues 0)
backlog 0b 0p requeues 0
# 查看qdisc，默认一般是pfifo_fast，先从fifo0 pop，再fifo1，fifo2等
sysctl net.core.default_qdisc
net.core.default_qdisc = pfifo_fast
# 修改为fq，根据src/dst的ip/port四元组来hash，防止单条流影响其它流，在man tc-sfq可以看到说明
sysctl net.core.default_qdisc=fq
修改缓存大小
# 看发送缓存(单位byte，分别是下限，默认值，上限)
sudo sysctl net.ipv4.tcp_wmem
net.ipv4.tcp_wmem = 4096        16384   4194304
# 看接收缓存，max要配置为大于等于BDF
sudo sysctl net.ipv4.tcp_rmem
net.ipv4.tcp_rmem = 2097152     4194304 8388608
# 自动调整接收端缓存大小
sudo sysctl -a | grep net.ipv4.tcp_moderate_rcvbuf
net.ipv4.tcp_moderate_rcvbuf = 1
# 看现有的tcp_mem情况(单位page的大小，比如4KB，分别是下限，默认值，上限)
sudo sysctl -a | grep net.ipv4.tcp_mem
net.ipv4.tcp_mem = 89028        118707  178056
增加路径的MTU 
如果iperf的输出显示pmtu比较小，没有达到期望的jumbo frame这么大，就可以看下哪里的mtu比较小，调大它，这样可以减少分片，从而提高吞吐量。
```
 