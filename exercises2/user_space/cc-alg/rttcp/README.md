
# rttcp

```
./rttcp.py analyze --type "flow" -i 100G.pcap -o trace.pcap.flow.txt
Traceback (most recent call last):
  File "./rttcp.py", line 129, in <module>
    main(sys.argv)
  File "./rttcp.py", line 115, in main
    packet_dumper.run()
  File "/root/rttcp/packet_dumper.py", line 106, in run
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
  File "/usr/lib/python2.7/subprocess.py", line 394, in __init__
    errread, errwrite)
  File "/usr/lib/python2.7/subprocess.py", line 1047, in _execute_child
    raise child_exception
OSError: [Errno 2] No such file or directory
```

```
apt-get install tshark
```
```
 python2 ./rttcp.py analyze --type "packet" -i    70G.pcap  -o trace.pcap.flow.txt
```

```
 tshark -i enp61s0f1np1 -f "tcp port 9999" -w test.pcap -P -a duration:10
```








+ 只对delta4, 进行分析抓包方法   
    
dst port 53972 是iperf client port   
```
tcpdump  -i enp61s0f1np1 dst port 53972  -eennvv -w 70G.pcap
```
然后执行./rttcp.py analyze --type "packet"    
```
 python2 ./rttcp.py analyze --type "packet" -i    70G.pcap  -o trace.pcap.flow.txt
```

## cpu freq

```
watch -n 0.5  "lscpu -e=CPU,MHZ"
CPU      MHZ
  0 2795.132
  1 2705.405
  2 2000.000
  3 2000.000
  4 2000.000
```
POPULAR_HZ_VALUES参数   
```
POPULAR_HZ_VALUES = [100., 200., 250., 1000.,2000.,2800.]

```

 


# tcpdump
默认情况不加参数tcpdump抓包的话只抓每个数据包的前68个字节，也就是通常情况下抓完整的tcp    

```
 tcpdump  -env  -r  70G.pcap   src host 10.22.116.221 
```
只抓tcp头部
```
tcpdump  -i enp61s0f1np1 port 53972  -s 68   -eennvv -w 90G.pcap
```
tcp头部最长60字节
```
tcpdump  -i enp61s0f1np1 port 53972  -s 88   -eennvv -w 90G.pcap
```

```
netstat -pan | grep 9999
tcp       28 3795420 10.22.116.220:54336     10.22.116.221:9999      ESTABLISHED 57808/iperf         
tcp       28 3459432 10.22.116.220:54380     10.22.116.221:9999      ESTABLISHED 57808/iperf         
tcp       28 3563132 10.22.116.220:54346     10.22.116.221:9999      ESTABLISHED 57808/iperf         
tcp       28 2866268 10.22.116.220:54328     10.22.116.221:9999      ESTABLISHED 57808/iperf         
tcp       28 3314252 10.22.116.220:54354     10.22.116.221:9999      ESTABLISHED 57808/iperf         
tcp       28 4135556 10.22.116.220:54368     10.22.116.221:9999      ESTABLISHED 57808/iperf         
tcp       28 3795420 10.22.116.220:54386     10.22.116.221:9999      ESTABLISHED 57808/iperf         
tcp       28 3854472 10.22.116.220:54402     10.22.116.221:9999      ESTABLISHED 57808/iperf 
```
对flow 10.22.116.220:54336进行tcpdump    

# tshark

for centos   
```
 yum install wireshark
```

# wireshark

```
tcp.analysis.ack_rtt
tcp.analysis.bytes_in_flight
tcp.window_size
tcp.analysis.retransmission
```
Calculate Total Data Sent   
```
tshark -r h7_capture_cubic_150.pcap -Y "tcp && frame.time_relative <= 150" -T fields -e tcp.len | awk '{sum += $1} END {print sum}'
```
Calculate Total Data Retransmitted   
```
tshark -r h7_capture_cubic_150.pcap -Y "tcp.analysis.retransmission && frame.time_relative <= 150" -T fields -e tcp.len | awk '{sum += $1} END {print sum}'
```
Calculate Maximum Window Size   
```
tshark -r h7_capture_cubic_150.pcap -T fields -e tcp.window_size_value | sort -nr | head -1
```

+ Time-Sequence Graph
   
```
Time sequence stevens graph 
```
[Time-Sequence Graph (tcptrace).](https://packetbomb.com/understanding-the-tcptrace-time-sequence-graph-in-wireshark/)   


```
tshark -nr input.pcap -Y "display filter" -T fields -e frame.number -e tcp.seq -e tcp.options.timestamp.tsval
```

## wireshark 分析
port 53972 是client的 port,    
server  port 999      
抓包方法（包含client requeust and server response）      
```
tcpdump  -i enp61s0f1np1 port 53972  -s 128  -eennvv -w 70G.pcap
```
![images](rtt1.png)

![images](rtt2.png)


## tcptrace

```
apt-get -y install tcptrace
```

```
$tcpdump -w example.pcap -i eth0 -n
$tcptrace -R example.pcap
$xpl2gpl a2b_rtt.xpl
$gnuplot a2b_rtt.gpl
```

+ help

```
  -T      create throughput graph[s], (average over 10 segments, see -A)\n\
  -R      create rtt sample graph[s]\n\
  -S      create time sequence graph[s]\n\
  -N      create owin graph[s] (_o_utstanding data on _N_etwork)\n\
  -F      create segsize graph[s]\n\
  -L      create time line graph[s]\n\
  -G	  create ALL graphs\n\
```

###  tcptrace Rtt
[wirelshark tcptrace 识别](https://www.cnblogs.com/codestack/p/18023800)   
```
tcptrace -R 100-70G.pcap 
a2b_rtt.xpl
```

`bash xpl2gpl.txt a2b_rtt.xpl `产生如下文件     
```
a2b_rtt.datasets  a2b_rtt.gpl  a2b_rtt.labels
```
+ step3更改a2b_rtt.gpl后两行设置png输出      

```
set term png
set output "a2b_tsg.png"
```
+ step4    
```
gnuplot a2b_rtt.gpl
```


### tcptrace time sequence

+ step1   
```
tcptrace -S 100-70G.pcap
```
产生如下文件    

```
a2b_tsg.xpl  b2a_tsg.xpl
```
+ step2   
```
bash xpl2gpl.txt a2b_tsg.xpl 
```
+ step3更改a2b_tsg.gpl后两行设置png输出      

```
set term png
set output "a2b_tsg.png"
```
+ step4 
```
gnuplot a2b_tsg.gpl
```





+  gnuplot
1. Producing PDF (.pdf)   
````
gnuplot> set term pdf     (will produce PDF )
gnuplot> set output "printme.pdf" (any filename.pdf you want)
gnuplot> replot                   (recreates plot but you don't see it, goes to output file)
gnuplot> unset output  
gnuplot> unset term  
```
2. Producing png   (image file, similar to .jpg)   
```
gnuplot> set term png             (will produce .png output)
gnuplot> set output "printme.png" (output to any filename.png you want)
gnuplot> replot
gnuplot> unset term 
```
 

#  ipsumdump (not succ)


```
$ curl -O http://www.read.seas.harvard.edu/~kohler/ipsumdump/ipsumdump-1.86.tar.gz
$ tar -xzf ipsumdump-1.86.tar.gz
$ cd ipsumdump-1.86
$ ./configure --prefix=/usr/
$ make
$ sudo make install
$ sudo make clean
```

# killed

```
dmesg -T| grep -E -i -B100 'killed process'
```