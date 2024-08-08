
# 40G

+ server
```
root@recv:~# iperf -s 10.10.15.135 -p  9999
iperf: ignoring extra argument -- 10.10.15.135
------------------------------------------------------------
Server listening on TCP port 9999
TCP window size:  128 KByte (default)
------------------------------------------------------------
[  1] local 10.10.15.135 port 9999 connected with 10.10.15.134 port 39454
[  2] local 10.10.15.135 port 9999 connected with 10.10.15.134 port 39462
[  3] local 10.10.15.135 port 9999 connected with 10.10.15.134 port 39468
[  4] local 10.10.15.135 port 9999 connected with 10.10.15.134 port 39480
[  5] local 10.10.15.135 port 9999 connected with 10.10.15.134 port 39496
[  6] local 10.10.15.135 port 9999 connected with 10.10.15.134 port 39508
[  7] local 10.10.15.135 port 9999 connected with 10.10.15.134 port 39512
[  8] local 10.10.15.135 port 9999 connected with 10.10.15.134 port 39510
[ ID] Interval       Transfer     Bandwidth
[  8] 0.0000-10.0014 sec  4.71 GBytes  4.04 Gbits/sec
[  2] 0.0000-10.0023 sec  6.22 GBytes  5.35 Gbits/sec
[  4] 0.0000-10.0037 sec  5.15 GBytes  4.42 Gbits/sec
[  1] 0.0000-10.0073 sec  4.60 GBytes  3.95 Gbits/sec
[  3] 0.0000-10.0030 sec  5.36 GBytes  4.60 Gbits/sec
[  5] 0.0000-10.0031 sec  5.83 GBytes  5.01 Gbits/sec
[  6] 0.0000-10.0066 sec  5.73 GBytes  4.92 Gbits/sec
[  7] 0.0000-10.0065 sec  5.29 GBytes  4.54 Gbits/sec
[SUM] 0.0000-10.0119 sec  42.9 GBytes  36.8 Gbits/sec
```

+ client   



```
root@gen:~# iperf -V -c 2008::5 -p 9999 -t 10 -P8
------------------------------------------------------------
Client connecting to 2008::5, TCP port 9999
TCP window size: 85.0 KByte (default)
------------------------------------------------------------
[  1] local 2008::4 port 39454 connected with 2008::5 port 9999
[  2] local 2008::4 port 39462 connected with 2008::5 port 9999
[  3] local 2008::4 port 39468 connected with 2008::5 port 9999
[  6] local 2008::4 port 39496 connected with 2008::5 port 9999
[  4] local 2008::4 port 39480 connected with 2008::5 port 9999
[  5] local 2008::4 port 39508 connected with 2008::5 port 9999
[  8] local 2008::4 port 39512 connected with 2008::5 port 9999
[  7] local 2008::4 port 39510 connected with 2008::5 port 9999
[ ID] Interval       Transfer     Bandwidth
[  1] 0.0000-10.0066 sec  4.60 GBytes  3.95 Gbits/sec
[  3] 0.0000-10.0068 sec  5.36 GBytes  4.60 Gbits/sec
[  7] 0.0000-10.0058 sec  4.71 GBytes  4.04 Gbits/sec
[  4] 0.0000-10.0062 sec  5.15 GBytes  4.42 Gbits/sec
[  6] 0.0000-10.0068 sec  5.83 GBytes  5.01 Gbits/sec
[  2] 0.0000-10.0061 sec  6.22 GBytes  5.34 Gbits/sec
[  8] 0.0000-10.0232 sec  5.29 GBytes  4.53 Gbits/sec
[  5] 0.0000-10.0232 sec  5.73 GBytes  4.91 Gbits/sec
[SUM] 0.0000-10.0001 sec  42.9 GBytes  36.8 Gbits/sec
[ CT] final connect times (min/avg/max/stdev) = 0.142/0.234/0.270/0.039 ms (tot/err) = 8/0
root@gen:~# 
```

```
root@gen:~# ps -elf | grep iperf
0 S root        8530    8295 99  80   0 - 171821 futex_ 09:36 pts/1   00:00:08 iperf -V -c 2008::5 -p 9999 -t 20 -P8
0 S root        8541    8405  0  80   0 -  3075 pipe_r 09:36 pts/2    00:00:00 grep --color=auto iperf
root@gen:~#  ps -T -p  8530
    PID    SPID TTY          TIME CMD
   8530    8530 pts/1    00:00:00 iperf
   8530    8531 pts/1    00:00:00 iperf
   8530    8532 pts/1    00:00:02 iperf
   8530    8533 pts/1    00:00:02 iperf
   8530    8534 pts/1    00:00:03 iperf
   8530    8535 pts/1    00:00:02 iperf
   8530    8536 pts/1    00:00:03 iperf
   8530    8537 pts/1    00:00:03 iperf
   8530    8538 pts/1    00:00:03 iperf
   8530    8539 pts/1    00:00:02 iperf
root@gen:~# 
```

12个线程    
```
root@gen:~# iperf -V -c 2008::5 -p 9999 -t 10 -P12
[  7] local 2008::4 port 54744 connected with 2008::5 port 9999
[  3] local 2008::4 port 54696 connected with 2008::5 port 9999
[  6] local 2008::4 port 54742 connected with 2008::5 port 9999
[  5] local 2008::4 port 54724 connected with 2008::5 port 9999
[  1] local 2008::4 port 54712 connected with 2008::5 port 9999
[ 12] local 2008::4 port 54794 connected with 2008::5 port 9999
[  8] local 2008::4 port 54754 connected with 2008::5 port 9999
[ 11] local 2008::4 port 54778 connected with 2008::5 port 9999
------------------------------------------------------------
Client connecting to 2008::5, TCP port 9999
TCP window size: 85.0 KByte (default)
------------------------------------------------------------
[  4] local 2008::4 port 54698 connected with 2008::5 port 9999
[  9] local 2008::4 port 54760 connected with 2008::5 port 9999
[ 10] local 2008::4 port 54762 connected with 2008::5 port 9999
[  2] local 2008::4 port 54714 connected with 2008::5 port 9999
[ ID] Interval       Transfer     Bandwidth
[  3] 0.0000-10.0075 sec  4.34 GBytes  3.73 Gbits/sec
[  6] 0.0000-10.0075 sec  4.19 GBytes  3.59 Gbits/sec
[ 10] 0.0000-10.0074 sec  3.04 GBytes  2.61 Gbits/sec
[ 11] 0.0000-10.0077 sec  3.23 GBytes  2.77 Gbits/sec
[  5] 0.0000-10.0238 sec  3.11 GBytes  2.67 Gbits/sec
[  1] 0.0000-10.0240 sec  3.86 GBytes  3.30 Gbits/sec
[  9] 0.0000-10.0242 sec  4.53 GBytes  3.88 Gbits/sec
[  7] 0.0000-10.0240 sec  3.33 GBytes  2.85 Gbits/sec
[  4] 0.0000-10.0076 sec  3.59 GBytes  3.08 Gbits/sec
[  8] 0.0000-10.0239 sec  4.01 GBytes  3.44 Gbits/sec
[  2] 0.0000-10.0241 sec  2.90 GBytes  2.48 Gbits/sec
[ 12] 0.0000-10.1856 sec  2.90 GBytes  2.45 Gbits/sec
[SUM] 0.0000-10.1677 sec  43.0 GBytes  36.4 Gbits/sec
[ CT] final connect times (min/avg/max/stdev) = 0.122/0.217/0.276/0.047 ms (tot/err) = 12/0
root@gen:~#
```