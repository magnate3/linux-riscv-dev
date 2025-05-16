

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
enp61s0f1np1
```

```
 tshark -i enp61s0f1np1 -f "tcp port 9999" -w test.pcap -P -a duration:10
```