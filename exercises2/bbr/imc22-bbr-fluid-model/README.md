
# 依赖
```
#!/bin/bash

# System packages
apt-get install texlive-latex-recommended cm-super dvipng
apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
# Python3 packages
pip3 install matplotlib pandas pyroute2 colou

```

+ pox   

```
~$ git clone http://github.com/noxrepo/pox
~$ cd pox
~/pox$ git checkout dart
```


#   run_experiments.py


```
./run_experiments.py configs/cubic_model_checking.yaml 
```




# c) What is the average computed throughput of the TCP transfer?
- Upon running the simulation and analyzing the `.pcap` files for all the nodes, we can calculate the sum of the bytes received and divide by the simulation time to get the average throughput. It can be acheived by running the following commmand:
  ```bash
  tshark -r {file} -q -z io,stat,10,"SUM(frame.len)frame.len"
  ```
  where `{file}` is the `.pcap` file for the node. The output will be as follows:
  ```
    ========================================
    | IO Statistics                        |
    |                                      |
    | Duration: {duration} secs            |
    | Interval: {duration} secs            |
    |                                      |
    | Col 1: SUM(frame.len)frame.len       |
    |--------------------------------------|
    |                     |1        |      |
    | Interval            |   SUM   |      |
    |-------------------------------|      |
    | 0.000 <> {duration} | {bytes} |      |
    ========================================
    ```
    The average throughput can be calculated as `bytes / duration`. 

- This can be automated using the following comands in python:
    ```python
    files = ["Q1_tcp-example-0-0.pcap", "Q1_tcp-example-1-0.pcap", "Q1_tcp-example-1-1.pcap", "Q1_tcp-example-2-0.pcap"]
    throughput = []

    for file in files:
        # Parse the pcap file for throughput
        command = f"tshark -r {file} -q -z io,stat,10,\"SUM(frame.len)frame.len\""
        output = os.popen(command).read()
        # print(output) 
        output = output.split()
        # print(output)
        frame_len = float(output[-4])
        duration = float(output[-6])
        throughput.append(frame_len * 8 / duration / 1000000)

    for i, file in enumerate(files):
        print(f"Throughput for {file} is {throughput[i]} Mbps")
    ```
    
- The following are the average throughputs for the nodes:
    ```
    Throughput for Q1_tcp-example-0-0.pcap is 3.2745145403899723 Mbps
    Throughput for Q1_tcp-example-1-0.pcap is 3.26832 Mbps
    Throughput for Q1_tcp-example-1-1.pcap is 3.2656683146067413 Mbps
    Throughput for Q1_tcp-example-2-0.pcap is 3.261553830577118 Mbps
    ```