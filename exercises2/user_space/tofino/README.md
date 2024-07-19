


```
 Current Dir: /sde/bf-sde-9.7.1/build/p4-build/tna_ports
   Executing: /sde/bf-sde-9.7.1/pkgsrc/p4-build/configure --prefix="/sde/bf-sde-9.7.1/install" --with-p4c="/sde/bf-sde-9.7.1/install/bin/bf-p4c" P4_PATH="/sde/bf-sde-9.7.1/pkgsrc/p4-examples/p4_16_programs/tna_ports/tna_ports.p4" P4_NAME="tna_ports" P4_PREFIX="tna_ports" P4_VERSION="p4-16" P4_ARCHITECTURE="tna" P4JOBS=8 P4FLAGS=" -g --verbose 2 --parser-timing-reports --display-power-budget --create-graphs" --with-tofino P4PPFLAGS="" 
```

## tofino

3. **Bringing up ports and testing the program**

The first thing you probably want to do is bring up some ports. This can be done in the control script, but you can also do it manually from bfshell > ucli > pm:


```
bfshell> ucli
Starting UCLI from bf-shell 
Cannot read termcap database;
using dumb terminal settings.
bf-sde> pm
bf-sde.pm> ?
# ... list of commands (not shown)
```
pm has a bunch of commands. The most important ones are: 

``show [-a]`` -- show the currently configured ports. If run with -a, show all ports.

```
bf-sde.pm> show -a
-----+----+---+----+-------+----+---+---+---+--------+----------------+----------------+-
PORT |MAC |D_P|P/PT|SPEED  |FEC |RDY|ADM|OPR|LPBK    |FRAMES RX       |FRAMES TX       |E
-----+----+---+----+-------+----+---+---+---+--------+----------------+----------------+-
1/0  |23/0|128|3/ 0|-------|----|YES|---|---|--------|----------------|----------------|-
1/1  |23/1|129|3/ 1|-------|----|YES|---|---|--------|----------------|----------------|-
```
Important columns are: 
PORT -- the front panel port name. 
D_P -- the port's id from inside of P4. 
RDY -- is a cable detected? 
ADM -- is the port configured? 
OPR -- is the port down (DWN) or up (UP)?
Most of the other columns are self explanatory. LPBK is whether the port is configured as a loopback port. 

To configure a port, use port-add. To bring a configured port up, use port-enb.

`` port-add        <port_str> <speed (1G, 10G, 25G, 40G, 40G_NB, 50G(50G/50G-R2, 50G-R1), 100G(100G/100G-R4, 100G-R2),200G(200G/200G-R4, 200G-R8), 400G 40G_NON_BREAKABLE)> <fec (NONE, FC, RS)>``

``<port_str>`` is the port's front panel id. 100G ports can be split into 4 separate 25G or 10G ports, so each port id has two components: the physical port name and the id of the port's channel: ``<port id>/<channel id>``

So, for example, `1/0` and `2/0` are the first two ports in 100G mode. If you configure port 1 in 4x25G mode, you can also use ports `1/1`, `1/2`, and `1/3`. 

You configure each port individually. So, to bring up 1/0 - 1/3 in 10G mode with no FEC, use: 

```
bf-sde.pm> port-add 1/0 10G NONE
bf-sde.pm> port-add 1/1 10G NONE
bf-sde.pm> port-add 1/2 10G NONE
bf-sde.pm> port-add 1/3 10G NONE
bf-sde.pm> show                 
-----+----+---+----+-------+----+---+---+---+--------+----------------+----------------+-
PORT |MAC |D_P|P/PT|SPEED  |FEC |RDY|ADM|OPR|LPBK    |FRAMES RX       |FRAMES TX       |E
-----+----+---+----+-------+----+---+---+---+--------+----------------+----------------+-
1/0  |23/0|128|3/ 0|10G    |NONE|YES|DIS|DWN|  NONE  |               0|               0| 
1/1  |23/1|129|3/ 1|10G    |NONE|YES|DIS|DWN|  NONE  |               0|               0| 
1/2  |23/2|130|3/ 2|10G    |NONE|YES|DIS|DWN|  NONE  |               0|               0| 
1/3  |23/3|131|3/ 3|10G    |NONE|YES|DIS|DWN|  NONE  |               0|               0| 
```

Next, to bring up a configured port, say ``1/0``, use ``port-enb``: 
```
bf-sde.pm> port-add 1/0 1G NONE 
2024-07-18 09:07:09.452871 BF_PM ERROR - pm_port_valid_speed_and_channel_internal:711 Port validation failed for Dev-family: 0 dev : 0 d_p : 168 : 1/0 speed : 1g num-lanes: 1 error-msg: 1G not supported on this  port
Add failed Invalid arguments (3)

```

```
bf-sde.pm> port-add 1/0 10G NONE
bf-sde.pm> port-add 1/1 10G NONE
bf-sde.pm> port-enb 1/0
bf-sde.pm> port-enb 1/1
bf-sde.pm> show
-----+----+---+----+-------+----+--+--+---+---+---+--------+----------------+----------------+-
PORT |MAC |D_P|P/PT|SPEED  |FEC |AN|KR|RDY|ADM|OPR|LPBK    |FRAMES RX       |FRAMES TX       |E
-----+----+---+----+-------+----+--+--+---+---+---+--------+----------------+----------------+-
1/0  |29/0|168|3/40|10G    |NONE|Au|Au|NO |ENB|DWN|  NONE  |               0|               0| 
1/1  |29/1|169|3/41|10G    |NONE|Au|Au|NO |ENB|DWN|  NONE  |               0|               0| 
bf-sde.pm> 
```

At this point, once the ethernet autonegotiation completes the OPR column for an enabled port should change to "UP". 

Once you have a server connected to a port that's UP, you can test the program: send a packet into the switch from the server, the P4 program should send a copy of the exact same packet back. You should see the frames RX and frames TX counters increase in the CLI. 



## docker 
```
docker export -o  p4i-docker.tar 1f43ce1c21ba 
docker import p4i-docker.tar p4i-img
docker  run  --net=host  --cap-add=NET_ADMIN  -v /sde:/sde --name tofino  -it  tofino-img  bash
docker  run  --net=host  --cap-add=NET_ADMIN  -v /work/tofino:/sde --name tofino  -it  tofino-img  bash
docker  run  --net=host  --cap-add=SYS_ADMIN  --cap-add=NET_ADMIN  -v /sde:/sde --name tofino  -it  p4i-img  bash
```


```
docker run -d --rm --name p4i -v /sde:/sde -w /sde/bf-sde-9.7.1/build/ -p 3000:3000/tcp --init --cap-add CAP_SYS_ADMIN --cap-add CAP_NET_ADMIN p4i-img bash
```

# p4i


```
 xvfb-run ./p4i --disable-gpu --no-sandbox -w /work/tofino/bf-sde-9.7.1/build/
```


```
root@ubuntux86:# xvfb-run ./p4i --no-sandbox --disable-gpu -h
p4i [--working-dir, -w <dir>][--open, -o <manifest.json|archive.tar.bz2>, ...][--licence, -l <license.info>][--editor-path-command, -e <command_string>][--editor-files-command, -f <command_string>][--no-browser][--defaults]

Editor commands
Commands with a parameter of %p to launch a file or directory path in your editor. For opening files specifically, parameter %l is available to set line number.

  [--editor-path-command | -e] <command_string> 
    Example for VS Code: p4i -e "code %p"

  [--editor-files-command | -f] <command_string> 
    Example for VS Code: p4i -f "code -g %p:%l"
root@ubuntux86:# 
```

##  chrome_sandbox 
chown root:root chrome_sandbox &&  chmod 4755 chrome_sandbox 
```
root@ubuntux86:# xvfb-run ./chrome-sandbox --disable-gpu --no-sandbox --disable-setuid-sandbox
The setuid sandbox provides API version 1, but you need 0
Please read https://chromium.googlesource.com/chromium/src/+/master/docs/linux/suid_sandbox_development.md.

close: Bad file descriptor
Read on socketpair: Success
```

[一个用于在虚拟桌面下跑chrome的docker镜像](https://blog.csdn.net/socrates/article/details/140284135)
```
run --rm -it -v /work/tofino:/sde -v /run/dbus/:/run/dbus/ --workdir /workdir --cap-add=SYS_ADMIN --net=host  --cap-add=NET_ADMIN -p 9222:9222 --entrypoint "/bin/bash"  socrateslee/xvfb-chrome
```

```
./usr/bin/google-chrome-stable
./usr/bin/google-chrome
```
进入容器，切换成root,执行   
```
 xvfb-run  google-chrome --disable-gpu --no-sandbox --disable-setuid-sandbox --remote-debugging-port=9222  --remote-debugging-address=0.0.0.0
```

```
docker run --rm -it -v /work/tofino:/sde --workdir /sde --cap-add=SYS_ADMIN --net=host  --cap-add=NET_ADMIN -p 9222:9222 socrateslee/xvfb-chrome bash
```
```
docker run --rm -it -v /work/tofino:/sde --workdir /sde --cap-add=SYS_ADMIN --net=host  --cap-add=NET_ADMIN -p 9222:9222 socrateslee/xvfb-chrome --xvfb-run --remote-debugging-port=9222  --disable-gpu
```

export HTTP_PROXY=http://127.0.0.1:8123 
Environment="HTTPS_PROXY=http://127.0.0.1:8123"
#   Print out Tofino resource utilization statistics

**Pipeline resources:**
```
vagrant@cebinaevm:/cebinae/tofino_prototype$ sed -n 10,26p ./main/pipe/logs/mau.resources.log
| Stage Number | Exact Match Input xbar | Ternary Match Input xbar | Hash Bit | Hash Dist Unit | Gateway | SRAM | Map RAM | TCAM | VLIW Instr | Meter ALU | Stats ALU | Stash | Exact Match Search Bus | Exact Match Result Bus | Tind Result Bus | Action Data Bus Bytes | 8-bit Action Slots | 16-bit Action Slots | 32-bit Action Slots | Logical TableID |
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|      0       |           21           |            0             |   144    |       4        |    3    |  8   |    0    |  0   |     4      |     0     |     0     |   2   |           3            |           4            |        4        |           18          |         0          |          0          |          0          |        8        |
|      1       |           22           |            0             |   142    |       2        |    6    |  17  |    8    |  0   |     6      |     4     |     0     |   3   |           5            |           9            |        6        |           16          |         0          |          0          |          0          |        11       |
|      2       |           31           |            0             |   113    |       2        |    2    |  12  |    2    |  0   |     3      |     1     |     0     |   2   |           3            |           4            |        4        |           26          |         0          |          0          |          0          |        7        |
|      3       |           21           |            16            |    33    |       2        |    2    |  7   |    2    |  4   |     3      |     1     |     0     |   0   |           1            |           1            |        1        |           8           |         0          |          0          |          0          |        5        |
|      4       |           10           |            0             |    20    |       2        |    2    |  4   |    4    |  0   |     2      |     2     |     0     |   0   |           1            |           2            |        2        |           8           |         0          |          0          |          0          |        4        |
|      5       |           0            |            1             |    0     |       0        |    0    |  1   |    0    |  1   |     32     |     0     |     0     |   0   |           0            |           0            |        2        |           4           |         0          |          0          |          0          |        2        |
|      6       |           16           |            16            |    30    |       3        |    0    |  9   |    6    |  9   |     4      |     3     |     0     |   0   |           0            |           0            |        4        |           8           |         0          |          0          |          0          |        4        |
|      7       |           0            |            17            |    0     |       0        |    0    |  2   |    0    |  4   |     32     |     0     |     0     |   0   |           0            |           0            |        2        |           4           |         0          |          0          |          0          |        2        |
|      8       |           18           |            32            |    42    |       3        |    2    |  5   |    4    |  6   |     2      |     2     |     0     |   0   |           1            |           2            |        1        |           4           |         0          |          0          |          0          |        3        |
|      9       |           6            |            0             |    16    |       1        |    1    |  2   |    2    |  0   |     1      |     1     |     0     |   0   |           1            |           1            |        1        |           0           |         0          |          0          |          0          |        2        |
|      10      |           0            |            0             |    0     |       0        |    0    |  0   |    0    |  0   |     1      |     0     |     0     |   0   |           0            |           0            |        1        |           0           |         0          |          0          |          0          |        1        |
|      11      |           0            |            8             |    0     |       0        |    0    |  3   |    2    |  2   |     3      |     1     |     0     |   0   |           0            |           0            |        1        |           0           |         0          |          0          |          0          |        1        |
|              |                        |                          |          |                |         |      |         |      |            |           |           |       |                        |                        |                 |                       |                    |                     |                     |                 |
|    Totals    |          145           |            90            |   540    |       19       |    18   |  70  |    30   |  26  |     93     |     15    |     0     |   7   |           15           |           23           |        29       |           96          |         0          |          0          |          0          |        50       |```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

vagrant@cebinaevm:/cebinae/tofino_prototype$ sed -n 320,345p ./main/pipe/logs/phv_allocation_summary_3.log
+-------------------+-----------------+----------------+----------------------+---------------------+----------------+---------------------------+--------------------------+----------------+
|     MAU Group     | Containers Used |   Bits Used    | Bits Used on Ingress | Bits Used on Egress | Bits Allocated | Bits Allocated on Ingress | Bits Allocated on Egress | Available Bits |
+-------------------+-----------------+----------------+----------------------+---------------------+----------------+---------------------------+--------------------------+----------------+
|       B0-15       |   5 ( 31.2 %)   |  35 ( 27.3 %)  |     35 ( 27.3 %)     |     0 (   0  %)     |  83 ( 64.8 %)  |       83 ( 64.8 %)        |       0 (   0  %)        |      128       |
|      B16-31       |   2 ( 12.5 %)   |  12 ( 9.38 %)  |     0 (   0  %)      |    12 ( 9.38 %)     |  12 ( 9.38 %)  |        0 (   0  %)        |       12 ( 9.38 %)       |      128       |
|      B32-47       |   0 (   0  %)   |  0 (   0  %)   |     0 (   0  %)      |     0 (   0  %)     |  0 (   0  %)   |        0 (   0  %)        |       0 (   0  %)        |      128       |
|      B48-63       |   0 (   0  %)   |  0 (   0  %)   |     0 (   0  %)      |     0 (   0  %)     |  0 (   0  %)   |        0 (   0  %)        |       0 (   0  %)        |      128       |
+-------------------+-----------------+----------------+----------------------+---------------------+----------------+---------------------------+--------------------------+----------------+
|       H0-15       |   7 ( 43.8 %)   | 105 (  41  %)  |    105 (  41  %)     |     0 (   0  %)     | 105 (  41  %)  |       105 (  41  %)       |       0 (   0  %)        |      256       |
|      H16-31       |   5 ( 31.2 %)   |  66 ( 25.8 %)  |     0 (   0  %)      |    66 ( 25.8 %)     |  66 ( 25.8 %)  |        0 (   0  %)        |       66 ( 25.8 %)       |      256       |
|      H32-47       |   0 (   0  %)   |  0 (   0  %)   |     0 (   0  %)      |     0 (   0  %)     |  0 (   0  %)   |        0 (   0  %)        |       0 (   0  %)        |      256       |
|      H48-63       |   0 (   0  %)   |  0 (   0  %)   |     0 (   0  %)      |     0 (   0  %)     |  0 (   0  %)   |        0 (   0  %)        |       0 (   0  %)        |      256       |
|      H64-79       |   0 (   0  %)   |  0 (   0  %)   |     0 (   0  %)      |     0 (   0  %)     |  0 (   0  %)   |        0 (   0  %)        |       0 (   0  %)        |      256       |
|      H80-95       |   0 (   0  %)   |  0 (   0  %)   |     0 (   0  %)      |     0 (   0  %)     |  0 (   0  %)   |        0 (   0  %)        |       0 (   0  %)        |      256       |
+-------------------+-----------------+----------------+----------------------+---------------------+----------------+---------------------------+--------------------------+----------------+
|       W0-15       |  16 (  100 %)   | 504 ( 98.4 %)  |    504 ( 98.4 %)     |     0 (   0  %)     | 632 (  123 %)  |       632 (  123 %)       |       0 (   0  %)        |      512       |
|      W16-31       |  10 ( 62.5 %)   | 320 ( 62.5 %)  |     0 (   0  %)      |    320 ( 62.5 %)    | 353 ( 68.9 %)  |        0 (   0  %)        |      353 ( 68.9 %)       |      512       |
|      W32-47       |   0 (   0  %)   |  0 (   0  %)   |     0 (   0  %)      |     0 (   0  %)     |  0 (   0  %)   |        0 (   0  %)        |       0 (   0  %)        |      512       |
|      W48-63       |   0 (   0  %)   |  0 (   0  %)   |     0 (   0  %)      |     0 (   0  %)     |  0 (   0  %)   |        0 (   0  %)        |       0 (   0  %)        |      512       |
+-------------------+-----------------+----------------+----------------------+---------------------+----------------+---------------------------+--------------------------+----------------+
|   Usage for 8b    |   7 ( 10.9 %)   |  47 ( 9.18 %)  |     35 ( 6.84 %)     |    12 ( 2.34 %)     |  95 ( 18.6 %)  |       83 ( 16.2 %)        |       12 ( 2.34 %)       |      512       |
|   Usage for 16b   |  12 ( 12.5 %)   | 171 ( 11.1 %)  |    105 ( 6.84 %)     |    66 (  4.3 %)     | 171 ( 11.1 %)  |       105 ( 6.84 %)       |       66 (  4.3 %)       |      1536      |
|   Usage for 32b   |  26 ( 40.6 %)   | 824 ( 40.2 %)  |    504 ( 24.6 %)     |    320 ( 15.6 %)    | 985 ( 48.1 %)  |       632 ( 30.9 %)       |      353 ( 17.2 %)       |      2048      |
+-------------------+-----------------+----------------+----------------------+---------------------+----------------+---------------------------+--------------------------+----------------+
| Overall PHV Usage |  45 ( 20.1 %)   | 1042 ( 25.4 %) |    644 ( 15.7 %)     |    398 ( 9.72 %)    | 1251 ( 30.5 %) |       820 (  20  %)       |      431 ( 10.5 %)       |      4096      |
+-------------------+-----------------+----------------+----------------------+---------------------+----------------+---------------------------+--------------------------+----------------+
```
These statistics were used to create Table 3 as follows:
- Stages: (from mau.resources.log) The program uses resources in all stages 0 - 11
- PHV:    (from phv_allocation_summary_3.log) The cell (overall PHV usage, Bits Used) is 1042
- SRAM: (from mau.resources.log) The program uses 70 SRAM blocks. Each block is 16KB each, for a total of 1120KB.
- TCAM: (from mau.resources.log) The program uses 26 TCAM blocks. Each block is 1.28KB, for a total of <34KB.
- VLIWs: (from mau.resources.log) The program uses 93 VLIW instructions.
- Queues: Not shown here. The program configures 2 physical queues per port, on a 32-port switch this is 64 queues.


## Compiling a P4 program
bf-p4c can compile a P4 program that is written for TNA or V1 model.    

To compile a P4_16 program written for V1model, use [2]:    

bf-p4c --std p4-16 --arch v1model --p4runtime-files <prog>.p4info.pb.txt <prog>.p4   

# proj

[The Cheetah Load Balancer - NSLab @ KTH](https://github.com/cheetahlb)   
[AnotherKamila/sdn-loadbalancer](https://github.com/AnotherKamila/sdn-loadbalancer)   
[P4NAT](https://github.com/zhy1658858023/P4NAT/blob/main/ServiceNAT_tofino/ServiceNAT_tofino.p4)     
 

