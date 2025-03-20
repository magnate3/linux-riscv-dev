
# Implementing HULA

[Hula-hoop](https://github.com/rachitnigam/Hula-hoop)   
[HULA_with_CLB](https://github.com/drobinkent/P4TE/blob/815797e85d8b848c29eb399a179bbcf9a3685e82/HULA_with_CLB/CLB/FCTAnalyzer.py)   

[fattree45_hula](https://github.com/alex1230608/contra/tree/7d9c39320c717decc63642f35ed067175c082c92/P4-mininet-cluster/fattree45_hula)   

## topo


```
python3 p4apprunner.py  --build-dir ./build/    --manifest  ./p4app.json    app multiswitch
```

## test
+ step1   ****python3 generatehula.py****    
 
+  step2  ping     
![images](hula2.png)

## Introduction

The objective of this exercise is to implement a simplified version of 
[HULA](http://web.mit.edu/anirudh/www/hula-sosr16.pdf).
In contrast to ECMP, which selects the next hop randomly, HULA load balances
the flows over multiple paths to a destination ToR based on queue occupancy
of switches in each path. Thus, it can use the whole bisection bandwidth.
To keep the example simple, we implement it on top of source routing exercise.

Here is how HULA works:
- Each ToR switch generates a HULA packet to each other ToR switch
  to probe the condition of every path between the source and the destination ToR.
  Each HULA packet is forwarded to the destination ToR (forward path), collects the maximum 
  queue length it observes while being forwarded, and finally delivers that information
  to the destination ToR. Based on the congestion information collected via probes, 
  each destination ToR then can maintain the current best path (i.e., least congested path)
  from each source ToR. To share the best path information with the source ToRs so that 
  the sources can use that information for new flows, the destination ToRs notify
  source ToRs of the current best path by returning the HULA probe back to the source 
  ToR (reverse path) only if the current best path changes. The probe packets include
  a HULA header and a list of ports for source routing. We describe the elements of HULA header later.
- In the forward path:
  - Each hop updates the queue length field in the hula header if the local queue depth observed by
  the HULA packet is larger than maximum queue depth recorded in the probe packet. Thus when 
  the packet reaches the destination ToR, queue length field will be the maximum observed queue length 
  on the forward path.
  - At destination ToR, 
    1. find the queue length of current best path from the source ToR.
    2. if the new path is better, update the queue length and best path and return
     the HULA probe to the source path. This is done by setting the direction field
     in the HULA header and returning the packet to the ingress port.
    3. if the probe came through the current best path, the destination ToR just updates
     the existing value. This is needed to know if the best path got worse and hence allow 
     other paths to replace it later. It is inefficient to save the whole path ID 
     (i.e., sequence of switch IDs) and compare it in the data plane; 
     note, P4 doesn't have a loop construct. Instead, we keep a 32 bit digest of a 
     path in the HULA header. Each destination ToR only saves and compares the 
     digest of the best path along with its queue length.
     The `hula.digest` field is set by source ToR upon creating the HULA packet
     and does not change along the path.
- In the reverse path:
  - Each hop will update the "routing next hop" to the destination ToR based on the port
   it received the HULA packet on (as it was the best path). Then it forwards the packet
   to the next hop in reverse path based on source routing.
  - Source ToR also drops the packet.
- Now for each data packet,
  - Each hop hashes the flow header fields and looks into a "flow table".
  - If it doesn't find the next hop for the flow, looks into "routing next hop" to 
    find the next hop for destination ToR. We assume each ToR serves a /24 IP address.
    The switch also updates the "flow table". "flow table" prevents the path of a flow to change
    in order to avoid packet re-ordering and path oscilation during updating next hops.
  - Otherwise, each hop just uses the next hop.

Your switch will have multiple tables, which the control plane will
populate with static rules. We have already defined
the control plane rules, so you only need to implement the data plane
logic of your P4 program.

> **Spoiler alert:** There is a reference solution in the `solution`
> sub-directory. Feel free to compare your implementation to the reference.


## Step 1: Run the (incomplete) starter code

The directory with this README also contains a skeleton P4 program,
`hula.p4`, which initially drops all packets. Your job (in the next
step) will be to extend it to properly update HULA packets and forward data packets.

Before that, let's compile the incomplete `hula.p4` and bring up a
switch in Mininet to test its behavior.

1. In your shell, run:
   ```bash
   ./run.sh
   ```
   This will:
   * compile `hula.p4`, and
   * start a Mininet instance with three ToR switches (`s1`, `s2`, `s3`)
     and two spine switches ( `s11`, `s22`).
   * The hosts (`h1`, `h2`, `h3`) are assigned IPs of `10.0.1.1`, `10.0.2.2`, and `10.0.3.3`.

2. You should now see a Mininet command prompt. Just ping `h2` from `h1`:
   ```bash
   mininet> h1 ping h2
   ```
It doesn't work as no path is set.

3. Type `exit` to close the Mininet command line.

The message was not received because each switch is programmed with
`hula.p4`, which drops all data packets. Your job is to extend
this file.

### A note about the control plane

P4 programs define a packet-processing pipeline, but the rules governing packet
processing are inserted into the pipeline by the control plane.  When a rule
matches a packet, its action is invoked with parameters supplied by the control
plane as part of the rule.

In this exercise, the control plane logic has already been implemented.  As
part of bringing up the Mininet instance, the `run.sh` script will install
packet-processing rules in the tables of each switch.  These are defined in the
`sX-commands.txt` files, where `X` corresponds to the switch number.

**Important:** A P4 program also defines the interface between the switch
pipeline and control plane.  The `sX-commands.txt` files contain lists of
commands for the BMv2 switch API. These commands refer to specific tables,
keys, and actions by name, and any changes in the P4 program that add or rename
tables, keys, or actions will need to be reflected in these command files.

## Step 2: Implement Hula

The `hula.p4` file contains a skeleton P4 program with key pieces of
logic replaced by `TODO` comments. These should guide your
implementation---replace each `TODO` with logic implementing the missing piece.

A complete `hula.p4` will contain the following components:

1. Header type definitions for Ethernet (`ethernet_t`), Hula (`hula_t`),
   Source Routing (`srcRoute_t`), IPv4 (`ipv4_t`), UDP(`udp_t`).
2. Parsers for the above headers.
3. Registers:
  - `srcindex_qdepth_reg`: At destination ToR saves queue length of the best path
     from each Source ToR
  - `srcindex_digest_reg`: At destination ToR saves the digest of the best path
     from each Source ToR
  - `dstindex_nhop_reg`: At each hop, saves the next hop to reach each destination ToR
  - `flow_port_reg`: At each hop saves the next hop for each flow
4. `hula_fwd table`: looks at the destination IP of a HULA packet. If it is the destination ToR,
   it runs `hula_dst` action to set `meta.index` field based on source IP (source ToR).
   The index is used later to find queue depth and digest of current best path from that source ToR.
   Otherwise, this table just runs `srcRoute_nhop` to perform source routing.
5. `hula_bwd` table: at revere path, updates next hop to the destination ToR using `hula_set_nhop`
action. The action updates `dstindex_nhop_reg` register.
6. `hula_src` table checks the source IP address of a HULA packet in reverse path.
   if this switch is the source, this is the end of reverse path, thus drop the packet.
   Otherwise use `srcRoute_nhop` action to continue source routing in the reverse path.
7. `hula_nhop` table for data packets, reads destination IP/24 to get an index.
   It uses the index to read `dstindex_nhop_reg` register and get best next hop to the 
   destination ToR.
8. dmac table just updates ethernet destination address based on next hop.
9. An apply block that has the following logic:
  * If the packet has a HULA header
    * In forward path (`hdr.hula.dir==0`):
      * Apply `hula_fwd` table to check if it is the destination ToR or not
      * If this switch is the destination ToR (`hula_dst` action ran and 
      set the `meta.index` based on the source IP address):
        * read `srcindex_qdepth_reg` for the queue length of
       the current best path from the source ToR
        * If the new queue length is better, update the entry in `srcindex_qdepth_reg` and
       save the path digest in `srcindex_digest_reg`. Then return the HULA packet to the source ToR
       by sending to its ingress port and setting `hula.dir=1` (reverse path)
      * else, if this HULA packet came through current best path (`hula.digest` is equal to 
       the value in `srcindex_digest_reg`), update its queue length in `srcindex_qdepth_reg`.
       In this case we don't need to send the HULA packet back, thus drop the packet.
    * in reverse path (`hdr.hula.dir==1`):
      * apply `hula_bwd` to update the HULA next hop to the destination ToR
      * apply `hula_src` table to drop the packet if it is the source ToR of the HULA packet
  * If it is a data packet
    * compute the hash of flow
    * **TODO** read nexthop port from `flow_port_reg` into a temporary variable, say `port`. 
    * **TODO** If no entry found (`port==0`), read next hop by applying `hula_nhop` table.
     Then save the value into `flow_port_reg` for later packets.
    * **TODO** if it is found, save `port` into `standard_metadata.egress_spec` to finish routing.
    * apply `dmac` table to update `ethernet.dstAddr`. This is necessary for the links that send packets
    to hosts. Otherwise their NIC will drop packets.
  * udpate TTL
5. **TODO:** An egress control that for HULA packets that are in forward path (`hdr.hula.dir==0`)
   compares `standard_metadata.deq_qdepth` to `hdr.hula.qdepth` 
     in order to save the maximum in `hdr.hula.qdepth`
7. A deparser that selects the order in which fields inserted into the outgoing
   packet.
8. A `package` instantiation supplied with the parser, control, checksum verification and
   recomputation  and deparser.
多播处理   
```
def mcast_grp_command(mcast_id, port_ids, handle_id):
    port_seq = " ".join(str(e) for e in port_ids)
    create = "mc_mgrp_create " + str(mcast_id)
    node = "mc_node_create 0 " + port_seq
    assoc = "mc_node_associate " + str(mcast_id) + " " + str(handle_id)
    return create + "\n" + node + "\n" + assoc
```

## Step 3: Run your solution

1. Run Mininet same as Step 1

2. Open a separate terminal, go to `exercises/hula`, and run `sudo ./generatehula.py`. 
   This python script makes each ToR switch generate one HULA probe for each other ToR and 
   through each separate forward path. For example, `s1` first probes `s2` via `s11` and then via `s22`. 
   Then `s1` probes `s3` again first via `s11` and then via `s22`. `s2` does the same thing to probe
   paths to `s1` and `s3`, and so does `s3`.

3. Now run `h1 ping h2`. The ping should work if you have completed the ingress control block in `hula.p4`.
Note at this point, every ToR considers all paths are equal because there isn't any congestion in the network.
 
Now we are going to test a more complex scenario.

We first create two iperf sessions: one from `h1` to `h3`, and the other from `h2` to `h3`. 
Since both `s1` and `s2` currently think their best paths to `s3` should go through `s11`,
the two connections will use the same spine switch (`s11`). Note we throttled the 
links from the spine switches to `s3` down to 1Mbps. Hence, each of the two connections 
achieves only ~512Kbps. Let's confirm this by taking the following steps.

1. open a terminal window on `h1`, `h2` and `h3`:
```bash
xterm h1 h2 h3
```
2. start iperf server at `h3`
```bash
iperf -s -u -i 1
```
3. run iperf client at `h1`
```bash
iperf -c 10.0.3.3 -t 30 -u -b 2m
```
4. run iperf client in `h2`. try to do step 3 and 4 simultaneously.
```bash
iperf -c 10.0.3.3 -t 30 -u -b 2m
```
While the connections are running, watch the iperf server's output at `h3`.
Although there are two completely non-overlapping paths for `h1` and `h2` to reach `h3`,
both `h1` and `h2` end up using the same spine, and hence the aggregate 
throughput of the two connections is capped to 1Mbps. 
You can confirm this by watching the performance of each connection.


Our goal is allowing the two connections to use two different spine switches and hence achieve
1Mbps each. We can do this by first causing congestion on one of the spines. More specifically
we'll create congestion at the queue in `s11` facing the link `s11-to-s3` by running a 
long-running connection (an elephant flow) from `s1` to `s3` through `s11`. 
Once the queue builds up due to the elephant, then we'll let `s2` generate HULA probes 
several times so that it can learn to avoid forwarding new flows destined to `s3` through `s11`. 
The following steps achieve this.

1. open a terminal window on `h1`, `h2` and `h3`. (By the way, if you have already closed mininet,
you need to re-run the mininet test and run `generatehula.py` first, to setup initial routes)
```bash
xterm h1 h2 h3
```
2. start iperf server at `h3`
```bash
iperf -s -u -i 1
```
3. create a long-running full-demand connection from `h1` to `h3` through `s11`. 
you can do this by running the following at `h1`
```bash
iperf -c 10.0.3.3 -t 3000 -u -b 2m
```
4. outside mininet (in a separate terminal), go to `exercises/hula`, and run the following several (5 to 10) times
```bash
sudo ./generatehula.py
```
This should let `s2` know that the path through `s11` to `s3` is congested and
the best path is now through the uncongested spine, `s22`.
5. Now, run iperf client at `h2`
```bash
iperf -c 10.0.3.3 -t 30 -u -b 2m
```
You will be able to confirm both iperf sessions achieve 1Mbps because they go through two different spines.

### Food for thought
* how can we implement flowlet routing (as opposed to flow routing) say based on the timestamp of packets
* in the ingress control logic, the destination ToR always sends a HULA packet 
back on the reverse path if the queue length is better. But this is not necessary
if it came from the best path. Can you improve the code?
* the hula packets on the congested path may get dropped or extremely delayed,
thus the destination ToR would not be aware of the worsened condition of the current best path.
A solution could be that the destination ToR uses a timeout mechanism to ignore the current best path
if it doesn't receive a hula packet through it for a long time.
How can you implement this inside dataplane? 

### Troubleshooting

There are several ways that problems might manifest:

1. `hula.p4` fails to compile.  In this case, `run.sh` will report the
error emitted from the compiler and stop.

2. `hula.p4` compiles but does not support the control plane rules in
the `sX-commands.txt` files that `run.sh` tries to install using the BMv2 CLI.
In this case, `run.sh` will report these errors to `stderr`.  Use these error
messages to fix your `hula.p4` implementation.

3. `hula.p4` compiles, and the control plane rules are installed, but
the switch does not process packets in the desired way.  The
`build/logs/<switch-name>.log` files contain trace messages describing how each
switch processes each packet.  The output is detailed and can help pinpoint
logic errors in your implementation.
The `build/<switch-name>-<interface-name>.pcap` also contains the pcap of packets on each
interface. Use `tcpdump -r <filename> -xxx` to print the hexdump of the packets.

#### Cleaning up Mininet

In the latter two cases above, `run.sh` may leave a Mininet instance running in
the background. Use the following command to clean up these instances:

```bash
mn -c
```

## Next Steps

Congratulations, your implementation works!
