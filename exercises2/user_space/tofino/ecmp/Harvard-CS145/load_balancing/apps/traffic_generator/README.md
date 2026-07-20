# Traffic generator

This directory contains the source code of the traffic generator. We make our traffic generator a C++ program to improve its performance.

## Usage

To use the traffic generator, we need to build it

```
make
```

Then you can see two binary files in this directory: `traffic_receiver` and `traffic_sender`.

To run the traffic generator, we need to run the receiver first in a node, and run the traffic sender in another node.
In this course project, you are not required to directly run the traffic generator manually.
However, in case you need to run it for debugging, we provide the instructions as follows.

### Running the traffic receiver

```
./traffic_receiver --host [host name] --topofile [topology json filename] --protocol [tcp or udp]
```

Using this command, you can start the traffic receiver on the host defined by `host name`, with `tcp` or `udp` protocol, and you need to provide the `topology json filename` after running `p4run`.
The protocol argument is optional - by default we use tcp.

### Running the traffic sender

```
./traffic_sender --host [host name] --tracefile [trace filename] --topofile [topology json filename] --protocol [tcp or udp] --start_time [clock number] --logdir [directory of log files] --verbose [true or false]
```

Using this command, you can start the traffic sender on the host defined by `host name`, with `tcp` or `udp` protocol. The traffic sender will play the traffic defined the `trace filename`, and it will start to send the traffic at the time defined by the global `clock number` (you can use C++ `clock()` function or Python `time.clock()` method to generate the current clock number).
The location of the detailed traffic logs will be defined by the `directory of log files`.
You can also set `verbose` to `true` (default is `false`) to add more detailed logs.

