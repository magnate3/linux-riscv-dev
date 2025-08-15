# In-Band Telemetry implementation in P4

This repository contains the implementation of [INT-MX and INT-MD](https://p4.org/p4-spec/docs/INT_v2_1.pdf) (version 2.1) and the associated generation of [telemetry report](https://p4.org/p4-spec/docs/telemetry_report_v2_0.pdf) packets (version 2.0).
The implementation was created using P4's bmv2 architecture.

## Requirements
- [bmv2](https://github.com/p4lang/behavioral-model)
- [p4c](https://github.com/p4lang/p4c)

## Usage
Clone this repository:
```sh
git clone https://github.com/ElhNoah/INT.git
```

Compile the desired INT mode:
```sh
cd INT
p4c INT-MD/int_md.p4    # If you want to use INT-MD mode
p4c INT-MX/int_mx.p4    # If you want to use INT-MX mode
```
Install the p4 program on the desired interfaces. The command below demonstrates how to associate specific interfaces (e.g., eth0, eth1) with particular ports (e.g., port 0, port 1). You can easily add or remove interfaces and map them to whichever ports you prefer by adjusting the -i flags:
```sh
simple_switch -i 0@eth0 -i 1@eth1 int_md.json &  # Run the INT-MD program with port 0 mapped to eth0 and port 1 to eth1
simple_switch -i 0@eth0 -i 1@eth1 int_mx.json &  # Run the INT-MX program with port 0 mapped to eth0 and port 1 to eth1
```
### Control plane configuration
Once the P4 program is running, open a new terminal to configure the port used for sending telemetry packets:
```sh
simple_switch_CLI <<< "mirroring_add 500 PORT" # Replace 'PORT' with the desired egress port number for telemetry packets
```
Next, you can populate and manage the program's tables through the CLI by running:
```sh
tools/runtime_CLI.py --thrift-port 9090
```
To explore the available commands, type `help` in the CLI. This will display a list of supported commands, including:
```sh
table_set_default <table name> <action name> <action parameters>
table_add <table name> <action name> <match fields> => <action parameters> [priority]
table_delete <table name> <entry handle>
```

## Useful INT tools
- [INTCollector](https://github.com/ElhNoah/INTCollector) : A high-performance collector to process INT reports and send data to a database server.