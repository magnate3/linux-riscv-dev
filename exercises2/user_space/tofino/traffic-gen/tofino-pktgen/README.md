# Get started with Tofino Pktgen

This repo generates packets using the Tofino switch.

To compile the simple forwarding program:
```sh
$SDE/p4_build.sh -p <this_dir>/simple_forwarding.p4
```

Run in `screen` the following to run the data plane.
```sh
screen -r
$SDE/run_switchd.sh -p simple_forwarding
```

Set up the ports as follows
```sh
ucli
pm
port-add <physical port no.>/0 40G NONE
an-set -/- 2
port-enb -/-
exit
# To show RX/TX packets at the port
show
```

Simple forwarding rules
```sh
bfrt
bfrt.simple_forwarding.pipe.Ingress.forward.entry_with_send(ingress_port=<pktgen dev port>, port=<output dev port>).push()
```

To generate packets, run the control plane
```sh
python3 <this_dir>/generate.py
```

Setting to show traffic rate at each port
```sh
bf-sde.pm> rate-period 1
bf-sde.pm> rate-show
```

Acknowledgments to Kashish and Archit for contributing to the traffic generation code.
