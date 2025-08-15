#!/bin/bash

MACHINES=("nhpcc416@172.16.1.101" "nhpcc416@172.16.1.102" "nhpcc416@172.16.1.103" "nhpcc416@172.16.1.104")

cleanup() {
    echo "Killing all iperf3 processes..."
    for machine in "${MACHINES[@]}"; do
        ssh $machine "sudo pkill iperf3"
    done
    exit 1
}

trap cleanup EXIT SIGINT SIGTERM

echo "Starting worker on 10.0.0.1..."
ssh ${MACHINES[0]} "sudo ./worker"

echo "Executing setneigh.sh on all machines..."
for machine in "${MACHINES[@]}"; do
    ssh $machine "sudo bash setneigh.sh"
done

ssh ${MACHINES[0]} "sudo tc qdisc del dev eth8 root"
#ssh ${MACHINES[0]} "sudo tc qdisc add dev eth8 root netem delay 5ms"
ssh ${MACHINES[1]} "sudo tc qdisc del dev enp3s0f1 root"
#ssh ${MACHINES[1]} "sudo tc qdisc add dev enp3s0f1 root netem delay 5ms"
ssh ${MACHINES[2]} "sudo tc qdisc del dev enp3s0f1 root"
#ssh ${MACHINES[2]} "sudo tc qdisc add dev enp3s0f1 root netem delay 5ms"

echo "Starting iperf3 servers on 10.0.0.4..."
ssh ${MACHINES[3]} "bash run_iperf3s.sh > /dev/null 2>&1 &"

ssh ${MACHINES[0]} "ping 10.0.0.4 -c 2"
ssh ${MACHINES[1]} "ping 10.0.0.4 -c 2"
ssh ${MACHINES[2]} "ping 10.0.0.4 -c 2"

echo "Starting first iperf test (10.0.0.1 -> 10.0.0.4 for 90s)..."
ssh ${MACHINES[0]} "bash run_iperf3c.sh 8100 8114 90"

sleep 15

echo "Starting second iperf test (10.0.0.2 -> 10.0.0.4 for 60s)..."
#ssh ${MACHINES[1]} "bash run_iperf3c.sh 8200 8214 60"

sleep 15

echo "Starting third iperf test (10.0.0.3 -> 10.0.0.4 for 30s)..."
#ssh ${MACHINES[2]} "bash run_iperf3c.sh 8300 8314 30"

echo "Waiting for all tests to complete..."
sleep 65

echo "All tests completed and iperf3 processes terminated."
