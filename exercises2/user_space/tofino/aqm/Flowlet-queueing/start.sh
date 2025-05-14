#!/bin/bash

MACHINES=("nhpcc416@172.16.1.101" "nhpcc416@172.16.1.102" "nhpcc416@172.16.1.103" "nhpcc416@172.16.1.104")

echo "Starting worker on 10.0.0.1..."
ssh ${MACHINES[0]} "sudo ./worker"

echo "Executing setneigh.sh on all machines..."
for machine in "${MACHINES[@]}"; do
    ssh $machine "sudo bash setneigh.sh"
done

echo "Starting iperf3 servers on 10.0.0.4..."
ssh ${MACHINES[3]} "nohup iperf3 -s -p 8010 > /tmp/iperf_8010.log 2>&1 &"
ssh ${MACHINES[3]} "nohup iperf3 -s -p 8011 > /tmp/iperf_8011.log 2>&1 &"
ssh ${MACHINES[3]} "nohup iperf3 -s -p 8012 > /tmp/iperf_8012.log 2>&1 &"

ssh ${MACHINES[0]} "ping 10.0.0.4 -c 2"
ssh ${MACHINES[1]} "ping 10.0.0.4 -c 2"
ssh ${MACHINES[2]} "ping 10.0.0.4 -c 2"

echo "Starting first iperf test (10.0.0.1 -> 10.0.0.4:8010 for 90s)..."
ssh ${MACHINES[0]} "iperf3 -c 10.0.0.4 -p 8010 -t 90 > /tmp/iperf_client_8010.log 2>&1 &"

sleep 15

echo "Starting second iperf test (10.0.0.2 -> 10.0.0.4:8011 for 60s)..."
ssh ${MACHINES[1]} "iperf3 -c 10.0.0.4 -p 8011 -t 60 > /tmp/iperf_client_8011.log 2>&1 &"

sleep 15

echo "Starting third iperf test (10.0.0.3 -> 10.0.0.4:8012 for 30s)..."
ssh ${MACHINES[2]} "iperf3 -c 10.0.0.4 -p 8012 -t 30 > /tmp/iperf_client_8012.log 2>&1 &"

echo "Waiting for all tests to complete..."
sleep 65

echo "Killing all iperf3 processes..."
for machine in "${MACHINES[@]}"; do
    ssh $machine "sudo pkill iperf3"
done

echo "All tests completed and iperf3 processes terminated."
