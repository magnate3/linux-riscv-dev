#!/bin/bash
simple_switch -i 1@sw1_p1 -i 2@sw1_p2 -i 3@sw1_p3 ../int_mx.json  --thrift-port 9092  --nanolog ipc:///tmp/bm-1-log.ipc --device-id 1  &
simple_switch -i 1@sw2_p1 -i 2@sw2_p2 -i 3@sw2_p3 ../int_mx.json --thrift-port 9093 --nanolog ipc:///tmp/bm-2-log.ipc --device-id 2 &
simple_switch -i 1@sw3_p1 -i 2@sw3_p2 -i 3@sw3_p3 ../int_mx.json --thrift-port 9094 --nanolog ipc:///tmp/bm-3-log.ipc --device-id 3 &

sleep 5
simple_switch_CLI --thrift-port 9092 < ./s1-commands.txt

sleep 5
simple_switch_CLI --thrift-port 9093 < ./s2-commands.txt
sleep 5
simple_switch_CLI --thrift-port 9094 < ./s3-commands.txt 
