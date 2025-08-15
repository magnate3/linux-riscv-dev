#!/bin/bash

set -e

echo "" > iperf_logs_total.txt
for i in {1..50}
do
    echo {time $i}
    python apps/project5_send_traffic.py 5 $i >> iperf_logs_total.txt
done
echo Finished.
