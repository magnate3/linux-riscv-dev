#!/bin/bash

# Usage ./apps/kill_traffic.sh
#   Kill the traffic generator background processes

sudo killall -9 "python apps/memcached_client.py" 2>/dev/null
sudo killall -9 memcached 2>/dev/null
sudo killall -9 "traffic_sender" 2>/dev/null
sudo killall -9 "traffic_receiver" 2>/dev/null
