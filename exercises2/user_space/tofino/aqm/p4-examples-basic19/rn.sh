#!/bin/bash

sudo make stop
sudo make clean
sudo make run
terminal -e command
#simple_switch_CLI --thrift-port 9090 < rules.cmd
#atom --no-sandbox ./logs/s1.log
