#!/bin/bash
source ~/.bashrc
$SDE/run_bfshell.sh -f add-port.txt 
$SDE/run_bfshell.sh -b $PWD/drop_prob_mapping.py 
$SDE/run_bfshell.sh -b $PWD/mcast_table.py 
$SDE/run_bfshell.sh -b $PWD/mod_mac.py 
#$SDE/run_bfshell.sh -b $PWD/meter_conf.py 
$SDE/run_bfshell.sh -b $PWD/porttable.py
