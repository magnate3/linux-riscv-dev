#!/bin/bash
source ~/.bashrc
$SDE/run_bfshell.sh -f add-port.txt 
$SDE/run_bfshell.sh -b $PWD/drop_prob_mapping.py 
$SDE/run_bfshell.sh -b $PWD/mod_mac.py 
$SDE/run_bfshell.sh -b $PWD/flowtable.py 
$SDE/run_bfshell.sh -b $PWD/timer.py
