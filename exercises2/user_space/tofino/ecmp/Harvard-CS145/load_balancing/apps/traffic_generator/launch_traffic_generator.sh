#!/bin/bash

set -e

script_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )
proj_root_dir=$( cd ${script_dir}/../.. >/dev/null 2>&1 && pwd )

base_time=$(expr `date +%s%N` / 1000 + 5000000)

tmux new-session -s traffic_generator -d
tmux selectp -t 0
tmux send-keys -t traffic_generator "sudo mx h1 ${proj_root_dir}/apps/traffic_generator/traffic_receiver --host h1 --protocol tcp" C-m
tmux splitw -h -p 50

sleep 1

tmux selectp -t 1
tmux send-keys -t traffic_generator "sudo mx h2 ${proj_root_dir}/apps/traffic_generator/traffic_sender --host h2 --protocol tcp --tracefile ${proj_root_dir}/apps/traffic_generator/trace.txt --start_time=$base_time" C-m

tmux attach -t traffic_generator
