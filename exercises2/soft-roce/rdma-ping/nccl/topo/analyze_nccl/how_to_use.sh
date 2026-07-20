#! /bin/bash


workdir=$(dirname $0)

python $workdir/nccl_graph_to_fig.py $workdir/nccl_graph.xml
python $workdir/nccl_topo_to_fig.py $workdir/nccl_topo.xml