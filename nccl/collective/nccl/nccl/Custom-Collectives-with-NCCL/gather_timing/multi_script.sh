#!/bin/bash

# Define an array of node counts you want to test
node_counts=(1 2 4 8 16 32)

# node_counts=(2)

for nodes in "${node_counts[@]}"; do
  sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=nccl_timing_${nodes}_nodes
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=out_mpi_${nodes}_nodes.out
#SBATCH --error=err_mpi_${nodes}_nodes.err
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --account=lp16
#SBATCH --partition=normal

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MPICH_MALLOC_FALLBACK=1
export MPICH_GPU_SUPPORT_ENABLED=1

export NCCL_NET='AWS Libfabric'
export NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_COMPAT=0
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_DISABLE_HOST_REGISTER=1
export NCCL_TOPO_DUMP_FILE=system.txt

export NCCL_DEBUG=WARN

ulimit -s unlimited

srun --cpu-bind=socket bash -c '
export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID;
./main
'
EOT
done