#!/bin/bash


#SBATCH --job-name=nccl_timing
#SBATCH --nodes=4  # The number of nodes, you will be currently limited to 1
#SBATCH --ntasks-per-node=4  # The number of (MPI) processes per node. Your total number of processes will be this number times the number of nodes above.
#SBATCH --cpus-per-task=4  # The number of CPU cores per (MPI) process.
#SBATCH --time=00:15:00  # The maximum duration your job will run in hours:minutes:seconds. It will be automatically killed if it takes longer than that. For your tests, 5-10 minutes should be enough.
#SBATCH --output=out_mpi.out # Name of the output file (whatever your application prints in STDOUT).
#SBATCH --error=err_mpi.err # Name of the error file (whatever your application prints in STDERR).
#SBATCH --exclusive  # Use that if you want exclusive access to a node.
#SBATCH --gres=gpu:4
#SBATCH --account=lp16
#SBATCH --partition=debug

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Not very important for you, but it sets the number of OpenMP threads (per process) equal to the requested number of CPU cores (per process)
export MPICH_MALLOC_FALLBACK=1
export MPICH_GPU_SUPPORT_ENABLED=1

export NCCL_NET='AWS Libfabric'
export NCCL_CROSS_NIC=1
# export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_COMPAT=0
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_DISABLE_HOST_REGISTER=1

# export NCCL_DEBUG=INFO

ulimit -s unlimited

#srun ./main  # Note that you use srun instead of mpirun/mpiexec and that you don't (or you don't have to) set the number of processes. This is already taken from the SBATCH parameters above


srun  --cpu-bind=socket bash -c '
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;
./main 
' 
