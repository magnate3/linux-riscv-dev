#!/bin/bash
echo 'Loading modules...'
module load cuda/12.6.2
module load nccl/2.22.3-1
module load cray-mpich/8.1.30
module load aws-ofi-nccl/git.v1.9.2-aws_1.9.2
