#
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#


.PHONY: all clean


TARGETS=src
#NCCL_DIR=/workspace/NCCL_GP
NCCL_DIR=/workspace/nccl-latest
#NCCL_DIR=/workspace/py-nccl

build:
	mpicxx  -g  rome_models.cc      -I$(NCCL_DIR)/build/include  -I$(NCCL_DIR)/fake_cuda/include  -I$(NCCL_DIR)/src/include -I$(NCCL_DIR)/src/graph  -I$(NCCL_DIR)/src/include/plugin -ldl -L$(NCCL_DIR)/build/lib -lnccl -o rome_models 
clean:
	rm   rome_models
