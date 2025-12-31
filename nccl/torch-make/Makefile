PYTORCH_PYTHON_PATH := $(shell python3 -c "import torch; print(torch.__file__)")
PYTORCH_PATH := $(shell dirname $(PYTORCH_PYTHON_PATH))
PYTORCH_VERSION := $(shell python3 -c "import torch; print(torch.__version__)")
PYTORCH_LIB_PATH := $(PYTORCH_PATH)/lib
PYTORCH_INCLUDE_PATH := $(PYTORCH_LIB_PATH)/include
#PYTORCH_INC := -L $(PYTORCH_LIB_PATH) -lc11_cuda -lcublas -lcublasLt -ltorch_python -ltorch_cuda_cu
PYTORCH_INC := -L $(PYTORCH_LIB_PATH) -lc10_cuda -lcublas -lcublasLt -ltorch_cuda_cu  -ltorch_python \
    -I $(PYTORCH_PATH)/include \
    -I $(PYTORCH_INCLUDE_PATH) \
    -I $(PYTORCH_INCLUDE_PATH)/TH \
    -I $(PYTORCH_INCLUDE_PATH)/THC

default : test_prune_gate test_limit test_assign test_counting

test_% : %.cu
	nvcc $< ../stream_manager.cpp -lcublas $(PYTORCH_INC) -o $@  
