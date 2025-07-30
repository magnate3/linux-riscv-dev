
# 编译
```
/work/NCCL_GP

export NCCL_ROOT_DIR=/work/NCCL_GP
export CUDA_LIB=$NCCL_ROOT_DIR/fake_cuda/lib
export CUDA_INC=$NCCL_ROOT_DIR/fake_cuda/include
export LD_LIBRARY_PATH=$NCCL_ROOT_DIR/fake_cuda/lib:$NCCL_ROOT_DIR/build/lib
export NCCL_TOPO_FILE=$NCCL_ROOT_DIR/topo/nvlink_5GPU.xml
export NCCL_GRAPH_DUMP_FILE=$NCCL_ROOT_DIR/topo/graph_dump.xml
export GPU_DEV_NUM=5
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
```
## 编译命令

**Note:** 编译前确保 run.sh 以及  src/collectivates/device/gen_rules.sh 两个脚本有可执行权限，使用如下命令增加可执行权限。

+ 1 运行依赖脚本
```bash
chmod +x ./run.sh
chmod +x ./src/collectives/device/gen_rules.sh
```

+ 2 更改so函数可见性
makefiles/common.mk    
```
CXXFLAGS   := -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) -fPIC -fvisibility=hidden \
              -Wall -Wno-unused-function -Wno-sign-compare -std=c++11 -Wvla \
              -I $(CUDA_INC) \
              $(CXXFLAGS)
```
改为-fvisibility=default    

src/Makefile添加-fvisibility=default
```
$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVICELIB)
        @printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
        mkdir -p $(LIBDIR)
        $(CXX) $(CXXFLAGS) -fvisibility=default  -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(DEVICELIB) $(LDFLAGS)
        ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
        ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)
```

3. 直接运行脚本
```bash
./run.sh
```

3. 或者使用命令
```bash
make -j4 DEBUG=1 TRACE=1 VERBOSE=1 NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

```
make[1]: Entering directory '/work/NCCL_GP/test'
g++ -g test_main.cpp -I../build/include  -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -o test_main
make[1]: Leaving directory '/work/NCCL_GP/test'
```


# mpi


```
 cat ~/.bashrc 
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/rdma-bench/NCCL_GP/build/lib/
```

```
export NCCL_SOCKET_IFNAME=eno8303
```

```
ompi_info -param all all | grep pml
                 MCA pml: v (MCA v2.1.0, API v2.0.0, Component v4.1.4)
                 MCA pml: cm (MCA v2.1.0, API v2.0.0, Component v4.1.4)
                 MCA pml: monitoring (MCA v2.1.0, API v2.0.0, Component v4.1.4)
                 MCA pml: ucx (MCA v2.1.0, API v2.0.0, Component v4.1.4)
                 MCA pml: ob1 (MCA v2.1.0, API v2.0.0, Component v4.1.4)
      MCA pml monitoring: ---------------------------------------------------
      MCA pml monitoring: performance "pml_monitoring_flush" (type: string, class: generic)
```