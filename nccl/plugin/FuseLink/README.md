# Unet

Efficient GPU communication over multiple NICs.

## Note: The following describes how to run Fuselink. Necessary modifications are required to run UNet.
## Build

Hardware:
- RDMA capable NICs (required).
- GPU with GPU Direct RDMA (GDR) capability (required).
- NVLink interconnect (recommended).

Software:
- NCCL v2.19.4
- libibverbs
- CUDA >= 11.8 (To enable GDR and peer access)
- Nvidia driver version 550.120 (Other versions wait for testing)

Build commands

**Build NCCL**
```bash
make -j src.build CUDA_HOME=<your cuda home>
```

**Build FuseLink NCCL Plugin**
```bash
CUDA_HOME=<your cuda home> NCCL_BUILD_DIR=./nccl/build make fl
```

**Disable PCIe ACS to ensure fast NIC access**
```bash
bash scripts/disable_acs.sh
```

## Usage

Set `NCCL_NET_PLUGIN` environment to `fuselink` and expose `libnccl-net-fuselink.so` to `LD_LIBRARY_PATH`.

```bash
export LD_LIBRARY_PATH=<FuseLink Dir>/build/lib:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=fuselink
```

## Files

```
src
├── checks.h
├── cumem.cu
├── cumem.h                 necessary operations in CUDA memory
├── extern                  extern files borrowed from NCCL, may be removed in the future
│   ├── ibvcore.h
│   ├── ibvwrap.h
│   ├── nccl_net.h
│   ├── param.h
│   ├── socket.cc
│   ├── socket.h
│   ├── timer.h
│   └── utils.h
├── fuselink.cc             FuseLink memory structures
├── fuselink.h
├── monitor_main.cpp        FuseLink monitors for NIC idleness
├── monitor.cpp
├── monitor.h
├── plugin.cc               FuseLink integrations in NCCL
├── unet.cc                     UNET memory structures
├── unet.h                      
├── plugin_unet.cc              UNET integrations in NCCL
```

## Note for OSDI Artifact

We are actively addressing the additional experiments requested by the Shepherd. Some of the codebase remain under active development and are currently optimized for the specialized configurations of our testing environments.
