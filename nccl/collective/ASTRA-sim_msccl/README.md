# Overview
This repository presents a sample workflow of collective algorithm generation & simulation using the Chakra ET representation.

Users define custom collective algorithms using the MSCCLang DSL, where the resulting collective algorithm is represented in Chakra ET.
This Chakra ET representation of the *collective algorithm* is fed into the ASTRA-sim distributed ML simulator, along with the *workload* represented in Chakra ET.

A detailed discussion on the background of this work and motivation for a common collective algorithm representation is provided in our paper, "Towards a Standardized Representation for Deep Learning Collective Algorithms". (todo: Add link)

## Directory Structure
The repository is a collection of the following submodules:
```
- astra-sim: The ASTRA-sim simulator and its collective API extension. This collective API extension allows users to define the collective algorithm, instead of using or writing the default algorithms defined in the simulator's System layer. 
- chakra: An updated version which includes the converter from MSCCL-IR to Chakra ET for collective communication algorithms. 
- msccl-tools (as-is): Provides examples of the MSCCLang DSL to define collective algorithms. 
```

# Running the Workflow
## Setup
Please refer to the [ASTRA-sim wiki](https://astra-sim.github.io/astra-sim-docs/getting-started/setup.html) for required setup environments. 
```
cd astra-sim
bash build/astra_analytical/build.sh
```

## Generating Workload ET
```
cd extern/graph_frontend/chakra
python3 -m utils.et_generator.et_generator --num_npus 64 --num_dims 1 --default_comm_size 16384
```

## Generating Collective Algorithm ET
```
cd ../../../../msccl-tools
python3 allreduce_a100_ring.py 64 1 1 > demo_allreduce.xml

cd ../chakra
python3 -m et_converter.et_converter \
        --input_type        msccl \
        --input_filename    ../msccl-tools/demo_allreduce.xml \
        --output_filename   ../msccl-tools/allreduce_ring_mscclang \
        --num_dims          1 \
        --coll_size         16384'
```

## Running the Simulation in ASTRA-sim
```
cd ../astra-sim
export SYSTEM_CONFIG="./inputs/system/Ring.json"
export MEMORY_CONFIG="./inputs/remote_memory/analytical/no_memory_expansion.json"
export WORKLOAD_CONFIG="./extern/graph_frontend/chakra/one_comm_coll_node_allreduce"
export NETWORK_CONFIG="./inputs/network/analytical/Ring.yml"

# Run
./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
    --workload-configuration=$WORKLOAD_CONFIG \
    --system-configuration=$SYSTEM_CONFIG \
    --network-configuration=$NETWORK_CONFIG \
    --remote-memory-configuration=$MEMORY_CONFIG
```