# ISCA-2025-DLB

Our artifact contains all source files for AccDirect and provides instructions on how to reproduce key experimental results shown in the paper. There are three key sets of experiments: 

1️⃣ Experiments that evaluate the throughput and p99 latency of existing intra-host load balancers with both core-to-core and end-to-end setups (Fig. 3, 4, 5).

2️⃣ Experiments showing the benefits of using AccDirect in p99 latency, throughput, and power saving (Fig. 9, 10).

3️⃣ Microbenchmark experiments that justify the guidelines to make the best use of DLB (Fig. 11, 12, 13).

---

## 🔩 Requirements

### Hardware
- A server with an Intel 4th-generation Xeon Scalable Processor equipped with DLB (Intel Xeon Gold 6438Y+ CPU) and Nvidia Bluefield-3.
- A client with 100/200 Gbps ConnectX-6 Dx NIC, we tested with Intel Xeon E5-2660 v4 (Broadwell).


### Software
- Ubuntu 22.04
- Linux kernel 6.2+
- gcc 11.4.0
- DPDK 22.11.2
- DLB software release 8.9.0
- Python 3
- Meson, Ninja


⚙ Here is a summary of our testbed system setup.

|                    | **Server**                                    | **Client**                                  |
|--------------------|-----------------------------------------------|---------------------------------------------|
| **OS**           | Ubuntu 22.04.5 LTS<br>(Linux 6.5.0)              | Ubuntu 22.04.1 LTS<br>(Linux 5.15.0)            |
| **Processor**    | Intel Xeon (Sappire Rapids)                    | Intel Xeon (Broadwell)                         |
| **Model**        | Gold 6438Y+                                    | E5-2660 v4                                   |
| **# Cores**      | 32 (1 socket)                                  | 14 (1 socket)                                 |
| **System**       | 512 GB DDR5                                    | 64 GB DDR4-2400                              |
| **Memory**       | 16 DIMMs, 8 channels                           | 2 DIMMs, 2 channels                          |
| **NIC**          | BlueField-3                                    | ConnectX-6 Dx                                |
| **DPDK**         | 22.11.2                                        | 22.11.3                                |
| **libdlb**       | 8.9.0                                          | -                                |

---

## 📖 Contents
- `scripts` contains scripts to reproduce figures in the paper.
- `src/dlb_8.9.0/` contains a modified version of [Intel's released version 8.9.0](https://www.intel.com/content/www/us/en/download/686372/823245/intel-dynamic-load-balancer.html) of DLB driver and `libdlb`.
- `src/dlb_bench/` contains dpdk and libdlb benchmarks.
- `src/dpdk-22.11.2-dlb/` contains modifed dpdk library and dpdk-based DLB benchmarks.

---

## 🚀 Experiment Workflow
For setting up the environment and building from scratch, please follow steps 1, 2, and 3.

For ISCA 2025 AE, we have already cloned this repo under `/home/isca25_ae/` and built all the necessary drivers, libraries, and benchmarks on server, client, and snic. You can skip step 2, and only perform steps 1 and 3.

### 1. Kernel boot parameter
Please make sure the following kernel boot parameters are properly set in `/etc/default/grub` on the server. 
```
GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.5.0-41-generic" #isca2025
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt no5lvl" #isca2025
```

Then, to apply these modifications, a reboot is required:
```
$ sudo update-grub
$ sudo reboot
```

### 2. Installation
a). Clone the GitHub repo and submodules.

    ```
    git clone https://github.com/ece-fast-lab/ISCA-2025-DLB
    cd ISCA-2025-DLB
    git submodule update --init --recursive
    git submodule update --remote
    ```

b). Build all necessary drivers, libraries, and benchmarks on server, snic, or client.

    ```
    cd scripts/common/
    ./build_all.sh {server|snic|client}
    ```


### 3. Recommanded experiment order
Follow the instructions provided in each experiment folder to reproduce the figures in the paper. We recommend running the experiments in the following order, either on the server or on the client.
1. Run DLB benchmarks on the server in `scripts/fig11e/`, `scripts/fig12/`, and `scripts/fig13/`.
2. Run core-to-core experiments on the server in `scripts/fig3-4/`.
3. Setup DLB driver on the server side with `sudo ./setup_libdlb_dlb2.sh` in `scripts/common/`.
4. Run end-to-end experiments on the client in `scripts/fig9/` and `scripts/fig10/`.


---


## 🎯Experiments to reproduce

### 1️⃣ Section 3
1. Throughput and latency of intra-host load balancers for varying numbers of worker cores (Fig. 3).
2. Throughput versus the number of CPU cores that prepare and enqueue QEs (Fig. 4).
3. Impact of intra-host load balancers on end-to-end throughput and latency at different network packet rates (Fig. 5).

### 2️⃣ Section 4
1. Absoluate power saving and p99 latency overhead of AccDirect comparing against host CPU-based baseline (Fig. 9).
2. Masstree request latency of the software baseline and DLB with different query mixes. Since the experiments for Masstree take hours for a single plot, we kindly request the reviewers to reproduce only Fig. 10(a).

### 3️⃣ Section 6
1. One producer core configured as a direct or load-balancing port, with a fixed enqueue of 4 MPPS (Fig. 11(e)).
2. DLB’s packet and queue priority performance (Fig. 12).
3. The overall throughput improves and the workloads are more balanced across the worker cores as the number of flows increases (Fig. 13).

---




