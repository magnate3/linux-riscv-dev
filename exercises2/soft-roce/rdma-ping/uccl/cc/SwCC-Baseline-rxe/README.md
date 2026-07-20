# Table of contents

- [Table of contents](#table-of-contents)
- [QDMA Software Guide](#qdma-software-guide)
    - [Driver Installation](#driver-installation)
    - [QDMA Lib Installation](#qdma-lib-installation)
    - [Software Test](#software-test)
        - [`qdma_throughput`](#qdma_throughput)
        - [`qdma_random`](#qdma_random)
        - [`qdma_latency`](#qdma_latency)
        - [`axil_latency`](#axil_latency)
- [Benchmark results](#benchmark-results)
    - [AXILite latency](#axilite-latency)
    - [Randowm access throughput (GB/s)](#randowm-access-throughput-gbs)
        - [H2C](#h2c)
        - [C2H (more than 8 qs will always fail)](#c2h-more-than-8-qs-will-always-fail)
        - [Concurrent](#concurrent)
    - [DMA latency](#dma-latency)
        - [H2C](#h2c-1)
        - [C2H](#c2h)
- [QDMA C2H bug](#qdma-c2h-bug)
- [Be careful](#be-careful)

# QDMA Software Guide

The following kernel and distros have been tested:  
* Ubuntu 18.04 LTS (Kernel 4.15.0-20-generic)  
* Ubuntu 22.04 LTS (Kernel 5.15.0-72-generic)  
* Ubuntu 22.04 LTS (Kernel 5.15.0-100-generic)

## Driver Installation

1. prepare 
clone these repo to your home dir or anywhere you like.
```bash
git clone git@github.com:RC4ML/qdma_driver.git
git clone --recursive git@github.com:RC4ML/rc4ml_qdma.git
sudo apt-get install libaio1 libaio-dev
```

2. compile
```bash
cd ~/qdma_driver
make modulesymfile=/usr/src/linux-headers-$(uname -r)/Module.symvers
make apps modulesymfile=/usr/src/linux-headers-$(uname -r)/Module.symvers
```

3. install apps and header files
```bash
sudo make install-apps modulesymfile=/usr/src/linux-headers-$(uname -r)/Module.symvers
```

4. install kernel mod
```bash
sudo insmod src/qdma-pf.ko
```

If you find the kernel module fails to install due to invalid module format, consider updating your Linux header files by the following script:
```bash
sudo apt update && sudo apt upgrade
sudo apt remove --purge linux-headers-*
sudo apt autoremove && sudo apt autoclean
sudo apt install linux-headers-generic
```

## QDMA Lib Installation

Note: `nvcc` is required in the system PATH.

```bash
$ cd ~/rc4ml_qdma
$ mkdir build
$ cd build
$ cmake  ..
$ sudo make install 
```

There are four binary files you can use:

### `qdma_throughput`

Coressponed to `QDMATop.scala`.

- `h2c_benchmark()` will test host to card channel
- `c2h_benchmark()` will test card to host channel
- `benchmark_bridge_write()` will test axi bridge channel

### `qdma_random`

Coressponed to `QDMARandomTop.scala`, which aims to benchmark random access performance. (1 GB memory)

- `h2c_benchmark_random()` will test host to card channel
- `c2h_benchmark_random()` will test card to host channel
- `concurrent_random()` will test concurrent performance, if you want to get one direction performance (such as h2c),
  you need to set another's factor (c2h_factor) to 2 ensuring that c2h is always running when h2c is performed.

### `qdma_latency`

Coressponed to `QDMALatencyTop.scala`, which aims to benchmark dma channel's host to card and card to host latency. (1
GB memory)

- `h2c_benchmark_latency()` will test host to card channel
- `c2h_benchmark_latency()` will test card to host channel
- `concurrent_latency()` will test concurrent performance, this is not fully implemented and latency would increase
  around 1us when fully loaded.

### `axil_latency`

Coressponed to `AXILBenchmarkTop.scala`, which aims to benchmark the AXIL read latency in various situations under
different workloads.

- `startFpgaH2C(...)` will initialize the host to card channel with a simple throughput benchmark
- `startFpgaC2H(...)` will initialize the card to host channel with a simple throughput benchmark
- `axilReadBenchmark(...)` will test the axi lite read latency

Before you run these binaries, you must program FPGA and reboot the host.
Each time you reboot you need to redo the insmod step (i.e., sudo insmod src/qdma-pf.ko)

And following instructions needs to be executed before you run binaries.

```bash
$ sudo su
$ echo 1024 > /sys/bus/pci/devices/0000:1a:00.0/qdma/qmax
$ dma-ctl qdma1a000 q add idx 0 mode st dir bi
$ dma-ctl qdma1a000 q start idx 0 dir bi desc_bypass_en pfetch_bypass_en
$ dma-ctl qdma1a000 q add idx 1 mode st dir bi
$ dma-ctl qdma1a000 q start idx 1 dir bi desc_bypass_en pfetch_bypass_en
$ dma-ctl qdma1a000 q add idx 2 mode st dir bi
$ dma-ctl qdma1a000 q start idx 2 dir bi desc_bypass_en pfetch_bypass_en
$ dma-ctl qdma1a000 q add idx 3 mode st dir bi
$ dma-ctl qdma1a000 q start idx 3 dir bi desc_bypass_en pfetch_bypass_en
```

Run your binaries according to which bitstream is in FPGA.

There are some useful commands  (provided by Xilinx QDMA Linux Kernel Driver) in cmd.txt.

# Benchmark results

Testbed: amax2, U280 board.

---
This benchmark the axi-lite read latency when dma channel is busy.

## AXILite latency

| AXIL Latency          |             | QDMA Bandwidth |             |
|-----------------------|-------------|----------------|-------------|
| axi lite read(yes/no) | latency(us) | read(GBps)     | write(GBps) |
| no                    | /           | 12.79          | 12.99       |
| yes                   | 0.88        | 0              | 0           |
| yes                   | 0.95        | 12.79          | 0           |
| yes                   | 1.47        | 0              | 12.98       |
| yes                   | 2.95        | 4.8            | 12.9		      |
| (512 * 4's average)   |             |                |             |

---

## Randowm access throughput (GB/s)

Host memory size = 1GB, total cmds = 256*1024. This benchmark the random access throughput

### H2C

| package size (Bytes) | Qs = 1 | 2    | 4    | 8    | 16   |           |
|----------------------|--------|------|------|------|------|-----------|
| 64                   | 2.07   | 1.97 | 1.96 | 2.05 | 2.03 | OPS ~ 32M |
| 128                  | 3.93   | 3.96 | 3.98 | 3.79 | 3.98 | OPS ~ 32M |
| 256                  | 7.24   | 7.27 | 7.85 | 7.52 | 7.58 | OPS ~ 29M |

### C2H (more than 8 qs will always fail)

| package size (Bytes) | Qs = 1 | 2     | 4     | 8     |           |
|----------------------|--------|-------|-------|-------|-----------|
| 64                   | 4.97   | 4.97  | 4.97  | 4.97  | OPS ~ 80M |
| 128                  | 9.85   | 9.93  | 9.93  | 9.93  | OPS ~ 79M |
| 256                  | 11.92  | 11.92 | 11.92 | 11.92 | OPS ~ 48M |

### Concurrent

| package size (Bytes) | Qs=1  |       | 2    |      | 4    |      | 8    |      |         |
|----------------------|-------|-------|------|------|------|------|------|------|---------|
|                      | H2C   | C2H   | H2C  | C2H  | H2C  | C2H  | H2C  | C2H  |         |
| 64                   | 1.64  | 1.83  | 1.77 | 1.83 | 1.79 | 1.8  | 1.64 | 1.84 | OPS~29M |
| 128                  | 3.42  | 3.52  | 3.31 | 3.35 | 3.36 | 3.44 | 3.52 | 3.5  | OPS~28M |
| 256                  | 6.56  | 6.63  | 6.37 | 6.45 | 6.64 | 6.4  | 6.57 | 6.47 | OPS~26M |
| 512                  | 6.66  | 12.35 | x	   | x	   | x    | x    | x    | x    | x       |
| 1024                 | 10.68 | 12.12 | x	   | x	   | x    | x    | x    | x    | x       |
| 2048                 | 10.88 | x	    | x	   | x	   | x    | x    | x    | x    | x       |
| 4096                 | 11.39 | x	    | x	   | x	   | x    | x    | x    | x    | x       |

---

## DMA latency

Host memory = 1GB, total cmds = 256*1024. This benchmark the dma read/write latency.

`Wait cycles` is the minimum duration when issuing two cmds, thus the maximum OPS is limited.

`Latency CMD` calculated duration begin when cmd issues, ends when axibridge returns.

`Latency DATA` calculated duration begin when last beat data issues, ends when axibridge returns.

*this latency can be thousands us sometimes, because write latency use bridge channel to reply, single thread can issue
around 8M bridge write, when ops exceeds this, the latency increase a lot.

### H2C

| Packet Size | Wait cycles | OPS limit | Throughput (Mops) | Throughput (GB/s) | Latency (us) |
|-------------|-------------|-----------|-------------------|-------------------|--------------|
| 64B         | 50          | 5M        | 4.6               | 0.3               | 1.0          |
|             | 25          | 10M       | 8.8               | 0.6               | 1.1          |
|             | 12          | 20M       | 17.0              | 1.1               | 1.0          |
|             | 6           | 40M       | 29.8              | 1.9               | 1.0          |
|             | 0           | N/A       | 36.2              | 2.3               | 2.5          |
|             |             |           |                   |                   |              |
| 4KB         | 100         | 2.5M      | 2.3               | 9.3               | 1.4          |
|             | 90          | 2.8M      | 2.6               | 10.4              | 1.5          |
|             | 85          | 2.9M      | 2.7               | 11.0              | 1.5          |
|             | 80          | 3.1M      | 2.9               | 11.6              | 1.6          |
|             | 75          | 3.3M      | 3.1               | 12.4              | 1.7          |
|             | 70          | 3.6M      | 3.2               | 12.8              | 11.2         |
|             | 50          | 5M        | 3.2               | 12.8              | 11.3         |

### C2H

| Packet Size | Wait cycles | OPS limit | Throughput (Mops) | Throughput (GB/s) | Latency CMD (us) | Latency DATA (us) |
|-------------|-------------|-----------|-------------------|-------------------|------------------|-------------------|
| 64B         | 50          | 5M        | 4.6               | 0.29              | 1.3              | 1.3               |
|             | 25          | 10M       | 8.8               | 0.55              | 1.7*             | 1.7*              |
|             |             |           |                   |                   |                  |                   |
| 4KB         | 100         | 2.5M      | 2.3               | 9.35              | 1.4              | 1.2               |
|             | 90          | 2.8M      | 2.6               | 10.37             | 1.3              | 1                 |
|             | 85          | 2.9M      | 2.7               | 10.96             | 1.5              | 1.2               |
|             | 80          | 3.1M      | 2.9               | 11.63             | 1.6              | 1.3               |
|             | 75          | 3.3M      | 3.1               | 12.39             | 1.6              | 1.3               |
|             | 70          | 3.6M      | 3.2               | 12.8              | 6.8              | 3.8               |
|             | 50          | 5M        | 3.2               | 12.82             | 6.6              | 3.6               |

---

# QDMA C2H bug

1. When running with more than 8 qs, it will always fail. QDMA C2H data port's ready would be down after receiving
   several data.

2. Even running with less than or equal to 8 qs, it can sometimes fail, try reprogram the FPGA. I guess only one q has
   the most chance to pass.

3. Tested situation:

   packet size: 1K/32K(which would be splited into multiple packets)

4. I have tried fetching tag index for each queue using dma-ctl, it's useless.

---

# Be careful

1. All statistics are calculated at the 250M user clock, (so if your speed is 10.6 GB/s at most, maybe you have used
   300M user clock).
