# RIFO: Pushing the Efficiency of Programmable Packet Schedulers

This repository contains the code for RIFO, which has been accepted for publication in the **IEEE/ACM Transactions on Networking**.

## Repository Contents

### `java-code`
This directory contains the Java-based implementation of RIFO, built on top of [NetBench](https://github.com/ndal-eth/netbench) and [AIFO](https://github.com/netx-repo/AIFO).  
This implementation was used to evaluate RIFO's performance under different workload distributions, as discussed in Section V.B. of the paper. It also includes comparisons with state-of-the-art programmable packet schedulers.

### `p4-code`
This directory contains the P4_16 implementation of RIFO for Tofino programmable switches.

### `RIFO-plots`
This directory contains the scripts and data used to generate the plots in the paper. If you prefer not to run the simulations, you can use these files directly to recreate the figures.

## Contact

Feel free to contact me via email at **h dot mostafaei at tue dot nl** for the following:

- If you're interested in collaborating on this project.
- If you encounter any issues while running the code.
- If you discover a bug.
- If you have any questions or concerns.
