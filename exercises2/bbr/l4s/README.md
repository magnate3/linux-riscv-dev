This repository contains artifacts for the paper [To adopt or not to adopt L4S-compatible congestion control? Understanding performance in a partial L4S deployment](https://link.springer.com/chapter/10.1007/978-3-031-85960-1_10), accepted for presentation at the PAM 2025 conference.

# L4S

With few exceptions, the path to deployment for any Internet technology requires that there be some benefit to unilateral adoption of the new technology. In an Internet where the technology is not fully deployed, is an individual better off sticking to the status quo, or adopting the new technology? This question is especially relevant in the context of the Low Latency, Low Loss, Scalable Throughput (L4S) architecture, where the full benefit is realized only when compatible protocols (scalable congestion control, accurate ECN, and flow isolation at queues) are adopted at both endpoints of a connection and also at the bottleneck router. 

In this experiment, we consider the perspective of the sender of an L4S flow using scalable congestion control, without knowing whether the bottleneck router uses an L4S queue, or whether other flows sharing the bottleneck queue are also using scalable congestion control. Specifically, we conduct single or multiple flow coexistence experiments where L4S flows (TCP Prague and L4S-compatible BBRv2) share the same bottleneck with non-L4S flows (TCP CUBIC, BBRv1/v2/v3). These experiments are performed over various AQM types on the router, including FIFO, CoDel, PIE, FQ, FQ-CoDel, and DualPI2.

This repository includes:

 - FABRIC testbed notebooks and descriptions for generating experiment data.
 - Google Colab notebooks for generating figures from the data.

To run this experiment on [FABRIC](https://fabric-testbed.net), you should have a FABRIC account with keys configured, and be part of a FABRIC project. You will need to have set up SSH keys and understand how to use the Jupyter interface in FABRIC.

## Reproducing the Figures

You can use our experiment data to generate the figures in our paper.

For single flow experiments follow this link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fatihsarpkaya/L4S-PAM2025/blob/main/plotting-notebooks/1vs1.ipynb)

For multiple flow experiments follow this link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fatihsarpkaya/L4S-PAM2025/blob/main/plotting-notebooks/multiple_flows.ipynb)

In these notebooks, we download our experiment data from Google Drive and plot the figures accordingly. You can also modify the notebooks to use your own experiment data to create your plots.

## Run my Experiment (Generating Experiment Data)

To reproduce our experiments on FABRIC, log in to the FABRIC testbed's JupyterHub environment. Open a new terminal from the launcher, and run:

> git clone https://github.com/fatihsarpkaya/L4S-PAM2025.git

For single flow experiments, run the `single_flow_experiments.ipynb` notebook. For multiple flow experiments, run the `multiple_flow_experiments.ipynb` notebook. In each notebook, you will find additional instructions that require careful attention. Please read and follow the instructions in the notebook thoroughly. Using this notebook structure, you can collect data for all the figures in the paper except Figures 1, and 2.

Upon completing the notebook execution, you should have JSON files containing the collected data. Make sure the JSON file names follow the naming rules given in the notebook. This is necessary for compatibility with our plotting notebooks. Otherwise, you will need to modify the plotting notebooks accordingly.



## Important Note on DualPI2 Configuration in the Results

In the paper, the size of the DualPI2 queue is configured with the default parameter of 10,000 packets (approximately 12 BDP in our settings), even though the heatmaps show different buffer sizes on the x-axis. Since the ECN threshold is set to 5 ms in the classic queue and 1 ms in the low-latency queue — and non-ECN packets are dropped — adjusting the queue size for lower BDP values does not significantly affect the results for CUBIC and BBRv1.
However, for BBRv2 and BBRv3, we observe better fairness (rather than BBR dominance) at 0.5 BDP and 1 BDP buffer sizes in most cases. For buffer sizes of 2 BDP and larger, the trends remain consistent with the default configuration as shown in the paper.
