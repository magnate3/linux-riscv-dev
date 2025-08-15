# Ultra-Low Latency User-Plane Cyberattack Detection in SDN-based Smart Grids

 This repository contains the public version of the code for our work on cyberattack detection in SDN-based smart grids at line rate leveraging user-plane inference.

The paper was presented at the ACM SIGEnergy Workshop on the Security and Privacy of Energy Systems (EnergySP), co-located with the 15th ACM International Conference on Future and Sustainable Energy Systems (e-Energy), Singapore, 4 June 2024.

We show that by leveraging user-plane inference on a per-packet basis, we can accurately and rapidly detect and mitigate cyberattacks on smart grid networks in real time. For details, please check out [our paper](https://dl.acm.org/doi/10.1145/3632775.3661995).

## Organization of the repository  
There are two folders:  
- _User_Plane_Inference_ : P4 code compiled and tested on an Intel Tofino switch, and the model table entries file.
- _Data_Analysis_ : scripts and instructions for processing the data, the jupyter notebooks for training the machine learning models, the python scripts for generating the M/A table entries from the saved trained models, and the control plane code for the benchmark solutions.

## Use case
The use case considered in the paper is a DNP3 attack detection and classification use case based on the publicly available <a href="https://ieee-dataport.org/documents/dnp3-intrusion-detection-dataset">DNP3 Intrusion Detection Dataset</a>. The challenge is to classify traffic into one of 7 classes of which 1 is benign and 6 are malicious.

If you make use of our code, please cite our paper.

```
@inproceedings{akem_sg2024,
author = {Akem, Aristide Tanyi-Jong and Gucciardo, Michele and Fiore, Marco},
title = {Ultra-Low Latency User-Plane Cyberattack Detection in SDN-based Smart Grids},
year = {2024},
isbn = {9798400704802},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3632775.3661995},
doi = {10.1145/3632775.3661995},
booktitle = {Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems},
pages = {676–682},
numpages = {7},
keywords = {P4, Smart grid, cyberattack, in-switch inference, machine learning},
location = {Singapore, Singapore},
series = {e-Energy '24}
}
```

If you need any additional information, send us an email at _aristide.akem_ at _imdea.org_.