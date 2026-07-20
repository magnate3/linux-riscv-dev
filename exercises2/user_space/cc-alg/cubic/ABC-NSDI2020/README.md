# ABC-NSDI2020
This repo contains code for running the ABC protocol using Mahimahi (emulator). Please see our paper for details of the scheme: https://www.usenix.org/system/files/nsdi20-paper-goyal.pdf.

## Relevant Files:
1. mahimahi/src/packet/cellular_packet_queue.cc: This file contains logic for the ABC router
2. tcp_abc.c: Basic linux endpoint for ABC
3. tcp_abccubic.c: Linux endpoint for ABC with modifications for coexistence with non-ABC routers (see section 4.1 of the paper)
4. run-exp.sh: Script to run an experiment

## Setup:
1. Install Mahimahi:  
a. cd mahimahi  
b. ./autogen.sh  
c. ./configure  
d. make && sudo make install  

2. Install ABC endpoint:  
a. make  
b. sudo insmod abc.ko  
c. cat /proc/sys/net/ipv4/tcp_available_congestion_control | sudo tee /proc/sys/net/ipv4/tcp_allowed_congestion_control


## refercens
[TCP CUBIC 动力学](https://blog.csdn.net/dog250/article/details/130139269)   
[tcp cubic 与随机丢包](https://blog.csdn.net/dog250/article/details/130628839)   
[TCP_Cubic_V2：改进版TCP Cubic拥塞控制算法实现](https://blog.csdn.net/gitblog_01127/article/details/142159858?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-142159858-blog-130139269.235^v43^pc_blog_bottom_relevance_base7&spm=1001.2101.3001.4242.3&utm_relevant_index=7)  
[RFC8312 中文翻译——用于快速长距离网络的 cubic 算法](https://www.jianshu.com/p/d4db63e2b519)  
[从 TCP Reno 经 BIC 到 CUBIC](https://zhuanlan.zhihu.com/p/760480117)  
[TCP CUBIC 曲线对 BIC 折线的拟合](https://zhuanlan.zhihu.com/p/782072342)   
