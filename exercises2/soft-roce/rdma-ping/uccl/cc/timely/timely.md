# Purpose
The purpose of this repository is to provide a *Practitioner's Guide* to implementing a userspace only network packet congestion control scheme to be used with a low-latency UDP messaging library on intra-corporate networks. This work shall be complete when,

* Concepts for congestion control, packet drop/loss are well explained
* Implementation choices are well described
* Code is provided to concretely demonstrate concepts and implementation
* A guide is provided on how congestion control might be incorporated into a DPDK messaging library

# Papers
The following papers are a good starting place. Many papers are co-authored by Google, Intel, and Microsoft and/or are used in production. Packets moving between data centers through WAN is not in scope. This work is for packets moving inside a corporate network. No code uses the kernel; this is all user-space work typically done with DPDK/RDMA.

* [1] [ECN or Delay: Lessons Learnt from Analysis of DCQCN and TIMELY](http://yibozhu.com/doc/ecndelay-conext16.pdf)
* [2] [ECN Github](https://github.com/jitupadhye-zz/rdma)
* [3] [TIMELY: RTT-based Congestion Control for the Datacenter](https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p537.pdf)
* [4] [TIMELY Power Point Slides](http://radhikam.web.illinois.edu/TIMELY-sigcomm-talk.pdf)
* [5] [TIMELY Source Code Snippet](http://people.eecs.berkeley.edu/~radhika/timely-code-snippet.cc) referenced and used in eRPC
* [6] [Datacenter RPCs can be General and Fast](https://www.usenix.org/system/files/nsdi19-kalia.pdf) aka **eRPC**
* [7] [eRPC Source Code](https://github.com/erpc-io/eRPC)
* [8] [Carousel: Scalable Traffic Shaping at End Hosts](https://saeed.github.io/files/carousel-sigcomm17.pdf)
* [9] [Receiver-Driven RDMA Congestion Control by Differentiating Congestion Types in Datacenter Networks](https://icnp21.cs.ucr.edu/papers/icnp21camera-paper45.pdf)

Points of orientation:

* [1] contains a correction to [3] and also describes DCQCN with ECN but not Carousel. ECN requires switches/routers to be ECN enabled. Equinix does not or cannot do ECN, for example.
* [1] concludes DCQCN is better than Timely
* [9] describes another approach handily beating Timely, however, the "deployment of RCC relies on high precision clock synchronization throughout the datacenter network. Some recent research efforts can reduce the upper bound of clock synchronization within a datacenter to a few hundred nanoseconds, which is sufficient for our work."
* Timely congestion control therefore is the least proscriptive, and probably worst performing. I would note most DPDK/RDMA work like [RAMCloud](https://dl.acm.org/doi/pdf/10.1145/2806887) rely on lossless packet hardware atypical in corporate data centers. This is another reason why Timely is in scope
* eRPC [6,7] uses Carousel [8] and Timely [3,4,5] to good effect

# Build Code
In an empty directory do the following assuming you have a C++ tool chain and cmake installed:

1. git clone https://github.com/gshanemiller/congestion.git
2. mkdir build && cd build
3. cmake ..

All tasks/libraries can be found in the './build' directory

# R Plots
Some test programs produce data plotted with R (freeware stats program) using a provided R script. See individual READMEs for details. You might encounter `ggplot2` unknown or not found. To fix missing R dependencies, run the following commands in the R CLI:

```
# R packages used here
install.packages("ggplot2")
install.packages("gridExtra")
```

You only need to do this once. R is smart enough to find external source repositories and install. R does not need to be restarted.

# Milestones
Congestion control involves four major problems:

* Detect and correct packet drop/loss. Detection usually involves timestamps and sequence numbers. Resending packets is more involved
* Detect and respond to congestion by not sending too much data too soon e.g. [1-5, 9]
* Determine when to send new data without exasperbating congestion e.g. [8]
* Do all of the above without wasting CPU

I suggest the following milestone trajectory. It's valid *provided* Timely is the goto congestion control method. I report other methods above, however, they arguably impose impactical constraints.

```
Milestones
0                        
|----------------------------------------------------------------------------->
Provide theoretical motivation for
Timely based on [1,3]


Milestones (cont)
1                        2                          3
|------------------------+--------------------------+------------------------->
Simulate Timely in C++   Figure out timestamps.     When to use Timely?
The goal is to get a     eRPC uses rdtsc. Would     eRPC has a complicated
good impl of the model   NIC timestamps be better?  way to selectively
                         For Mellanox NICs, how     ignore Timely or use it
                         where are they?


Milestones (cont)
4                        5                          6
|------------------------+--------------------------+------------------------->
Combine 1-3 into a impl  Extend (4) adding kernel   Extend (5) adding
closer to production     UDP sockets and eRPC       sequence numbers to detect
code                     packet level pacing. ACKs  packet drop/reorder. Figure
                         here form RTT. Run sender/ out a way to resend lost
                         receiver pair to test      data in order


Milestones (cont)
7                        8                          9
|------------------------+--------------------------+------------------------->
Describe Carousel arch.  Document arch to add       Implement design in (8)
Delineate where work in  Carousel to (6)            and test/validate it.
(6) stops and Carousel
starts.


Milestones (cont)
10                       11
|------------------------+--------------------------+------------------------->
Document how to bring    Implement (10) using code
the work in (9) into a   in Reinvent library
DPDK setting
```

# Milestone Completion Status
0. **DONE**: see [congestion.pdf](https://github.com/gshanemiller/congestion/blob/main/congestion.pdf) sections 3,4
1. **DONE** See [Timely Basic](https://github.com/gshanemiller/congestion/tree/main/experiment/timely_basic), and [Timely eRPC](https://github.com/gshanemiller/congestion/tree/main/experiment/timely_erpc)
2. **STARTED**
3. Not started
4. Not started
5. Not started
6. Not started
7. **ALMOST DONE**: see [congestion.pdf](https://github.com/gshanemiller/congestion/blob/main/congestion.pdf) sections 5
8. Not started
9. Not started
10. Note started
11. Note started
