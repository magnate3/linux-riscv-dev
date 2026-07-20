TCP Inigo Congestion Control
============================

This is an implementation of TCP Inigo, which takes the measure of
the extent of congestion introduced in DCTCP and applies it to
networks outside the data center.

[TCP Inigo: Fighting Congestion With Both Hands](https://www.soe.ucsc.edu/research/technical-reports/UCSC-SOE-14-14)
describes the enhancements being pursued in this project. At this
point, only the RTT-based congestion control, RTT-fairness, and
stability enhancements have been implemented.

[Mininet tests for TCP Inigo](https://github.com/systemslab/mininet-tests/tree/inigo)
include the basic Iperf incast test that was done by the [Mininet](http://mininet.org)
folks for DCTCP, as well as the ability run
[Flent, aka netperf-wrapper](https://github.com/tohojo/netperf-wrapper) tests.

[Experiments](https://github.com/systemslab/tcp_inigo_experiments)
show that TCP Inigo is able to keep queue depths at DCTCP levels using only
RTTs as a signal of congestion. And Inigo can use both.

The motivation behind the RTT fairness functionality comes from
the 2nd DCTCP paper listed below.

Authors:

     Andrew Shewmaker <agshew@gmail.com>

Forked from DataCenter TCP (DCTCP) congestion control.

[DCTCP]( http://simula.stanford.edu/~alizade/Site/DCTCP.html)
is an enhancement to the TCP congestion control algorithm
designed for data centers. DCTCP leverages Explicit Congestion
Notification (ECN) in the network to provide multi-bit feedback to
the end hosts. DCTCP's goal is to meet the following three data
center transport requirements:

 - High burst tolerance (incast due to partition/aggregate)
 - Low latency (short flows, queries)
 - High throughput (continuous data updates, large file transfers)
   with commodity shallow buffered switches

The algorithm is described in detail in the following two papers:

1) Mohammad Alizadeh, Albert Greenberg, David A. Maltz, Jitendra Padhye,
   Parveen Patel, Balaji Prabhakar, Sudipta Sengupta, and Murari Sridharan:
     "Data Center TCP (DCTCP)", Data Center Networks session
     Proc. ACM SIGCOMM, New Delhi, 2010.
  [PDF](http://simula.stanford.edu/~alizade/Site/DCTCP_files/dctcp-final.pdf)

2) Mohammad Alizadeh, Adel Javanmard, and Balaji Prabhakar:
     "Analysis of DCTCP: Stability, Convergence, and Fairness"
     Proc. ACM SIGMETRICS, San Jose, 2011.
  [PDF](http://simula.stanford.edu/~alizade/Site/DCTCP_files/dctcp_analysis-full.pdf)

Initial prototype from Abdul Kabbani, Masato Yasuda and Mohammad Alizadeh.

DCTCP Authors:

     Daniel Borkmann <dborkman@redhat.com>
     Florian Westphal <fw@strlen.de>
     Glenn Judd <glenn.judd@morganstanley.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.
