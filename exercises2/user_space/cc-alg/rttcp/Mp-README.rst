=========
mptctrace
=========

mptcptrace analyze MPTCP traces

Reference
=========

If you plan to use this tool in a publication, please use the following reference:

.. code-block:: console

        @inproceedings{Hesmans:2014:TMT:2619239.2631453,
         author = {Hesmans, Benjamin and Bonaventure, Olivier},
         title = {Tracing Multipath TCP Connections},
         booktitle = {Proceedings of the 2014 ACM Conference on SIGCOMM},
         series = {SIGCOMM '14},
         year = {2014},
         isbn = {978-1-4503-2836-4},
         location = {Chicago, Illinois, USA},
         pages = {361--362},
         numpages = {2},
         url = {http://doi.acm.org/10.1145/2619239.2631453},
         doi = {10.1145/2619239.2631453},
         acmid = {2631453},
         publisher = {ACM},
         address = {New York, NY, USA},
         keywords = {Multipath TCP},
        } 


Building
========

You can build mptctrace with:

.. code-block:: console

        $ ./autogen.sh
        $ ./configure --prefix=whatever/
        $ make
        $ make install

I you have troubles to compile it, you can contact me.

Use it
======

You need to provide a pcap trace to mptcptrace with the ``-f`` option. Mptcptrace will recognize ETH and Linux cooked header, if it's something else, you can use "-o" to tell mptctrace the offset to go to the IP header.

There is manpage in the man directory.

To get started you can try the ``-s`` option that will output MPTCP sequence graph:

.. code-block:: console

        $ mptcptrace -f myDump.pcap -s

This will generate 2 xplot files for each MPTCP connection inside the trace (one to show sequences numbers from client to server (c2s) and the other to show sequences numbers from the server to the client (s2c)).

You can also try the goodput graph with ``-G 20``:

.. code-block:: console

        $ mptcptrace -f myDump.pcap -G 20

That will generate the gput files.


CSV output usage
================

Mptcptrace let you output information into CSV format. It's easy to reuse to plot the information, make statistics, be creative.

To get the CSV output, you can use ``-w 2`` options, and all other regular options.


.. code-block:: console

        $ mptcptrace -f myDump.pcap -s -w 2

Will output MPTCP sequence inforamtions in a CSV format.

One quick GNU plot script example can be found in ``res/scripts/gnuplot/seq_sf``

.. code-block:: console
        
        $ mptcptrace -f myDump.pcap -s -w 2
        $ gnuplot -e "maxsf=16" seq_sf < c2s_seq_0.csv > seq_sf.eps
        $ evince seq_sf.eps

|

.. figure:: http://mptcptrace.multipath-tcp.org/res/seq_sf.png
   :width: 100 %
   :align: center
   :figwidth: 100%


The output of the example is available in ``res/pics`` in eps format. This graph shows the MPTCP mappings that pass trough subflows. In red you can also see, the mappings that cause reinjections, and in green on which sublfows they have been reinjected.

You can also use use the CSV format to easely convert some ``xplot.org`` graphs, for instance, we use the ``R`` script in ``res/scripts/R/`` to translate the flight graph.

.. code-block::

        $ mptcptrace -f myDump.pcap -F 3 -w 2
        $ // prepend ts,val,met,DONT,USE,ME to c2s_flight_0.csv
        $ ./flightR c2s_flight_0.csv win.eps

|

.. figure:: http://mptcptrace.multipath-tcp.org/res/win.png
   :width: 100 %
   :align: center
   :figwidth: 50%

The output is available in ``res/pics`` in eps format.

Man page
========

.. code-block::

        MPTCPTRACE(1)                  mptcptrace Manual                 MPTCPTRACE(1)



        NAME
               mptcptrace - MPTCP connection analysis

        SYNOPSIS
               mptcptrace [options] -f filename

        DESCRIPTION
               mptcptrace  is  a  tool  that enable the analysis of dump that contains
               MPTCP capable connection(s).

        OPTIONS
               The following options are supported:

               -s     MPTCP sequence number graph

               -a     MPTCP ack size graph

               -r     RTT at MPTCP level, X axis may be selected :
                      1     x is timestamp of the ACK arrival
                      2     x is timestamp of the SEQ departure
                      4     x is SEQ numbers
                      To get more than one graph, just add the  value.  E.g.  6  would
                      give the second and the third graph.

               -F     MPTCP  Flight  size  graphs.  You have two kinds of MPTCP flight
                      size graphs.
                      1     Shows the receive window, the MPTCP  flightsize,  and  the
                      sum of the TCP (sublfow) flight size.
                      2     Show the flight size per subflow.
                      To  get more than one graph, just add the value.  -G You have to
                      specify the size of the table to make the moving average.  Small
                      number  will  be  closed to instantaneous goodput but may be too
                      variable. Big numbers will lead to a smoother graph but may  not
                      reflect  some holes in the connection.  Measures the MPTCP good-
                      put. The red line is the average good put since  the  bbegining.
                      The blue diamond represents the moving average.

               -S     Output  statistics in a CSV format. The set of statistics is not
                      yet well defined.

               -q     Specify the length of the queue that contains sequence  that  we
                      have to keep in memory for reinjection checking. By default this
                      option is set to 0 which means infinite queue. If you have  very
                      long trace, you may be forced to limit the size of the queue.

               -o     Specify the offset of the IP packet. Could be usefull if the top
                      layer is unknown by the program. It currently recognize automat-
                      ically ETHernet and Cooked.

               -w     Select  a writer to output the results. This option is not fully
                      implemented.  The default writer is xlot (0).
                      0     Output xplot files. (default)
                      1     Output google chart files. No fully implemented. The  main
                      reason  is  scability issues. Nevertheless can be used for small
                      traces. For demo.
                      2     Output csv files. Can be used to plot the information with
                      other program or post-process the output. E.g. we used this out-
                      put to generate Gnuplot graph on the web interface. We also  use
                      this output with R.


               -h     Print a short help and then exit.

               -v     TODO

        LICENCE
               TODO

        BUGS
               report to benjamin.hesmans@uclouvain.be

        AUTHOR
               Benjamin Hesmans benjamin.hesmans@uclouvain.be

        SEE ALSO
               tcptrace(1), xplot.org(1), gnuplot(1), R(1)



        Version 0.1                       May 7, 2014                    MPTCPTRACE(1)

