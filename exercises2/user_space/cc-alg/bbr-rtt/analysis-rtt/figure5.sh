#!/bin/bash

# This shell script must be run using sudo.

run () {
    dir=$1

    chmod -R 0777 $dir
    #su $SUDO_USER -c "tshark -2 -r $dir/bbr.pcap -R 'tcp.stream eq $flow && tcp.analysis.ack_rtt'  -e frame.time_relative -e tcp.analysis.ack_rtt -Tfields -E separator=, > $dir/bbr_rtt.txt"
    su $SUDO_USER -c "tshark -2 -r $dir/bbr.pcap -R 'tcp.analysis.ack_rtt'  -e frame.time_relative -e tcp.analysis.ack_rtt -Tfields -E separator=, > $dir/bbr_rtt.txt"
    su $SUDO_USER -c "tshark -2 -r $dir/cubic.pcap -R 'tcp.analysis.ack_rtt'  -e frame.time_relative -e tcp.analysis.ack_rtt -Tfields -E separator=, > $dir/cubic_rtt.txt"
    #su $SUDO_USER -c "tshark -2 -r $dir/cubic.pcap -R 'tcp.stream eq $flow && tcp.analysis.ack_rtt'  -e frame.time_relative -e tcp.analysis.ack_rtt -Tfields -E separator=, > $dir/cubic_rtt.txt"

    python2 plot_ping.py -f $dir/bbr_rtt.txt $dir/cubic_rtt.txt --xlimit 8 -o $dir/figure5_$type.png
}

run out
