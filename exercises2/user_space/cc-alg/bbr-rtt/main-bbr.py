#!/usr/bin/python


from numpy import std

from subprocess import Popen, PIPE
from time import sleep, time
from multiprocessing import Process
from argparse import ArgumentParser

#from monitor import monitor_qlen, monitor_bbr, capture_packets
#import termcolor as T

import json
import math
import os
import sched
import socket
import sys
default_dir = '.'
parser = ArgumentParser(description="BBR Replication")
#parser.add_argument('--bw-host', '-B',
#                    type=float,
#                    help="Bandwidth of host links (Mb/s)",
#                    default=1000)
#
#parser.add_argument('--bw-net', '-b',
#                    type=float,
#                    help="Bandwidth of bottleneck (network) link (Mb/s)",
#                    required=True)
#
#parser.add_argument('--delay',
#                    type=float,
#                    help="Link propagation delay (ms)",
#                    required=True)
#
parser.add_argument('--dir', '-d',
                    help="Directory to store outputs",
                    required=True)

#parser.add_argument('--time', '-t',
#                    help="Duration (sec) to run the experiment",
#                    type=int,
#                    default=10)
#
#parser.add_argument('--maxq',
#                    type=int,
#                    help="Max buffer size of network interface in packets",
#                    default=100)
#
## Linux uses CUBIC-TCP by default.
#parser.add_argument('--cong',
#                    help="Congestion control algorithm to use",
#                    default="bbr")
#
#parser.add_argument('--fig-num',
#                    type=int,
#                    help="Figure to replicate. Valid options are 5 or 6",
#                    default=6)
#
#parser.add_argument('--flow-type',
#                    default="netperf")
#
#parser.add_argument('--environment',
#                    default="vms")
#
#parser.add_argument('--no-capture',
#                    action='store_true',
#                    default=False)
#
#parser.add_argument('--dest-ip',
#                    default="10.138.0.3")
#
## Expt parameters
args = parser.parse_args()


# Simple wrappers around monitoring utilities.
def start_tcpprobe(outfile="cwnd.txt"):
    os.system("rmmod tcp_probe; modprobe tcp_probe full=1;")
    Popen("cat /proc/net/tcpprobe > %s/%s" % (args.dir, outfile),
          shell=True)

def stop_tcpprobe():
    Popen("killall -9 cat", shell=True).wait()

def start_qmon(iface, interval_sec=0.1, outfile="q.txt"):
    monitor = Process(target=monitor_qlen,
                      args=(iface, interval_sec, outfile))
    monitor.start()
    return monitor

def monitor_bbr(dst, interval_sec = 0.01, fname='%s/bbr.txt' % default_dir, runner=None):
    cmd = "ss -iet dst %s" % (dst)
    runner = Popen if runner is None else runner
    print dst
    ret = []
    open(fname, 'w').write('')
    while 1:
        p = runner(cmd, shell=True, stdout=PIPE)
        output = p.stdout.read()
        try:
            start = output.find("bbr:(")
            end = output.find(")", start)
            if start == -1 or end == -1:
                continue
            data_elems = output[start+5:end].split(",")
            data = {}
            for d in data_elems:
                k, v = d.split(":")
                data[k] = v
            csvformat = "%s, %s, %s, %s" % (
                data["bw"],
                data["mrtt"],
                data["pacing_gain"],
                data["cwnd_gain"]
            )
            open(fname, 'a').write(csvformat + "\n")
        except Exception:
            pass
        sleep(interval_sec)
    return
def start_bbrmon(dst, interval_sec=0.1, outfile="bbr.txt", runner=None):
    monitor = Process(target=monitor_bbr,
                      args=(dst, interval_sec, outfile, runner))
    monitor.start()
    return monitor

if __name__ == "__main__":
    default_dir = args.dir
    p1 =start_bbrmon(dst="10.22.116.221:5202")
    #p1.start()
    p1.join()
