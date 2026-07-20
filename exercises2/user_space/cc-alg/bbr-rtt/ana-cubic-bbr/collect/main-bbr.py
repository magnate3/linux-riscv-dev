#!/usr/bin/python2


import signal
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
import re
import logging
import subprocess
processes = []
running = True
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

def monitor_qlen2(iface, interval_sec = 0.01, fname='%s/qlen.txt' % default_dir):
    pat_queued = re.compile(r'backlog\s+([\d]+\w+)\s+\d+p')
    pat_dropped = re.compile(r'dropped\s+([\d]+)') 
    cmd = "tc -s qdisc show dev %s" % (iface)
    f = open(fname, 'w')
    f.write('time,root_pkts,root_drp,child_pkts,child_drp\n')
    f.close()
    while 1:
        p = Popen(cmd, shell=True, stdout=PIPE)
        output = p.stdout.read()
        tmp = ''
        output = output.decode('utf-8')
        matches_queued = pat_queued.findall(output)
        matches_dropped = pat_dropped.findall(output)
        if len(matches_queued) != len(matches_dropped):
            print("WARNING: Two matches have different lengths!")
            print(output)
        if matches_queued and matches_dropped:
            tmp += '%f,%s,%s' % (time(), matches_queued[0],matches_dropped[0])
            if len(matches_queued) > 1 and len(matches_dropped)> 1: 
                tmp += ',%s,%s\n' % (matches_queued[1], matches_dropped[1])
            else:
                tmp += ',,\n'
        f = open(fname, 'a')
        f.write(tmp)
        f.close
        sleep(interval_sec)
    return
def monitor_qlen(iface, interval_sec = 0.01, fname='%s/qlen.txt' % default_dir):
    pat_queued = re.compile(r'backlog\s[^\s]+\s([\d]+)p')
    #pat_queued = re.compile(rb'backlog\s[^\s]+\s([\d]+)p')
    cmd = "tc -s qdisc show dev %s" % (iface)
    t0 = "%f" % time()
    while 1:
        p = Popen(cmd, shell=True, stdout=PIPE)
        output = p.stdout.read()
        tmp = ''
        output = output.decode('utf-8')
        matches= pat_queued.findall(output)
        if matches and len(matches) > 1:
            t1 = "%f" % time()
            #t = float(t1)-float(t0)
            y = int(matches[1])
            #tmp = str(t)+','+str(y)+'\n'
            tmp = str(t1)+','+str(y)+'\n'
            #print(tmp)
            f = open(fname, 'a')
            f.write(tmp)
            f.close
        sleep(interval_sec)
    return
def start_qmon(iface, interval_sec=0.1, outfile="queue.txt"):
    monitor = Process(target=monitor_qlen,
                      args=(iface, interval_sec, outfile))
    monitor.start()
    return monitor

def monitor_cubic(dst, interval_sec = 0.01, fname='%s/cubic.txt' % default_dir, runner=None):
    
    cmd = "ss -iet dst %s" % (dst)
    runner = Popen if runner is None else runner
    print dst
    ret = []
    open(fname, 'w').write('')
    while running:
        p = runner(cmd, shell=True, stdout=PIPE)
        output = p.stdout.read()
        #output = str(subprocess.check_output('ss -ti'.split()))
        #print(output)

        try:
            #start = output.find("cwnd:")
            #if start == -1:
            #    continue
            start = output.find("cubic")
            if start == -1:
                continue
            output = output[start:-1]
            m = re.match(r'.*(cwnd:\d+).ssthresh*',output)
            '''
            if m is not None:
                logging.error('Matched cwnd: {}'.format(m.group(1)))
            else:
                logging.error('not Matched cwnd')
            '''
            '''
            m = re.match(r'.*bbr:\(bw:(\S+)bps.*mrtt:(\S+),pac.*(pacing_rate \S+)bps.*(delivery_rate \S+)bps.*', output)
            #m = re.match(r'.*bbr:\(bw:(\S+).bps.*mrtt:(\S+),pac.*(pacing_rate \S+).bps.*(delivery_rate \S+).bps.*', output)
            if m is not None:
                #print(output)
                csvformat='btl_bw {}bps | mrtt {} | {}bps | {}bps'.format(m.group(1), m.group(2), m.group(3), m.group(4))
                #csvformat='{}bps,{},{}bps,{}bps'.format(m.group(1), m.group(2), m.group(3), m.group(4))
                #print(csvformat)
                open(fname, 'a').write(csvformat + "\n")
            '''
            if m is not None:
                csvformat = '{}'.format(m.group(1))
                open(fname, 'a').write(csvformat + "\n")
        except Exception as e:
            #pass
            logging.error("Oops !Failed to execute : " + str(e))
            #print("Oops!    Try again...")
            #pass
        sleep(interval_sec)
def monitor_bbr_v2(dst, interval_sec = 0.01, fname='%s/bbr.txt' % default_dir, runner=None):
    
    cmd = "ss -iet dst %s" % (dst)
    runner = Popen if runner is None else runner
    print dst
    ret = []
    open(fname, 'w').write('')
    while running:
        p = runner(cmd, shell=True, stdout=PIPE)
        output = p.stdout.read()
        #output = str(subprocess.check_output('ss -ti'.split()))
        #print(output)

        try:
            cwnd=''
            #start = output.find("cwnd:")
            #if start == -1:
            #    continue
            start = output.find("bbr")
            if start == -1:
                continue
            output = output[start:-1]
            m = re.match(r'.*cwnd:(\d+).ssthresh*',output)
            if m is not None:
                cwnd = 'cwnd {}'.format(m.group(1))
                #logging.error('Matched cwnd: {}'.format(m.group(1)))
            else:
                cwnd = 'cwnd {}'.format(0)
            start = output.find("bbr:(")
            if start == -1:
                continue
            #m = re.match(r'.*(cwnd:\d+).ssthresh*',output)
            #if m is not None:
            #    logging.error('Matched cwnd: {}'.format(m.group(1)))
            #else:
            #    logging.error('not Matched cwnd')
            output = output[start:-1]
            #logging.error(output)

            #m = re.match('cwnd:\d+.*', output)
            #m = re.match('.*cwnd:\d+.*', output)
            #if m is not None:
            #     print('Matched cwnd: {}'.format(m.group(0)))
            m = re.match(r'.*bbr:\(bw:(\S+)bps.*mrtt:(\S+),pac.*(pacing_rate \S+)bps.*(delivery_rate \S+)bps.*', output)
            #m = re.match(r'.*bbr:\(bw:(\S+).bps.*mrtt:(\S+),pac.*(pacing_rate \S+).bps.*(delivery_rate \S+).bps.*', output)
            if m is not None:
                #print(output)
                csvformat='btl_bw {}bps | mrtt {} | {}bps | {}bps | {} '.format(m.group(1), m.group(2), m.group(3), m.group(4),cwnd)
                #csvformat='{}bps,{},{}bps,{}bps'.format(m.group(1), m.group(2), m.group(3), m.group(4))
                #print(csvformat)
                open(fname, 'a').write(csvformat + "\n")
        except Exception as e:
            #pass
            logging.error("Oops !Failed to execute : " + str(e))
            #print("Oops!    Try again...")
            #pass
        sleep(interval_sec)

def monitor_bbr(dst, interval_sec = 0.01, fname='%s/bbr.txt' % default_dir, runner=None):
    cmd = "ss -iet dst %s" % (dst)
    runner = Popen if runner is None else runner
    print dst
    ret = []
    open(fname, 'w').write('')
    while running:
        p = runner(cmd, shell=True, stdout=PIPE)
        output = p.stdout.read()
        try:
            #data_res = [re.compile(r"cwnd:(?P<cwnd>\d+)", re.MULTILINE),
            #    re.compile(r"rtt:(?P<rtt>\d+\.\d+)/(?P<rtt_var>\d+\.\d+)",
            #               re.MULTILINE),
            #    re.compile(r"pacing_rate (?P<pacing_rate>\d+(\.\d+)?[MK]?bps)",
            #               re.MULTILINE),
            #    re.compile(r"delivery_rate (?P<delivery_rate>\d+(\.\d+)?[MK]?bps)",
            #               re.MULTILINE),
            #    re.compile(r"bbr:\(bw:(?P<bbr_bw>\d+(\.\d+)?[MK]?bps),"
            #               r"mrtt:(?P<bbr_mrtt>\d+\.\d+),"
            #               r"pacing_gain:(?P<bbr_pacing_gain>\d+(\.\d+)?),"
            #               r"cwnd_gain:(?P<bbr_cwnd_gain>\d+(\.\d+)?)\)",
            #               re.MULTILINE)]
            #print(data_res)
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
        except Exception as e:
            #pass
            logging.error("Oops !Failed to execute : " + str(e))
        sleep(interval_sec)
    return
def start_cubic(dst, interval_sec=0.1, outfile="cubic.txt", runner=None):
    monitor = Process(target=monitor_cubic,
                      args=(dst, interval_sec, outfile, runner))
    monitor.start()
    return monitor
def start_bbrmon(dst, interval_sec=0.1, outfile="bbr.txt", runner=None):
    #monitor = Process(target=monitor_bbr,
    #                  args=(dst, interval_sec, outfile, runner))
    monitor = Process(target=monitor_bbr_v2,
                      args=(dst, interval_sec, outfile, runner))
    monitor.start()
    return monitor

def signal_handler(sig, frame):
    #running = False
    for process in processes:
        os.kill(process.pid, signal.SIGTERM) 
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    #signal.signal(signal.SIGTERM, signal_handler) 
    default_dir = args.dir
    p1 =start_bbrmon(dst="10.22.116.221:5202")
    p2 =start_cubic(dst="10.22.116.221:5201")
    p3 =start_qmon(iface="enp61s0f1np1")
    #p1.start()
    processes.append(p1)
    processes.append(p2)
    processes.append(p3)
    p1.join()
    p2.join()
    p3.join()
