import argparse
import subprocess
import time
import signal
import sys
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=10000, type=int, help="Number of iterations")
parser.add_argument('-p', default=1, type=int, help="Number of processes")
parser.add_argument('-s', default=False, action="store_true", help="Server mode?")
parser.add_argument('-c', default="10.0.0.2", help="client mode connect to server IP")
parser.add_argument('-i', default=1, type=float, help="interval")
parser.add_argument('-q', default=1, type=float, help="number of queue pairs")
parser.add_argument('-T', default=64, type=int, help="size of TX queue")
parser.add_argument('--cmd', default="ib_write_bw", choices=["ib_read_bw", "ib_write_bw"], help="Which command to use for test?")
parser.add_argument('--msize', default=64, type=int, help="message size")
parser.add_argument('--quiet', default=False)

FLAGS = parser.parse_args()

START_PORT = 18000
OUTPUT = ""
NUM_CPUS = 8
pat_spaces = re.compile(r'\s+')

if FLAGS.quiet:
    OUTPUT = "> /dev/null 2>&1"

def get_qpairs():
    qpairs = ''
    if 'read' in FLAGS.cmd:
        qpairs = "-o %d" % FLAGS.q
    else:
        qpairs = '-q %d' % FLAGS.q
    return qpairs

def start_servers():
    procs = []
    for i in xrange(FLAGS.p):
        cmd = "taskset -c %d %s -x 0 -p %d -n %d -s %d %s -t %d %s" % \
            (i % NUM_CPUS, FLAGS.cmd, START_PORT + i, FLAGS.n, FLAGS.msize, get_qpairs(), FLAGS.T, OUTPUT)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=65536)
        procs.append(proc)
    return procs

def start_clients():
    procs = []
    for i in xrange(FLAGS.p):
        cmd = "taskset -c %d %s -x 0 -p %d -n %d -s %d %s -t %d %s %s" % \
            (i % NUM_CPUS, FLAGS.cmd, START_PORT + i, FLAGS.n, FLAGS.msize, get_qpairs(), FLAGS.T, FLAGS.c, OUTPUT)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=65536)
        procs.append(proc)
    return procs

def cleanup():
    os.system("killall -9 ib_write_cmd ib_read_cmd > /dev/null 2>&1")

def sigint_handler(*args):
    cleanup()
    sys.exit(-1)

def read_counter(f):
    f.seek(0)
    return int(f.read())

def main():
    signal.signal(signal.SIGINT, sigint_handler)
    cleanup()
    time.sleep(3)
    procs = []
    if FLAGS.s:
        procs = start_servers()
    else:
        procs = start_clients()

    avges = []
    for p in procs:
        p.wait()
        data = p.stdout.read()
        #print data
        try:
            lastline = data.split('\n')[-3]
            data = pat_spaces.split(lastline.strip())
            #print '*******************', data[-1]
            avges.append(float(data[-1]) * 8.0)
        except:
            continue
    print "Averages from each process in Mb/s: ", avges
    print "Total of averages in Gb/s:          ", sum(avges)/1e3
    cleanup()

if __name__ == "__main__":
    main()
