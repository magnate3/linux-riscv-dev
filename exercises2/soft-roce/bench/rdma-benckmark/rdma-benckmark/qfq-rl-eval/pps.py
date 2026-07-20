import argparse
import subprocess
import time
import signal
import sys
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('-i', default=1, type=float, help="interval")
parser.add_argument('--dir', default="tx", choices=["tx", "rx"], help="which direction?")
parser.add_argument('-p', default=False, action="store_true", help="store only +ves")
FLAGS = parser.parse_args()

COUNTER_FILE = '/sys/class/infiniband/mlx4_0/ports/1/counters/port_xmit_packets'
if FLAGS.dir == "rx":
    COUNTER_FILE = '/sys/class/infiniband/mlx4_0/ports/1/counters/port_rcv_packets'
U32_MAX = (1 << 32) - 1

def sigint_handler(*args):
    sys.exit(-1)

def read_counter(f):
    f.seek(0)
    return int(f.read())

def main():
    signal.signal(signal.SIGINT, sigint_handler)
    prev = None
    prev_t = time.time()
    with open(COUNTER_FILE, 'r') as f:
        while True:
            t = time.time()
            dt = t - prev_t
            curr = read_counter(f)
            if prev is None:
                prev = curr

            diff = curr - prev
            if diff < 0:
                diff += U32_MAX # Wrapped around.  It can happen at most once as long as FLAGS.i <= 1s.

            rate = diff / dt / 1e6
            if FLAGS.p == False or rate > 0:
                print "%.3f %d %.3f" % (t, curr, rate)
            time.sleep(FLAGS.i)
            prev = curr
            prev_t = t

if __name__ == "__main__":
    main()
