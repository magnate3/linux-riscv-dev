#!/usr/bin/env python3
import asyncio
import io
import os
import shutil
import statistics
import subprocess
import threading
from argparse import ArgumentParser

import ijson
import matplotlib.pyplot as plt
import numpy as np


def group(data, interval):
    records_fed = 0
    bits = 0
    max_time = data[0][0] + interval
    for time, length in data:
        if time >= max_time:
            return records_fed, bits
        records_fed += 1
        bits += length * 8
    return records_fed, bits


def plot(data: dict):
    bw_bps = []

    for frame_number in data:
        point = data[frame_number]
        if len(point['acks_rtt']) >= 1:
            mean_rtt = statistics.mean(point['acks_rtt'])
            estimated_bps = point['tcp_window_size'] / mean_rtt
            bw_bps.append((point['time'], estimated_bps))

    start_time = bw_bps[0][0]
    plt.step([x - start_time for x, _ in bw_bps],
             [y / (10 ** 6) for _, y in bw_bps],
             label='iperf3')

    plt.legend(loc='best')
    plt.ylabel('Estimated Bandwidth (Mbps)')
    plt.xlabel('Time (s)')
    plt.grid()
    #plt.show()
    plt.tight_layout()
    plt.savefig("bdp.png")


def do_wait(proc: subprocess.Popen, write_pipe):
    proc.wait()
    os.close(write_pipe)


def main(args):
    r, w = os.pipe()
    tshark = shutil.which('tshark')
    if tshark is None:
        raise ValueError('tshark not found')

    cmd = [tshark, '-r', args.pcap, '-T', 'json', '-e', 'frame.len',
           '-e', 'frame.number', '-e', 'frame.time_epoch',
           '-e', 'ip.src', '-e', 'ip.dst', '-e', 'tcp.window_size',
           '-e', 'tcp.analysis.acks_frame', '-e', 'tcp.analysis.ack_rtt',
           '-e', 'tcp.analysis.bytes_in_flight']
    tsh_p = subprocess.Popen(cmd, shell=False, stdout=w)

    thread = threading.Thread(target=do_wait, args=(tsh_p, w))
    thread.start()

    data = {}

    acked_segments = 0
    sent_data = 0

    with open(r, 'r', encoding='utf-8') as f:
        for record in ijson.items(f, 'item'):
            layers = record['_source']['layers']

            if 'tcp.window_size' not in layers:
                continue

            time = float(layers['frame.time_epoch'][0])
            length = int(layers['frame.len'][0])
            frame = int(layers['frame.number'][0])
            src = layers['ip.src'][0]
            dst = layers['ip.dst'][0]

            if src == args.src and dst == args.dst:
                if 'tcp.analysis.bytes_in_flight' not in layers:
                    continue
                tcp_window_size = int(layers['tcp.analysis.bytes_in_flight'][0]) * 8  # to bits
                if frame in data:
                    raise ValueError('Duplicate frame {}'.format(frame))
                sent_data += 1
                data[frame] = {
                    'time': time,
                    'length': length,
                    'tcp_window_size': tcp_window_size,
                    'acks_rtt': []
                }

            # check acks
            if dst == args.src and src == args.dst:
                if 'tcp.analysis.acks_frame' not in layers:
                    continue
                acked_segments += 1
                acked_frame = int(layers['tcp.analysis.acks_frame'][0])
                ack_rtt = float(layers['tcp.analysis.ack_rtt'][0])

                if acked_frame in data:
                    data[acked_frame]['acks_rtt'].append(ack_rtt)

    print(f"acked_segments {acked_segments}")
    print(f"sent_segments {sent_data}")
    thread.join()
    print('plot')
    plot(data)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parse bandwidth pcap')
    parser.add_argument('-f', '--file', dest='pcap', required=True,
                        help="Pcap file to parse")
    parser.add_argument('-s', '--src', dest='src', required=True,
                        help="IP source")

    parser.add_argument('-d', '--dst', dest='dst', required=True,
                        help="IP destination")

    main(parser.parse_args())
