#!/usr/bin/python3

import subprocess
import pprint
import time
import os

#inets = []
#with open('inets.conf') as f:
#    inets = f.read().strip().split('\n')
#    inets = [item.strip() for item in inets]
CSV_FILE = "fetched/ib_write_bw.csv"

if __name__ == '__main__':
    #lat_re = re.compile(r"^[^:]*: *min= *(?P<min>[0-9.e]+), *" \
    #        r"max= *(?P<max>[0-9.e]+), *" r"avg= *(?P<avg>[0-9.e]+), *"\
    #        r"stdev= *(?P<stdev>[0-9.e]+)")
    with open(CSV_FILE, "w") as f:
        f.write("tx-length,qpn,size,BW-average(MB/s),MsgRate(Mpps)\n")
    path = "./log"
    files= os.listdir(path)
    for file in files: 
        if not os.path.isdir(file) and os.path.splitext(file)[-1] == ".txt": 
            filenames = path+"/"+file
            try:
                with open(filenames,'r', encoding="utf8") as log:
                    text = log.read()
                    #text = log.read().decode("ascii")
                    ls = text.split('\n')
                    #print(ls)
                    byte_nr, iterations, bw_peak, bw_avg, msg_rate = ls[-3].split()
                    #sum_bw += float(bw_avg)
                    #sum_msg_rate += float(msg_rate)
                     
                    #print('{}, {},{},{},{}\n'.format(byte_nr, iterations, bw_peak, bw_avg, msg_rate))
                    print('byte_nr: {}, iterations: {}, bw_peak: {}, bw_avg: {}, msg_rate: {}\n'.format(byte_nr, iterations, bw_peak, bw_avg, msg_rate))
            except Exception as e:
                print(e)
                print(filenames)
    #with open("fetched/ib_write_bw.csv", "a") as f:
    #    f.write('{}, {},{},{},{}\n'.format(
    #        tx_length, qpn, size, sum_bw, sum_msg_rate))
