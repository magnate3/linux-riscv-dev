#!/usr/bin/python3

import subprocess
import pprint
import time
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#inets = []
#with open('inets.conf') as f:
#    inets = f.read().strip().split('\n')
#    inets = [item.strip() for item in inets]
CSV_FILE = "fetched/ib_write_bw.csv"
col_Mbits = 'BW average[Mb/sec]'
col_Mbytes = 'BW average[MB/sec]'
col_lat_min = 't_min[usec]'
col_lat_median = 't_typical[usec]'
col_lat_max = 't_max[usec]'
col_size = '#bytes'
logroot = os.path.join(os.environ['HOME'], 'results', 'perftest')
def log_file_name(app, logid, role):
    return "../multiple-ib-process/logs/test_result_c144_bw"
    #return os.path.join(logroot, app, logid, '{}.{}.log'.format(app, role))
def size_formatter(x, pos):
    x = int(x)
    if x < 1000:
        return '{:d}'.format(x)
    elif x < 1000000:
        return '{} kB'.format(x // 1000)
    elif x < 1000000000:
        return '{} MB'.format(x // 1000000)
    else:
        return '{} GB'.format(x // 1000000000)
def find_table(fh):
    hdrline = re.compile('#bytes')
    last = 0
    line = fh.readline()
    while line:
        if hdrline.search(line):
            fh.seek(last)
            return
        last = fh.tell()
        line = fh.readline()
    raise ValueError
def load_data_frame(fn, header_names):
    with open(fn, 'r') as fh:
        find_table(fh)
        if header_names is not None:
            fh.readline()
            df = pd.read_table(fh, sep='[ \t]+', names=header_names,
                               skipfooter=1,engine='python')
        else:
            df = pd.read_table(fh, '[ \t]+', engine='python',
                               skipfooter=1)
        if col_Mbytes in df:
            df[col_Mbits] = df[col_Mbytes] * 8
        return df
def do_plot_bw(apps, df, logid):
    fig, ax = plt.subplots()
    df.plot(x=col_size, ax=ax, y=apps, title='Throughput', logx=True)
    ax.xaxis.set_major_formatter(FuncFormatter(size_formatter))
    ax.yaxis.set_label_text('Throughput (Mbps)')
    plt.savefig('bw-{}.pdf'.format(logid))
    #plt.savefig(os.path.join(os.environ['HOME'], 'results', 'perftest',
    #            'bw-{}.pdf'.format(logid)))
    #plt.savefig(os.path.join(os.environ['HOME'], 'results', 'perftest',
    #            'bw-{}.eps'.format(logid)))
def plot_throughput():
    logid = "2048"
    bw_tests = ['ib_write_bw']
    app = 'ib_write_bw'
    #bw_tests = ['ib_send_bw', 'ib_write_bw', 'ib_read_bw']
    df = pd.DataFrame(data={col_size: []})
    #  #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
    bw_header_names = [col_size, '#iterations', 'BW peak[MB/sec]', 'BW average[MB/sec]', 'MsgRate[Mpps]']
    path = "../../multiple-ib-process/logs/"
    files= os.listdir(path)
    data = pd.DataFrame(columns=[col_size, col_Mbits])
    for file in files: 
        #if not os.path.isdir(file) and os.path.splitext(file)[-1] == ".txt": 
        if not os.path.isdir(file): 
            fn= path+"/"+file
            #fn = log_file_name(app, logid, 'client')
            nextdf = load_data_frame(fn, bw_header_names)
            nextdf = nextdf.filter(items=[col_size, col_Mbits])
            #nextdf = nextdf.rename(columns={col_Mbits: app})
            #df = df.merge(nextdf, on=col_size, how='right')
            data=pd.concat([data,nextdf])
            #size = len(df)
            #for row in range(0,len(nextdf)):
            #    df.loc[size + row]= nextdf.loc[row]
            #df.append(nextdf)
            #df = pd.concat([df, nextdf])
            #print(df)

    df = data.rename(columns={col_Mbits: app})
    print(df)
    #df = df.merge(nextdf, on=col_size, how='right')
    print("total throughout : %f \n"%(df[app].mean()*128))
    #do_plot_bw(bw_tests, df, logid)
def plot_throughput_test():
    logid = "2048"
    bw_tests = ['ib_write_bw']
    app = 'ib_write_bw'
    #bw_tests = ['ib_send_bw', 'ib_write_bw', 'ib_read_bw']
    df = pd.DataFrame(data={col_size: []})
    #  #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
    bw_header_names = [col_size, '#iterations', 'BW peak[MB/sec]', 'BW average[MB/sec]', 'MsgRate[Mpps]']
    fn = log_file_name(app, logid, 'client')
    nextdf = load_data_frame(fn, bw_header_names)
    nextdf = nextdf.filter(items=[col_size, col_Mbits])
    nextdf = nextdf.rename(columns={col_Mbits: app})
    df = df.merge(nextdf, on=col_size, how='right')
    do_plot_bw(bw_tests, df, logid)

def test():
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
                     
                    #print('byte_nr: {}, iterations: {}, bw_peak: {}, bw_avg: {}, msg_rate: {}\n'.format(byte_nr, iterations, bw_peak, bw_avg, msg_rate))
            except Exception as e:
                print(e)
                print(filenames)
if __name__ == '__main__':
    #lat_re = re.compile(r"^[^:]*: *min= *(?P<min>[0-9.e]+), *" \
    #        r"max= *(?P<max>[0-9.e]+), *" r"avg= *(?P<avg>[0-9.e]+), *"\
    #        r"stdev= *(?P<stdev>[0-9.e]+)")
    plot_throughput()
