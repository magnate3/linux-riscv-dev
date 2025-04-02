import subprocess
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import time
from matplotlib.backends.backend_pdf import PdfPages
LS = [
    'solid',
    'dashed',
    'dotted',
    'dashdot'
]

def get_pctl(a, p):
        i = int(len(a) * p)
        return a[i]

def get_steps_from_raw2(prefix,cc,type,time_limit, step=5):
    #filename = f"../mix/output/{id}/{id}_out_fct.txt"
    filename = "../simulation/mix/%s_%s.txt"%(args.prefix, cc)
    time_start = int(2.000 * 1000000000)
    #time_start = int(1.800 * 1000000000)
    time_end = int(3.0 * 1000000000) 
    cmd_slowdown = "cat %s"%(filename)+" | awk '{ if ($6>"+"%d"%time_start+" && $6+$7<"+"%d"%(time_end)+") { slow=$7/$8; print slow<1?1:slow, $5} }' | sort -n -k 2"    
    print(cmd_slowdown)
    output_slowdown = subprocess.check_output(cmd_slowdown, shell=True)
    aa = output_slowdown.decode("utf-8").split('\n')[:-2]
    nn = len(aa)
    if 0 == nn:
        print("dataset len %d"%nn)
        exit(0)
    # CDF of FCT
    res = [[i/100.] for i in range(0, 100, step)]
    for i in range(0,100,step):
        l = int(i * nn / 100)
        r = int((i+step) * nn / 100)
        fct_size = aa[l:r]
        fct_size = [[float(x.split(" ")[0]), int(x.split(" ")[1])] for x in fct_size]
        fct = sorted(map(lambda x: x[0], fct_size))
        
        res[int(i/step)].append(fct_size[-1][1]) # flow size
        
        res[int(i/step)].append(sum(fct) / len(fct)) # avg fct
        res[int(i/step)].append(get_pctl(fct, 0.5)) # mid fct
        res[int(i/step)].append(get_pctl(fct, 0.95)) # 95-pct fct
        res[int(i/step)].append(get_pctl(fct, 0.99)) # 99-pct fct
        res[int(i/step)].append(get_pctl(fct, 0.999)) # 99-pct fct
    

    result = {"avg": [], "p99": [], "size": []}
    for item in res:
        if item[1] / (10 ** 3) <= 400:
            result["avg"].append(item[2])
            result["p99"].append(item[5])
            result["size"].append(item[1] / (10 ** 3))

    return result
def get_steps_from_raw(prefix,cc,type,time_limit, step=5):
    #file = "%s_%s.txt"%(args.prefix, cc)
    file = "../simulation/mix/%s_%s.txt"%(args.prefix, cc)
    print(file)
    if type == 0:
            cmd = "cat %s"%(file)+" | awk '{if ($4==100 && $6+$7<"+"%d"%time_limit+") {slow=$7/$8;print slow<1?1:slow, $5}}' | sort -n -k 2"
            # print cmd
            output = subprocess.check_output(cmd, shell=True)
    elif type == 1:
            cmd = "cat %s"%(file)+" | awk '{if ($4==200 && $6+$7<"+"%d"%time_limit+") {slow=$7/$8;print slow<1?1:slow, $5}}' | sort -n -k 2"
            #print cmd
            output = subprocess.check_output(cmd, shell=True)
    else:
            cmd = "cat %s"%(file)+" | awk '{$6+$7<"+"%d"%time_limit+") {slow=$7/$8;print slow<1?1:slow, $5}}' | sort -n -k 2"
            #print cmd
            output = subprocess.check_output(cmd, shell=True)

    # up to here, `output` should be a string of multiple lines, each line is: fct, size
    # python3
    #a = output.decode("utf-8").split('\n')[:-2]
    a = output.split('\n')[:-2]
    n = len(a)
    res = [[i/100.] for i in range(0, 100, step)]
    for i in range(0,100,step):
            l = i * n / 100
            r = (i+step) * n / 100
            d = map(lambda x: [float(x.split()[0]), int(x.split()[1])], a[l:r])
            fct=sorted(map(lambda x: x[0], d))
            res[i/step].append(d[-1][1]) # flow size
            res[i/step].append(sum(fct) / len(fct)) # avg fct
            res[i/step].append(get_pctl(fct, 0.5)) # mid fct
            res[i/step].append(get_pctl(fct, 0.95)) # 95-pct fct
            res[i/step].append(get_pctl(fct, 0.99)) # 99-pct fct
    result = {"avg": [], "p99": [], "size": []}
    for item in res:
        if item[1] / (10 ** 3) <= 400:
            result["avg"].append(item[2])
            result["p99"].append(item[5])
            result["size"].append(item[1] / (10 ** 3))

    return result
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', dest='prefix', action='store', default='fct_topology_flow', help="Specify the prefix of the fct file. Usually like fct_<topology>_<trace>")
    parser.add_argument('-s', dest='step', action='store', default='5')
    parser.add_argument('-t', dest='type', action='store', type=int, default=0, help="0: normal, 1: incast, 2: all")
    parser.add_argument('-T', dest='time_limit', action='store', type=int, default=3000000000, help="only consider flows that finish before T")
    parser.add_argument('-b', dest='bw', action='store', type=int, default=25, help="bandwidth of edge link (Gbps)")
    args = parser.parse_args()

    type = args.type
    time_limit = args.time_limit

    # Please list all the cc (together with parameters) that you want to compare.
    # For example, here we list two CC: 1. HPCC-PINT with utgt=95,AI=50Mbps,pint_log_base=1.05,pint_prob=1; 2. HPCC with utgt=95,ai=50Mbps.
    # For the exact naming, please check ../simulation/mix/fct_*.txt output by the simulation.
    #CCs = [
    #       'hpccPint95ai50log1.05p1.000',
    #       'hp95ai50',
    #]
    #CCs = [ "dcqcn" ]
    step = int(args.step)
    cc = "dcqcn"
    cc_result = get_steps_from_raw2(args.prefix,cc,type,time_limit,step)
    with PdfPages('plot.pdf') as pdf:
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("Flow Size (KB)", fontsize=11.5)
        ax.set_ylabel("Avg FCT Slowdown", fontsize=11.5)
        xvals = [i for i in range(step, 100 + step, step)]
        ax.plot(cc_result["size"], cc_result["avg"], markersize=1.0, linewidth=3.0, label="{}".format(cc), linestyle=LS[0], color=(102 / 255, 8 / 255, 116 / 255))

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        # 0.0,1.2
        ax.legend(bbox_to_anchor=(0.7, 0.3), loc="upper left", borderaxespad=0,
                frameon=False, fontsize=12, facecolor='white', ncol=1,
                labelspacing=0.4, columnspacing=0.8)

        ax.set_xscale('log')
        log_ticks = np.log10([1, 10, 10 ** 2])
        # log_ticks = np.log10([1, 10, 10 ** 2, 10 ** 3, 2 * (10 ** 3)]) 
        ax.set_xticks(10 ** log_ticks)
        ax.set_xticklabels([r"$1$", r"$10$", r"$100$"])
        # ax.set_xticklabels([r"$1$", r"$10$", r"$100$", r"$1000$", r"$2000$"])

        ax.grid(True, which='major')

        ax.grid(False, which='minor')

        ax.grid(True, which='major', alpha=0.2)
        # ax.set_ylim(1.4, 2.2)

        plt.tight_layout()     
        pdf.savefig(fig)    
        # plt.show()
        plt.savefig("plot.png")
        time.sleep(1)
