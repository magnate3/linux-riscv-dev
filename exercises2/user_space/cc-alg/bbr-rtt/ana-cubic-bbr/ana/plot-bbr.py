import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FormatStrFormatter

of = open(sys.argv[1],"r")
lines = of.readlines()
of.close()


bdp = []
mrttv = []
cwndgainv = []
rtpropv = []
btlbwv = []
pacinggainv = []
pacingratev = []
inflightv = []

modes = []
lastmode = ""
n = 0

timev = range(0,len(lines))
for line in lines:
    line = line.split(",")
    BtlBw = float(0)
    #print(line[0])
    if -1  != line[0].find("Gbps"):
        BtlBw = float(line[0].replace("Gbps",""))*1024
    elif -1 != line[0].find("Mbps"):
        BtlBw = float(line[0].replace("Mbps",""))
    #BtlBw = float(line[0].replace("Gbps","").replace("Mbps",""))
    mrtt = float(line[1])
    # SECS = RTprop/100
    pacing_gain = float(line[2])
    #pacing_rate = float(line[3])
    cwnd_gain = float(line[3])
    cwndgainv.append(cwnd_gain)
    mrttv.append(mrtt)
    btlbwv.append(BtlBw)
    pacinggainv.append(pacing_gain)
    #pacingratev.append(pacing_rate)
    n = n +1

plt.figure(figsize=(24,16))


'''
ax = plt.subplot(222)
ax.set_title("BDP/INFLIGHT")
ax.text(timev[5],bdp[5], r'BDP', fontsize=10,color="red", verticalalignment='bottom', horizontalalignment='left')
ax.text(timev[10],inflightv[10], r'INFLIGHT', fontsize=10,color="green", verticalalignment='bottom', horizontalalignment='right')
ax.plot(timev,bdp,c="red")
ax.plot(timev,inflightv,c="green")
ylimv = max(max(bdp),max(inflightv))*1.5
ax.set_ylim(0,ylimv)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for mode in modes:
	plt.axvline(mode, c='grey', label='status')
'''
ax = plt.subplot(221)
ax.set_title("mrtt")
ax.plot(timev,mrttv,c="red")
for mode in modes:
	plt.axvline(mode, c='grey', label='status')
ax.set_ylabel("ms")
#ax.set_ylim(1,20)
ax.set_yscale("log")

ax = plt.subplot(222)
ax.set_title("CWND")
ax.plot(timev,cwndgainv)
for mode in modes:
	plt.axvline(mode, c='grey', label='status')
ax.set_ylim(0.5,3)

ax = plt.subplot(223)
ax.set_title("BTlBw")
ax.set_ylabel("MBytes")
ax.plot(timev,btlbwv,c="green")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.set_ylim(0,ylimv)
for mode in modes:
	plt.axvline(mode, c='grey', label='status')


ax = plt.subplot(224)
ax.set_title("GAIN")
ax.plot(timev,pacinggainv)
for mode in modes:
	plt.axvline(mode, c='grey', label='status')
ax.set_ylim(0.5,3)
plt.tight_layout()
plt.savefig('results.png')
plt.close()
