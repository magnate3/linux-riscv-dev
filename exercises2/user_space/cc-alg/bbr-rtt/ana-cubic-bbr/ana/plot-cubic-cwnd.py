# -*- coding: UTF-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FormatStrFormatter

#lines = ["btl_bw 63.3Mbps | mrtt 0.017 | pacing_rate 62.6Mbps | delivery_rate 13.1Mbps"]

btlbwv = []
pacingRatev = []
deliverRateV = []
cubic_cwndV = []
bbr_cwndV = []
cubic_timev = []
def deal_cubic():
    of = open("cubic.txt","r")
    cubics = of.readlines()
    of.close()
    #global cubic_timev = range(0,len(cubics))
    #print('len cubic_timev {}'.format(len(cubic_timev)))
    for line in cubics:
        #pass
        secs = line.strip("\n").split(":")
        #cubic_cwndV.append(int(secs[1])) 
        #print(secs)
        if int(secs[1]) < 100:
            cubic_cwndV.append(int(secs[1])) 
        #else:


    #print('len cubic_timev {}, len cubic cwndv {}'.format(len(cubic_timev),len(cubic_cwndV)))
# Normalize bandwidth fields to Mbps
def convert_to_mbps(value):
    try:
        value = str(value).strip().lower()
        if value.endswith('Gbps'):
            return float(value.replace('mbps', ''))*1000
        elif value.endswith('mbps'):
            return float(value.replace('mbps', ''))
        elif value.endswith('kbps'):
            return float(value.replace('kbps', ''))/1000
    except Exception:
        return 0.0

deal_cubic()
fig = plt.figure(figsize=(24,16))



cubic_timev = range(0,len(cubic_cwndV))


print('len cubic_timev {}, len cubic cwndv {}'.format(len(cubic_timev),len(cubic_cwndV)))
plt.plot(cubic_timev, cubic_cwndV, label='cwnd', color='blue', linestyle='-.')
plt.ylabel("cwnd")
plt.title("cwnd Curves of bbr")
plt.legend()

# Displaying the plot
#plt.show()
plt.tight_layout()
plt.savefig('cubic-cwnd.png')
plt.close()
