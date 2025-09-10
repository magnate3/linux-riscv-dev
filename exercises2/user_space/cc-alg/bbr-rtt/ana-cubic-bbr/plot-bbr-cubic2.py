# -*- coding: UTF-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FormatStrFormatter

of = open(sys.argv[1],"r")
lines = of.readlines()
of.close()

#lines = ["btl_bw 63.3Mbps | mrtt 0.017 | pacing_rate 62.6Mbps | delivery_rate 13.1Mbps"]

btlbwv = []
pacingRatev = []
deliverRateV = []
cubic_cwndV = []
bbr_cwndV = []

modes = []
lastmode = ""
n = 0

timev = range(0,len(lines))
cubic_timev = []
def deal_cubic():
    of = open("cubic.txt","r")
    cubics = of.readlines()
    of.close()
    #global cubic_timev = range(0,len(cubics))
    print('len cubic_timev {}'.format(len(cubic_timev)))
    for line in cubics:
        #pass
        secs = line.strip("\n").split(":")
        cubic_cwndV.append(int(secs[1])) 
        #print(secs)
        #if int(secs[1]) < 100:
        #    cubic_cwndV.append(int(secs[1])) 
        #else:


    print('len cubic_timev {}, len cubic cwndv {}'.format(len(cubic_timev),len(cubic_cwndV)))
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
for line in lines:
    print(line)
    secs = line.split("|")
    bw = convert_to_mbps(secs[0].strip(" ").split(" ")[1])
    btlbwv.append(bw)
    pacing_rating = convert_to_mbps(secs[2].strip(" ").split(" ")[1])
    pacingRatev.append(pacing_rating)
    deliver_rating = convert_to_mbps(secs[3].strip(" ").split(" ")[1])
    deliverRateV.append(deliver_rating)
    bbr_cwnd = int(secs[4].strip(" ").split(" ")[1])
    #print(bbr_cwnd)
    bbr_cwndV.append(bbr_cwnd)
    #print('{}Mbps,{}Mbps,{}Mbps'.format(bw, pacing_rating, deliver_rating))
    n = n +1

deal_cubic()
fig = plt.figure(figsize=(24,16))



#ax = plt.subplot(211)
# 创建一个子图，占据3行3列网格的第0行所有3列
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
#spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3,1])
#spec = fig.add_gridspec(nrows=2, ncols=1, width_ratios=[3, 1], height_ratios=[3,1])

# Plotting the curves
ax1.plot(timev, btlbwv, label='btlbw', color='blue', linestyle='-.')
ax1.plot(timev, pacingRatev, label='pacingRate', color='red', linestyle='--')
ax1.plot(timev, deliverRateV, label='deliverRate', color='green', linestyle='-')

# Adding labels, title, and legend
#ax.xlabel("X-axis")
ax1.set_title("Multiple Curves of bbr")
ax1.set_ylabel("Mbps")

cubic_timev = range(0,len(cubic_cwndV))


print('len cubic_timev {}, len cubic cwndv {}'.format(len(cubic_timev),len(cubic_cwndV)))
#ax = plt.subplot(212)
## 创建一个子图，占据第1行的前2列
#ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
## 创建一个子图，从第1行第3列开始，纵向跨越2行
#ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
## 创建一个子图，在第2行第1列
#ax4 = plt.subplot2grid((3, 3), (2, 0))
## 创建一个子图，在第2行第2列
#ax5 = plt.subplot2grid((3, 3), (2, 1))
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
#ax = fix.add_subplot(spec[1,0])

# Plotting the curves
ax2.plot(cubic_timev, cubic_cwndV, label='cwnd', color='blue', linestyle='-.')
# Adding labels, title, and legend
#ax.xlabel("X-axis")
ax2.set_ylabel("cwnd")
ax2.set_title("cwnd Curves of cubic ")

ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
ax3.plot(timev, bbr_cwndV, label='cwnd', color='blue', linestyle='-.')
ax3.set_ylabel("cwnd")
ax3.set_title("cwnd Curves of bbr")
#plt.legend()

# Displaying the plot
#plt.show()
plt.tight_layout()
plt.savefig('cubic-bbr.png')
plt.close()
