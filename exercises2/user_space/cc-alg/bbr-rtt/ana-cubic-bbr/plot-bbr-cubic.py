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
        #print(secs)
        cubic_cwndV.append(int(secs[1])) 
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
    secs = line.split("|")
    bw = convert_to_mbps(secs[0].strip(" ").split(" ")[1])
    btlbwv.append(bw)
    pacing_rating = convert_to_mbps(secs[2].strip(" ").split(" ")[1])
    pacingRatev.append(pacing_rating)
    deliver_rating = convert_to_mbps(secs[3].strip(" ").split(" ")[1])
    deliverRateV.append(deliver_rating)
    #print('{}Mbps,{}Mbps,{}Mbps'.format(bw, pacing_rating, deliver_rating))
    n = n +1

deal_cubic()
fig = plt.figure(figsize=(24,16))



#ax = plt.subplot(211)
#ax = plt.subplot2grid((3, 3), (0, 0), colspan=3)
#spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3,1])
#spec = fig.add_gridspec(nrows=2, ncols=1, width_ratios=[3, 1], height_ratios=[3,1])

# Plotting the curves
plt.plot(timev, btlbwv, label='btlbw', color='blue', linestyle='-.')
plt.plot(timev, pacingRatev, label='pacingRate', color='red', linestyle='--')
plt.plot(timev, deliverRateV, label='deliverRate', color='green', linestyle='-')

# Adding labels, title, and legend
#ax.xlabel("X-axis")
plt.ylabel("Mbps")
plt.tight_layout()
plt.savefig('bbr.png')
plt.close()

cubic_timev = range(0,len(cubic_cwndV))


print('len cubic_timev {}, len cubic cwndv {}'.format(len(cubic_timev),len(cubic_cwndV)))
#ax = plt.subplot(212)
#ax = plt.subplot2grid((3, 3), (1, 0), colspan=3)
#ax = fix.add_subplot(spec[1,0])

# Plotting the curves
fig = plt.figure(figsize=(24,16))
plt.plot(cubic_timev, cubic_cwndV, label='cwnd', color='blue', linestyle='-.')
# Adding labels, title, and legend
#ax.xlabel("X-axis")
plt.ylabel("cwnd")
plt.title("Multiple Curves of cubic ")
#plt.legend()

# Displaying the plot
#plt.show()
plt.tight_layout()
plt.savefig('cubic.png')
plt.close()
