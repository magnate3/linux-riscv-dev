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
inflightv = []

modes = []
lastmode = ""
n = 0

timev = range(0,len(lines))
def deal_cubic():
    of = open("cubic.txt","r")
    lines = of.readlines()
    of.close()
    for line in lines:
        pass
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

plt.figure(figsize=(24,16))


# Sample data
#x = np.linspace(0, 2 * np.pi, 100)
#y1 = np.sin(x)
#y2 = np.cos(x)
#y3 = np.sin(2 * x)

# Plotting the curves
plt.plot(timev, btlbwv, label='btlbw', color='blue', linestyle='-.')
plt.plot(timev, pacingRatev, label='pacingRate', color='red', linestyle='--')
plt.plot(timev, deliverRateV, label='deliverRate', color='green', linestyle='-')

# Adding labels, title, and legend
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Multiple Curves on a Single Plot")
plt.legend()

# Displaying the plot
#plt.show()
plt.tight_layout()
plt.savefig('results2.png')
plt.close()
