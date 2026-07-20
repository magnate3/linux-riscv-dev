'''
Plot ping RTTs over time
'''
from helper import *
import plot_defaults

from matplotlib.ticker import LinearLocator
from pylab import figure

parser = argparse.ArgumentParser()
parser.add_argument('--files', '-f',
                    help="Ping output files to plot",
                    required=True,
                    action="store",
                    nargs='+')

parser.add_argument('--xlimit',
                    help="Upper limit of x axis, data after ignored",
                    type=float,
                    default=8)

parser.add_argument('--out', '-o',
                    help="Output png file for the plot.",
                    default=None) # Will show the plot

args = parser.parse_args()

m.rc('figure', figsize=(32, 12))
fig = figure()
ax = fig.add_subplot(111)
for i, f in enumerate(args.files):
    data = read_list(f)
    xaxis = map(float, col(0, data))
    rtts = map(float, col(1, data))
    xaxis = [x - xaxis[0] for x in xaxis]
    rtts = [r * 1000 for j, r in enumerate(rtts)
            if xaxis[j] <= args.xlimit]
    xaxis = [x for x in xaxis if x <= args.xlimit]
    if "bbr" in args.files[i]:
	name = "bbr"
    else:
	name = "cubic"
    ax.plot(xaxis, rtts, lw=2, label=name)
    plt.legend()
    ax.xaxis.set_major_locator(LinearLocator(5))

plt.ylabel("RTT (ms)")
plt.xlabel("Seconds")
plt.grid(True)
plt.tight_layout()

if args.out:
    plt.savefig(args.out)
else:
    plt.show()
