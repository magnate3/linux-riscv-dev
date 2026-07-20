'''
Plot queue occupancy over time
'''
from helper import *
import plot_defaults

from matplotlib.ticker import LinearLocator
from pylab import figure


parser = argparse.ArgumentParser()
parser.add_argument('--files', '-f',
                    help="Throughput timeseries output to one plot",
                    required=True,
                    action="store",
                    nargs='+',
                    dest="files")

parser.add_argument('--legend', '-l',
                    help="Legend to use if there are multiple plots.  File names used as default.",
                    action="store",
                    nargs="+",
                    default=None,
                    dest="legend")

parser.add_argument('--out', '-o',
                    help="Output png file for the plot.",
                    default=None, # Will show the plot
                    dest="out")

parser.add_argument('--labels',
                    help="Labels for x-axis if summarising; defaults to file names",
                    required=False,
                    default=[],
                    nargs="+",
                    dest="labels")

parser.add_argument('--xlimit',
                    help="Upper limit of x axis, data after ignored",
                    type=float,
                    default=50)

parser.add_argument('--every',
                    help="If the plot has a lot of data points, plot one of every EVERY (x,y) point (default 1).",
                    default=1,
                    type=int)

args = parser.parse_args()

if args.legend is None:
    args.legend = []
    for file in args.files:
        args.legend.append(file)

to_plot=[]
def get_style(i):
    if i == 0:
        return {'color': 'red'}
    if i == 1:
        return {'color': 'blue'}
    if i == 2:
        return {'color': 'green'}
    else:
        return {'color': 'black'}

m.rc('figure', figsize=(32, 12))
fig = figure()
ax = fig.add_subplot(111)
time_btwn_flows = 2.0
for i, f in enumerate(sorted(args.files)):
    data = read_list(f)
    xaxis = list(map(float, col(0, data)))
    throughput = list(map(float, col(1, data)))
    throughput = [t for j, t in enumerate(throughput)
                  if xaxis[j] <= args.xlimit]
    xaxis = [x for x in xaxis if x <= args.xlimit]

    ax.plot(xaxis, throughput, label=args.legend[i], lw=2, **get_style(i))
    ax.xaxis.set_major_locator(LinearLocator(6))

if args.legend is not None:
	plt.legend()
plt.ylabel("Throughput (Mbits)")
plt.grid(True)
plt.xlabel("Seconds")
plt.tight_layout()

if args.out:
    print('saving to {}'.format( args.out))
    plt.savefig(args.out)
else:
    plt.show()
