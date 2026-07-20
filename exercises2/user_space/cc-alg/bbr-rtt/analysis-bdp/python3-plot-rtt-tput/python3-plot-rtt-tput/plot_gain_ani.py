'''
Plot queue occupancy over time
'''
from helper import *
import plot_defaults

from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pylab import figure
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np

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
ani = None

for i, f in enumerate(sorted(args.files)):
    data = read_list(f)
    xaxis = list(map(float, col(0, data)))
    min_value = min(xaxis)
    xaxis = list(map(lambda x: x - min_value, xaxis))
    pace_gain = list(map(float, col(1, data)))
    pace_gain = [t for j, t in enumerate(pace_gain)
                  if xaxis[j] <= args.xlimit]
    deliver_gain = list(map(float, col(2, data)))
    deliver_gain = [t for j, t in enumerate(deliver_gain)
                  if xaxis[j] <= args.xlimit]
    xaxis = [x for x in xaxis if x <= args.xlimit]

    line1, =ax.plot(xaxis, pace_gain, label="pace_gain", lw=2, **get_style(i))
    line2, =ax.plot(xaxis, deliver_gain, label="cwnd_gain", lw=2, **get_style(i+1))
    #ax.xaxis.set_major_locator(LinearLocator(6))
    
    #plt.ylim(0, 4)
    yminorLocator = MultipleLocator(0.25)
    ax.yaxis.set_minor_locator(yminorLocator)
    def update(num, x, y1,y2, line1,line2):
        line1.set_data(x[:num], y1[:num])
        line2.set_data(x[:num], y2[:num])
        return (line1, line2)
    ani = animation.FuncAnimation(fig, update,  frames=len(xaxis), interval=100,fargs=[xaxis,pace_gain,deliver_gain,line1,line2],blit=True)
    #ani = animation.FuncAnimation(fig, update,  frames=np.arange(1, 20), interval=100,fargs=[xaxis,pace_gain,deliver_gain,line1,line2],blit=True)
    #ani = FuncAnimation(fig, animate, interval=100, frames=int(len(xaxis)/10))
    #ani = animation.FuncAnimation(fig=fig,
    #                          animate,
    #                          #func=lambda i: animate(i),  # 指定动画函数
    #                          #func=lambda i: animate(i, pace_gain, xaxis),  # 指定动画函数
    #                          frames=np.arange(1, 200),  # 设置动画帧数
    #                          #init_func=lambda: init(pace_gain, xaxis),  # 指定初始化函数
    #                          interval=20,  # 设置更新间隔（毫秒）
    #                          blit=False)  # blit=True时仅重绘变化部分以提高效率，但在此设为False以保证兼容性


if args.legend is not None:
	plt.legend()
plt.ylabel("gain")
plt.grid(True)
plt.xlabel("Seconds")
#plt.tight_layout()

if args.out:
    print('saving to {}'.format( args.out))
    #plt.savefig(args.out)
    #ani.save(filename=args.out+".gif", writer="pillow")
    ani.save(args.out+".gif", writer='imagemagick', fps=60)
else:
    plt.show()
