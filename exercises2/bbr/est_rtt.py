from random import gauss, uniform
import matplotlib.pyplot as plt
import sys
import numpy as np
alpha = .125
beta = 0.25
EstimatedRTT = 1.0
DevRTT = 0.0
samples = []
estimates = []
tointervals = []
devRtts = []
font_size = 15
def plot_rtt(time_list, SampleRTT, EstimatedRTT, DeviationRTT, TimeoutInterval):
    plt.plot(time_list, SampleRTT, color='red', label='SampleRTT')
    plt.plot(time_list, EstimatedRTT, color='green', label='EstimatedRTT')
    plt.plot(time_list, DeviationRTT, color='blue', label='DeviationRTT')
    plt.plot(time_list, TimeoutInterval, color='black', label='TimeoutInterval')
    plt.legend()
    plt.xlabel('Time (s)', fontdict={'size':font_size})
    plt.ylabel('Time (ms)', fontdict={'size':font_size})
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('rtt.png', dpi=600)
    #plt.cla()
for i in range(100):
    SampleRTT = uniform(-0.5,.5) + 1.5
    EstimatedRTT = (1.0-alpha) * EstimatedRTT + alpha * SampleRTT
    DevRTT = (1-beta) * DevRTT + beta * abs(SampleRTT-EstimatedRTT)
    TimeoutInterval = EstimatedRTT + 4 * DevRTT
    samples.append(SampleRTT)
    estimates.append(EstimatedRTT)
    tointervals.append(TimeoutInterval)
    devRtts.append(DevRTT)
plot_rtt(range(100), samples,estimates,devRtts,tointervals)
