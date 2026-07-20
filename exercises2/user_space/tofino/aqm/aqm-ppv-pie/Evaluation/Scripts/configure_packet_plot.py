###############################################################################
 # A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 # Karlstad University 2021.
 # Author: L. Dahlberg
###############################################################################

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
import sys
import warnings
import numpy as np
from statistics import mean
import getpass
from scipy.stats import norm

# Edit corresponding with the enviornment the code is run in.
if getpass.getuser() == "dv":
    path = "/home/dv/pythonEnv/Notebook_Love/"
    pdfPath = path + "Results/Packet_result/"
    if path + "Scripts/" not in sys.path:
        sys.path.append(path + "Scripts/")

if getpass.getuser() == "love":
    path = "/home/love/tofino-master/tofino-master-thesis/Current_project/"
    pdfPath = path + "Evaluation/Results/Packet_result/"
    if path + "Evaluation/Scripts/" not in sys.path:
        sys.path.append(path + "Evaluation/Scripts/")

from configure_packet_data import GeneratePlots
from Plot import SetPlotOptions, Plot

def SessionDelayPlot(Plots, Names, PlotOrder, save, pp, testname, version):
    """
    Plot the delay for given graphs with the delay in the identification field.

    Args:
        Plots (dict): dict of all plots.
        Names (list): list of the names of plots to plot.  
        PlotOrder (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.
        testname (str): if set to "Overview_r200Mus" and "Overview_r2ms" plot cdf.
        version (str): if set to "Ev2" plot comparison cdf.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    if(Names == []):
        return PlotOrder
    if(testname != "Overview_r200Mus" and testname != "Overview_r2ms"):
            
        options = {
            "Gold"      : ["goldenrod", "Gold traffic" ],
            "Silver"    : ["silver", "Silver traffic" ]  
            }
        plt.figure(PlotOrder, figsize = (10,5))
        delay = []
        for name in Names:
            if(type(testname) is list and name.split("_")[0] == "Overview"):
                continue
            delay.append(float(sum(Plots[name].y)/len(Plots[name].y)))
            Plots[name].plot(color = options[Plots[name].type.split("\n")[0]][0])
        SetPlotOptions(options,
                    title = "Queing delay",
                    xlabel = "Time(s)",
                    ylabel = "Delay(ms)",
                    legend = True)
        PlotOrder += 1
        if(save):
            pp.savefig()
        print("Average delay {:g} ms".format(sum(delay)/len(delay)))
        
        if(version == "Ev2"):
            Values = {}
            LineOption = {
                "dynamic" : ["darkred", "Dynamic flow"],
                "static"  : ["green", "Static flow"]
            }
            for name in Names:
                entry = name.split("_")[0]
                if(entry not in Values):
                    Values[entry] = [Plots[name].y[:], LineOption["dynamic"][0] if entry == "TEST" else LineOption["static"][0]]
                else:
                    Values[entry][0].extend(Plots[name].y[:])
            plt.figure(PlotOrder, figsize = (10,5))
            for name in Values:
                Values[name][0].sort()
                Plots[name + "CDF"] = Plot(x = Values[name][0], y = 1. * np.arange(len(Values[name][0])) / (len(Values[name][0]) - 1))
                Plots[name + "CDF"].plot(color = Values[name][1])
            SetPlotOptions(LineOption,
                title = "Average CDF",
                xlabel = "Queuing Delay (ms)",
                ylabel = "CDF",
                legend = True)
            plt.ylim(0,1.1)
            if("Overview_r200Mus" in testname[1]):
                Plots["200µs"] = Plot(x = [0.2, 0.2], y = [0,2])
                Plots["200µs"].plot(color = "black")
                plt.xlim(0,2)
            elif("Overview_r2ms" in testname[1]):
                Plots["2ms"] = Plot(x = [2, 2], y = [0,2])
                Plots["2ms"].plot(color = "black")
                plt.xlim(0,2.5)
            if(save):
                pp.savefig()
            PlotOrder += 1
    else:
        options = {
            "m13_c14_t1Mus" : ["midnightblue","m13, c14, t1µs"],
            "m13_c14_t5ms" : ["darkred", "m13, c14, t5ms"],
            "m8_c10_t1Mus" : ["green", "m8, c10, t1µs"],
            "m8_c10_t5ms" : ["indigo","m8, c10, t5ms"]
        }
        LineOption = {
            "1" : ["solid", "1g and 1s"],
            "5" : ["dashdot", "5g and 5s"],
            "10" : ["dashed", "10g and 10s"],
            "50" : ["dotted", "50g and 50s"]
        }
        Values = {}
        for name in Names:
            entry = name.split("_")[2] + "-" + [i for i in options.keys() if i in name][0]
            if(entry not in Values):
                Values[entry] = [Plots[name].y[:], "solid"]
                if(name.split("_")[2] in LineOption):
                    Values[entry][1] = LineOption[name.split("_")[2]][0]
            else:
                Values[entry][0].extend(Plots[name].y[:])
        plt.figure(PlotOrder, figsize = (10,5))
        for name in Values:
            Values[name][0].sort()
            Plots[name + "CDF"] = Plot(x = Values[name][0], y = 1. * np.arange(len(Values[name][0])) / (len(Values[name][0]) - 1))
            Plots[name + "CDF"].plot(color = options[name.split("-")[1]][0], line = Values[name][1])
        SetPlotOptions(LineOption,
                    title = "Average CDF",
                    xlabel = "Queuing Delay (ms)",
                    ylabel = "CDF",
                    legend = True,
                    line = True)
        plt.ylim(0,1.1)
        if(testname == "Overview_r200Mus"):
            Plots["200µs"] = Plot(x = [0.2, 0.2], y = [0,2])
            Plots["200µs"].plot(color = "black")
            plt.xlim(0,2)
        elif(testname == "Overview_r2ms"):
            Plots["2ms"] = Plot(x = [2, 2], y = [0,2])
            Plots["2ms"].plot(color = "black")
            plt.xlim(0,5)
        PlotOrder += 1 
        if(save):
            pp.savefig()
    return PlotOrder

def SessionThroughputPlot(Plots, Names, PlotOrder, save, pp, testname):
    """

    Plot the throughput for given graphs with the delay in the identification field.

    Args:
        Plots (dict): dict of all plots.
        Names (list): list of the names of plots to plot.  
        PlotOrder (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.
        testname (str): if set to "Overview_r200Mus" and "Overview_r2ms" plot bar diagram.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    if(Names == []):
        return PlotOrder
    if(testname == "Overview_r200Mus" or testname == "Overview_r2ms"):
        Values = {}
        options = {
            "Gold"      : ["cornsilk", "Gold" ],
            "Silver"    : ["silver", "Silver" ]  
        }
        LineOption = {
            "m13_c14_t1Mus" : ["midnightblue","m13, c14, t1µs", 0],
            "m13_c14_t5ms"  : ["darkred", "m13, c14, t5ms", 2],
            "m8_c10_t1Mus"  : ["green", "m8, c10, t1µs", 4],
            "m8_c10_t5ms"   : ["indigo","m8, c10, t5ms", 6]
        }
        correction = {}
        order = 0
        for name in Names:
            if(type(testname) is list and name.split("_")[0] == "Overview"):
                continue
            entry = name.split("_")[2] + "-" + [i for i in LineOption.keys() if i in name][0]
            if(entry not in Values):
                Values[entry] = {
                    "Gold" : [],
                    "Silver" : []
                }
                Values[entry][Plots[name].type] = Plots[name].y[:]
                Type = entry.split("-")[1]
                if(Type not in correction):
                    if(Type ==   "m13_c14_t1Mus"):
                        correction[Type] =  LineOption[Type][2]
                    elif(Type == "m13_c14_t5ms"):
                        correction[Type] =  LineOption[Type][2]
                    elif(Type == "m8_c10_t1Mus"):
                        correction[Type] =  LineOption[Type][2]
                    elif(Type == "m8_c10_t5ms"):
                        correction[Type] =  LineOption[Type][2]
            else:
                if(Values[entry][Plots[name].type] == []):
                    Values[entry][Plots[name].type] = Plots[name].y[:]
                else:
                    Values[entry][Plots[name].type].extend(Plots[name].y[:])
        width = 0.4
        xticks = [i +  6/2 * width for i in  [1, 5, 10, 15]]
        xlabels = ["1", "5", "10", "50"]
        plt.figure(PlotOrder, figsize = (10,5))
        for name in Values:
                Plots[name + "Silver" + "delay"] = Plot(x = [], y = [i * int(name.split('-')[0]) for i in Values[name]["Silver"]])
                bottom = Plots[name + "Silver" + "delay"].bar(color = options["Silver"][0], label = name.split('-')[0] if name.split('-')[0] != "50" else "15", correction = correction[name.split("-")[1]], edgecolor = LineOption[name.split('-')[1]][0], width = width)
                Plots[name + "Gold" + "delay"] = Plot(x = [], y = [i * int(name.split('-')[0]) for i in Values[name]["Gold"]])
                Plots[name + "Gold" + "delay"].bar(color = options["Gold"][0], label = name.split('-')[0] if name.split('-')[0] != "50" else "15", correction = correction[name.split("-")[1]], edgecolor = LineOption[name.split('-')[1]][0], width = width, bottom = bottom)
        SetPlotOptions(options,
                title  = "Average goodput",
                xlabel = "TCP Connections",
                ylabel = "Goodput(Mbps)",
                legend = True)
        plt.xticks(xticks, xlabels)
        plt.xlim(-1,20)
        plt.ylim(0,1100)
        if(save):
            pp.savefig()
        PlotOrder += 1 
    else:
        options = {
            "Gold"      : ["goldenrod", "Gold traffic" ],
            "Silver"    : ["silver", "Silver traffic" ]  
        }
        throughput = {}
        plt.figure(PlotOrder, figsize = (10,5))
        for name in Names:
            Entry = str(Plots[name].type) + " flows between " + str(round(Plots[name].x[0])) + "-" + str(round(Plots[name].x[-1])) + "s" 
            if(Entry not in throughput):
                throughput[Entry] = []
            throughput[Entry].append(float(sum(Plots[name].y)/len(Plots[name].y)))
            Plots[name].plot(color = options[Plots[name].type.split("\n")[0]][0])
        SetPlotOptions(options,
                        title  = "Total throughput",
                        xlabel = "Time(s)",
                        ylabel = "Throughput(Mbps)",
                        legend = True)
        PlotOrder += 1
        if(save):
            pp.savefig()

        for key in throughput.keys():
            print("Average throughput for {:g} {:s}: {:g} Mbps".format(len(throughput[key]), key, round(sum(throughput[key]))))
        if(False):    
            for name in Names:
                plt.figure(PlotOrder, figsize = (10,5))
                Plots[name].plot(color = options[Plots[name].type.split("\n")[0]][0])
                Plots[name].average(color = options[Plots[name].type.split("\n")[0]][0])
                SetPlotOptions(options,
                            title = "Throughput for " +  Plots[name].type + name.split('_')[-1].split("-")[0],
                            xlabel = "Time(s)",
                            ylabel = "Throughput(Mbps)",
                            legend = False)
                PlotOrder += 1    
                if(save):
                    pp.savefig()
    return PlotOrder

def MarkingPacketValuePlot(Plots, Names, PlotOrder, save, pp):
    """
    
    Plot packet values for tests that recorded PV instead of delay in the identification field.

    Args:
        Plots (dict): dict of all plots.
        Names (list): list of the names of plots to plot.  
        PlotOrder (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    if(Names == []):
        return PlotOrder
    options = {
        "Gold"      : ["goldenrod", "Gold traffic", 0 ],
        "Silver"    : ["silver", "Silver traffic", 1 ]  
        }
    marking = {}
    totalSum = 0
    Plots["Gold"] = Plot([],[], empty = True)
    Plots["Silver"] = Plot([],[], empty = True)
    for name in Names:
        Entry = str(Plots[name].type) + " marking " + str(round(Plots[name].x[0])) + "-" + str(round(Plots[name].x[-1])) + "s" 
        if(Entry not in marking):
            marking[Entry] = []
        Plots[name].y = np.divide(Plots[name].y,2**9 / 1000000)
        marking[Entry].append(np.mean(Plots[name].y))
        Plots[Plots[name].type].y.extend(Plots[name].y)
        totalSum += len(Plots[name].y)

    plt.figure(PlotOrder, figsize = (10,5))
    for key in options:
        print("Percent of {:s} packets: {:g}% ".format(key,(len(Plots[key].y)/totalSum)*100))
        Plots[key].box(color = options[key][0], labels = options[key][1], positions = options[key][2])
    SetPlotOptions(options,
                title = "Packet values from the marker for Gold and Silver",
                xlabel = "",
                ylabel = "Packet value",
                legend = False)
    PlotOrder += 1
    if(save):
        pp.savefig()
    for key in marking.keys():
        median = np.median(np.array(marking[key]))
        print("Mean of the marking for {:g} {:s}: {:g}, median: {:g}".format(len(marking[key]), key, round(sum(marking[key])) / len(marking[key]),median))
    if(False):
        for name in Names:
            plt.figure(PlotOrder, figsize = (10,5))
            Plots[name].hist(color = options[name][0])
            SetPlotOptions(options,
                        title = "Packet values produced by the marker for {:s}: Histogram".format(name.split("_")[0]), 
                        ylabel = "Count", 
                        xlabel = "Packet Value")
            PlotOrder += 1
            if(save):
                pp.savefig()
    plt.figure(PlotOrder, figsize = (10,5))
    for name in Names:
        if(Plots[name].type == "Silver"):
            Plots[name].hist(color = options[Plots[name].type][0])
            SetPlotOptions(options,
                            title = "Packet values produced by the marker for Silver: Histogram", 
                            ylabel = "Count", 
                            xlabel = "Packet Value",
                            legend = True)
    PlotOrder += 1
    if(save):
        pp.savefig()  
    plt.figure(PlotOrder, figsize = (10,5))
    for name in Names:
        if(Plots[name].type == "Gold"):
            Plots[name].hist(color = options[Plots[name].type][0])
            SetPlotOptions(options,
                            title = "Packet values produced by the marker for Gold: Histogram", 
                            ylabel = "Count", 
                            xlabel = "Packet Value",
                            legend = True)
    PlotOrder += 1
    if(save):
        pp.savefig() 
    return PlotOrder

def StartPlot(name, save, version = ""):
    """
    Plots all plots found under the given name and version. 

    Args:
        name (string): Name of test to plot
        save (bool): Set to true to save all plots to a pdf.
        version (str, optional): Specifies what to plot: "Ev2" or "". Defaults to "".

    Returns:
        [type]: [description]
    """
    warnings.filterwarnings("ignore")
    Plots, names = GeneratePlots(name)
    pp = PdfPages(pdfPath + name + '.pdf' if type(name) is not list else pdfPath + "dynamic" + name[1] + ".pdf") if save else 0
    PlotOrder = 0
    PlotOrder = SessionDelayPlot(Plots, [names[1][i] for i in range(names[0]) if names[1][i].split("_")[-2] == "Session" and names[1][i].split("-")[1] == "Delay"], PlotOrder, save, pp, name, version)
    PlotOrder = SessionThroughputPlot(Plots, [names[1][i] for i in range(names[0]) if names[1][i].split("_")[-2] == "Session" and names[1][i].split("-")[1] == "Throughput"],  PlotOrder, save, pp, name )
    PlotOrder = MarkingPacketValuePlot(Plots, [names[1][i] for i in range(names[0]) if names[1][i].split("_")[-2] == "Marking" and names[1][i].split("-")[1] == "Packet value"],  PlotOrder, save, pp)
    if(save):
        pp.close()
    return Plots