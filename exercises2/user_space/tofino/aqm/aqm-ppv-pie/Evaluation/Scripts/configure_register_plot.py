###############################################################################
 # A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 # Karlstad University 2021.
 # Author: L. Dahlberg
###############################################################################

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
import sys
import warnings
import collections
import getpass
import numpy as np

# Edit corresponding with the enviornment the code is run in.
if getpass.getuser() == "dv":
    path = "/home/dv/pythonEnv/Notebook_Love/"
    pdfPath = path + "Results/Register_result/"
    if path + "Scripts/" not in sys.path:
        sys.path.append(path + "Scripts/")

if getpass.getuser() == "love":
    path = "/home/love/tofino-master/tofino-master-thesis/Current_project/"
    pdfPath = path + "Evaluation/Results/Register_result/"
    if path + "Evaluation/Scripts/" not in sys.path:
        sys.path.append(path + "Evaluation/Scripts/")

from configure_register_data import GeneratePlots
from Plot import SetPlotOptions

def BigMarkingPlot(Plots, order, save, pp):
    """
    Plots all marking plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.
        
    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
        "CTV"        : ["red", "CTV"], 
        "Marker_pv_1": ["goldenrod", "Gold traffic" ],
        "Marker_pv_2": ["silver", "Silver traffic" ]
    }

    if("CTV" in Plots or "Marker_pv_1" in Plots or "Marker_pv_2" in Plots):
        plt.figure(order, figsize = (10,5))
        if("CTV" in Plots):
            Plots["CTV"].box(color = options["CTV"][0], labels = "CTV", positions = 0)
        if("Marker_pv_1" in Plots):
            Plots["Marker_pv_1"].box(color = options["Marker_pv_1"][0], labels = "Gold traffic", positions = 1)
        if("Marker_pv_2" in Plots):
            Plots["Marker_pv_2"].box(color = options["Marker_pv_2"][0], labels = "Silver traffic", positions = 2)
        SetPlotOptions(options,
                        title = "Gold and Silver packet marking with AQM CTV value", 
                        xlabel = "", 
                        ylabel = "Packet Value",
                        legend = False)
    if(save):
        pp.savefig()
    return order + 1

def MarkingPlot(Plots, order, save, pp, version):
    """
    Plots the marking plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
    "CTV"        : ["red", "CTV"], 
    "Marker_pv_1": ["blue", "Marking for client 10.0.0.1"],
    "Marker_pv_2": ["green", "Marking for client 10.0.0.2"]
    }
    if("CTV" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["CTV"].box(color = options["CTV"][0], labels = "", positions = 0)
        SetPlotOptions(options,
            title = "CTV", 
            xlabel = "", 
            ylabel = "Packet Value")
        if(save):
            pp.savefig()
        order += 1
        plt.figure(order, figsize = (10,5))
        Plots["CTV"].scatter(color = options["CTV"][0])
        plt.ylim(23000, 37000)
        SetPlotOptions(options,
            title = "CTV", 
            xlabel = "time(s)", 
            ylabel = "Packet Value")
        if(save):
            pp.savefig()
        zero = 0
        nonzero = 0
        for i in Plots["CTV"].y:
            if(i == 0):
                zero += 1
            else:
                nonzero += 1
        median = np.median(np.array(Plots["CTV"].y))
        print("CTV: Proportion of 0s: {:g}%, other: {:g}%. Average = {:g}, Median {:g}".format(zero/(zero + nonzero)*100,nonzero/(zero + nonzero)*100, sum(Plots["CTV"].y)/len(Plots["CTV"].y),median))

    if("Marker_pv_1" in Plots and version != "Evaluation_2"):
        plt.figure(order, figsize = (10,5))
        Plots["Marker_pv_1"].scatter(color = options["Marker_pv_1"][0])
        SetPlotOptions(options,
               title = "Marking for gold 10.0.0.1", 
               xlabel = "Time(s)", 
               ylabel = "Packet Value")
        if(save):
            pp.savefig()
        order += 1 
    if("Marker_pv_2" in Plots and version != "Evaluation_2"):
        plt.figure(order, figsize = (10,5))
        Plots["Marker_pv_2"].scatter(color = options["Marker_pv_2"][0])
        SetPlotOptions(options,
               title = "Marking for silver 10.0.0.2", 
               xlabel = "Time(s)", 
               ylabel = "Packet Value")
        if(save):
            pp.savefig()
        order += 1
    return order + 1

def DelayPlot(Plots, order, save, pp):
    """
    Plots the delay plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
        "delay"        : ["red", ""], 
    }
    if("delay" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["delay"].plot(color = options["delay"][0])
        print("Average Queue delay {:g} ms".format(sum(Plots["delay"].y)/len(Plots["delay"].y)))
        SetPlotOptions(options,
                       title = "Recorded queue delay", 
                       xlabel = "Time(s)", 
                       ylabel = "Queue delay(ms)")
    if(save):
        pp.savefig()
    return order + 1

def MarkingHist(Plots, order, save, pp, version):
    """
    Plots the marking histogram plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
        "Marker_pv_1": ["red", "10.0.0.1"],
        "Marker_pv_2": ["blue", "10.0.0.2"]
    }
    if("Marker_pv_1" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["Marker_pv_1"].hist(color = options["Marker_pv_1"][0])
        SetPlotOptions(options,
                       title = "Packet values produced by the marker for 10.0.0.1: Histogram", 
                       ylabel = "Count", 
                       xlabel = "Packet Value")
        if(save):
            pp.savefig()
        order += 1
    if("Marker_pv_2" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["Marker_pv_2"].hist(color = options["Marker_pv_2"][0])
        SetPlotOptions(options,
                       title = "Packet values produced by the marker for 10.0.0.2: Histogram", 
                       ylabel = "Count", 
                       xlabel = "Packet Value")
        if(save):
            pp.savefig()
        order += 1
    if(version == ""):
        if("Marker_pv_2" in Plots and "Marker_pv_1" in Plots):
            plt.figure(order, figsize = (10,5))
            Plots["Marker_pv_2"].hist(color = options["Marker_pv_2"][0])
            Plots["Marker_pv_1"].hist(color = options["Marker_pv_1"][0])
            SetPlotOptions(options,
                    title = "Packet values produced by the marker for both clients: Histogram", 
                    ylabel = "Count", 
                    xlabel = "Packet Value",
                    legend = True)
            if(save):
                pp.savefig()

    return order + 1

def IngressCounterPlot(Plots, order, save, pp):
    """
    Plots the ingress counter plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """

    options = {
        "PacketCount_ingress"         : ["red", ""], 
    }
    if("PacketCount_ingress" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["PacketCount_ingress"].plot(color = options["PacketCount_ingress"][0])
        SetPlotOptions(options,
                       title = "Sum of packets arrival ingress packets", 
                       xlabel = "Time(s)", 
                       ylabel = "Number of packets")
    if(save):
        pp.savefig()
    return order + 1     

def ecdfPlot(Plots, order, save, pp, ecdfLength):
    """
    Plots the ECDF plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {"ecdf" + str(i) : ["green", ""] for i in range(ecdfLength)}
    for i in range(ecdfLength):
        if(i == 0):
            continue
        plt.figure(order, figsize = (10,5))
        Plots["ecdf" + str(i)].scatter(color = options["ecdf" + str(i)][0])
        SetPlotOptions(options,
                       title = "Probability to Packet value at {:s} seconds".format(str(Plots["ecdf" + str(i)].option)), 
                       xlabel = "Probability", 
                       ylabel = "Packet value")
        order += 1
        if(save):
            pp.savefig()
    return order 


def IntermProbPlot(Plots, order, save, pp):
    """
    Plots the intermediate probability plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
    "interm_prob"        : ["red", ""], 
    }
    if("interm_prob" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["interm_prob"].scatter(color = options["interm_prob"][0])
        SetPlotOptions(options,
                       title = "Calculated probability after alpha + beta operation", 
                       xlabel = "Time(s)", 
                       ylabel = "Probability")
    if(save):
        pp.savefig()
    return order + 1

def ProbBeforeTablePlot(Plots, order, save, pp):
    """
    Plots the prob before table plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
    "prob_before_table"        : ["blue", ""], 
    }
    if("prob_before_table" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["prob_before_table"].scatter(color = options["prob_before_table"][0])
        SetPlotOptions(options,
                       title = "Calculated probability right before div2square operation", 
                       xlabel = "Time(s)", 
                       ylabel = "Probability")
    if(save):
        pp.savefig()
    return order + 1


def ModProbPlot(Plots, order, save, pp):
    """
    Plots the mod prob plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
    "modprob"        : ["blue", ""], 
    }
    if("modprob" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["modprob"].scatter(color = options["modprob"][0])
        SetPlotOptions(options,
                       title = "Calculated probability after div2square operation", 
                       xlabel = "Time(s)", 
                       ylabel = "Probability")
    if(save):
        pp.savefig()
    return order + 1


def actualECDFplot(Plots, order, save, pp, actualECDFLength):
    """
    Plots the actual ECDF (Probability to CTV) plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {"actualECDF" + str(i) : ["blue", ""] for i in range(actualECDFLength)}
    for i in range(actualECDFLength):
        if(i == 0):
            continue
        plt.figure(order, figsize = (10,5))
        Plots["actualECDF" + str(i)].plot(color = options["actualECDF" + str(i)][0])
        SetPlotOptions(options,
                       title = "ECDF at {:s} seconds".format(str(Plots["actualECDF" + str(i)].option)), 
                       ylabel = "Probability", 
                       xlabel = "Packet Value")
        order += 1
        if(save):
            pp.savefig()
    return order 

def averageECDFplot(Plots, order, save, pp):
    """
    Plots the average ECDF plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
    "AverageECDF"        : ["red", ""], 
    }
    if("AverageECDF" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["AverageECDF"].plot(color = options["AverageECDF"][0])
        SetPlotOptions(options,
                       title = "Average ECDF", 
                       ylabel = "Probabilty", 
                       xlabel = "Packet Value")
    if(save):
        pp.savefig()
    return order + 1

def ECDFtimerPlot(Plots, order, save, pp):
    """
    Plots the ECDF timer plots. 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
    "ECDFtimer Variant 1"        : ["red", "Variant 1"],
    "ECDFtimer Variant 2"        : ["blue", "Variant 2"]
    }
    if("ECDFtimer Variant 1" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["ECDFtimer Variant 1"].plot(color = options["ECDFtimer Variant 1"][0])
        print("Average ECDF time of Variant 1: {:f} ms".format(sum(Plots["ECDFtimer Variant 1"].y)/len(Plots["ECDFtimer Variant 1"].y)))
        SetPlotOptions(options,
                       title = "Variant 1: ECDF time for table size {:s} and counter size {:s}".format(Plots["ECDFtimer Variant 1"].option[0], Plots["ECDFtimer Variant 1"].option[1]), 
                       ylabel = "ECDF time(ms)", 
                       xlabel = "time(s)")
        order += 1
    if(save):
        pp.savefig()
    if("ECDFtimer Variant 2" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["ECDFtimer Variant 2"].plot(color = options["ECDFtimer Variant 2"][0])
        print("Average ECDF time of Variant 2: {:f} ms".format(sum(Plots["ECDFtimer Variant 2"].y)/len(Plots["ECDFtimer Variant 2"].y)))
        SetPlotOptions(options,
                       title = "Variant 2: ECDF time for table size {:s} and counter size {:s}".format(Plots["ECDFtimer Variant 2"].option[0], Plots["ECDFtimer Variant 2"].option[1]), 
                       ylabel = "ECDF time(ms)", 
                       xlabel = "time(s)")
        order += 1
    if(save):
        pp.savefig()
    if("ECDFtimer Variant 1" in Plots and "ECDFtimer Variant 2" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["ECDFtimer Variant 2"].plot(color = options["ECDFtimer Variant 2"][0])
        Plots["ECDFtimer Variant 1"].plot(color = options["ECDFtimer Variant 1"][0])
        SetPlotOptions(options,
                       title = "ECDF time for table size {:s} and counter size {:s} for both variant".format(Plots["ECDFtimer Variant 2"].option[0], Plots["ECDFtimer Variant 2"].option[1]), 
                       ylabel = "ECDF time(ms)", 
                       xlabel = "time(s)",
                       legend = True)
        order += 1
    if(save):
        pp.savefig()
    return order + 1

def CounterIndex(Plots, order, save, pp):
    """
    Plots the counter index.

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    options = {
    "Counter Index"       : ["red", ""],
    }
    if("Counter Index" in Plots):
        plt.figure(order, figsize = (10,5))
        Plots["Counter Index"].plot(color = options["Counter Index"][0])
        Plots["Counter Index"].average(color = options["Counter Index"][0])
        SetPlotOptions(options,
                       title = "Counter index", 
                       ylabel = "Index", 
                       xlabel = "time(s)")
    if(save):
        pp.savefig()
    return order + 1

def Ev1Plot(Plots, name, order, save, pp):
    """
    Plots the plots associated with version "Evaluation_1". 

    Args:
        Plots (dict): dict of all Plots.
        order (int): Plot order.
        save (bool): If true save plot to pdf.
        pp (pdf object): Pdf save object.

    Returns:
        int: The correctly incremented PlotOrder.
    """
    if(len(name[len("Ev1_mX"):]) != 0 and len(name[len("Ev1_mXY"):]) != 0 or name[:len("Ev1")] != "Ev1"):
        return order + 1
    Plots = dict(sorted(Plots.items(), key = lambda x: int(x[0].split(" ")[1].split("-")[0].split("_")[2].split("c")[1])))
    position = {
        "1024\nv1"  : 0,
        "1024\nv2"  : 1,
        "2048\nv1"  : 2,
        "2048\nv2"  : 3,
        "4096\nv1"  : 4,
        "4096\nv2"  : 5,
        "8192\nv1"  : 6,
        "8192\nv2"  : 7,
        "16384\nv1" : 8,
        "16384\nv2" : 9,
        "32768\nv1" : 10,
        "32768\nv2" : 11
    }
    colors = {
        "m8" : "tab:blue",
        "m9" :  "tab:orange",
        "m10" :  "tab:green",
        "m11" :  "tab:red",
        "m12" : "tab:purple",
        "m13" : "tab:cyan"
    }
 
    options = {}
    plt.figure(order, figsize = (10,5))
    for key in Plots.keys():
        variant = key.split("-")[1].split(" ")[1]
        Type = key.split("-")[0].split(" ")[1].split("_")[-1]
        m = key.split(" ")[1].split("_")[1]
        c = key.split(" ")[1].split("-")[0].split("_")[2].split("c")[1]
        if(variant == "1" or Type == "2"):
            if(m not in options):
                options[m] = [colors[m], m]
            Plots[key].box(color = options[m][0], positions = position[c + "\nv" + variant if variant == "1" else c + "\nv" + Type])
    plt.xticks([i for i in range(12)],position.keys())
    SetPlotOptions(dict(sorted(options.items(), key = lambda x: 100 - int(x[1][1].split("m")[1]))),
                   title = "ECDF time for all table and counter sizes for both variants", 
                   ylabel = "ECDF time(ms)", 
                   xlabel = "Counter size and version",
                   legend = True)
    if(save):
        pp.savefig()
    order += 1

    position = {
        "1024"  : 0,
        "2048"  : 1,
        "4096"  : 2,
        "8192"  : 3,
        "16384" : 4,
        "32768" : 5
    }
    options = {}
    plt.figure(order, figsize = (10,5))
    for key in Plots.keys():
        variant = key.split("-")[1].split(" ")[1]
        m = key.split(" ")[1].split("_")[1]
        c = key.split(" ")[1].split("-")[0].split("_")[2].split("c")[1]
        if(variant== "1"):
            if(m not in options):
                options[m] = [colors[m], m]
            Plots[key].box(color = options[m][0], positions = position[c])
    plt.xticks([0,1,2,3,4,5],[1024,2048, 4096, 8192, 16384, 32768])
    SetPlotOptions(dict(sorted(options.items(), key = lambda x: 100 - int(x[1][1].split("m")[1]))),
                   title = "ECDF time for all table and counter sizes for variant 1", 
                   ylabel = "ECDF time(ms)", 
                   xlabel = "Counter size",
                   legend = True)
    if(save):
        pp.savefig()
    order += 1

    plt.figure(order, figsize = (10,5))
    options = {}
    for key in Plots.keys():
        variant = key.split("-")[1].split(" ")[1]
        Type = key.split("-")[0].split(" ")[1].split("_")[-1]
        m = key.split(" ")[1].split("_")[1]
        c = key.split(" ")[1].split("-")[0].split("_")[2].split("c")[1]
        if(variant == "2" and Type == "2"):
            if(m not in options):
                options[m] = [colors[m], m]
            Plots[key].box(color = options[m][0], positions = position[c])
    plt.xticks([0,1,2,3,4,5],[1024,2048, 4096, 8192, 16384, 32768])
    SetPlotOptions(dict(sorted(options.items(), key = lambda x: 100 - int(x[1][1].split("m")[1]))),
                title = "ECDF time for all table and counter sizes for variant 2", 
                ylabel = "ECDF time(ms)", 
                xlabel = "Counter size",
                legend = True)

    if(False):
        color_index = 0
        plt.figure(order, figsize = (10,5))
        for key in Plots.keys():
            if(key.split("-")[1].split(" ")[1] == "2"):
                options[key] = [colors[color_index], int(key.split(" ")[1].split("-")[0].split("_")[2].split("c")[1])]
                color_index += 1
                Plots[key].box(color = options[key][0], labels = options[key][1], positions = color_index)
        SetPlotOptions(options,
                    title = "ECDF time for table size {:s} and all counter sizes for variant 2".format(list(Plots.keys())[0].split(" ")[1].split("-")[0].split("_")[1].split("m")[1]), 
                    ylabel = "ECDF time(ms)", 
                    xlabel = "time(s)",
                    legend = False)
        if(save):
            pp.savefig()
    return order + 1

def StartPlot(name, save, PlotOrder = 0, version = ""):
    """
    Plots all plots found under the given name and version. 

    Args:
        name (string): Name of test to plot
        save (bool): Set to true to save all plots to a pdf.
        PlotOrder (int, optional): Represents different plots. Defaults to 0.
        version (str, optional): Specifies what to plot: "Evaluation_1", "Evaluation_2" or "". Defaults to "".

    Returns:
        int: The correctly incremented PlotOrder.
    """
    warnings.filterwarnings("ignore")
    Plots, ecdfLength, actualECDFLength = GeneratePlots(name)
    if(Plots == "error" or Plots == {}):
        return
    if(save):
        pp = PdfPages(pdfPath + name + '.pdf')
    else:
        pp = 0
    if(version == "Evaluation_1"):
        PlotOrder = Ev1Plot(Plots, name, PlotOrder, save, pp)
        PlotOrder = ECDFtimerPlot(Plots, PlotOrder, save, pp)
    if(version == "Evaluation_2" or version == ""):
        PlotOrder = MarkingPlot(Plots,PlotOrder, save, pp, version)
        PlotOrder = averageECDFplot(Plots, PlotOrder,save, pp)
    if(version == ""):
        PlotOrder = DelayPlot(Plots, PlotOrder, save, pp)
        PlotOrder = MarkingHist(Plots, PlotOrder, save, pp, version)
        PlotOrder = BigMarkingPlot(Plots, PlotOrder, save, pp)
        PlotOrder = CounterIndex(Plots, PlotOrder, save, pp)
        PlotOrder = IntermProbPlot(Plots, PlotOrder, save, pp)
        PlotOrder = ProbBeforeTablePlot(Plots, PlotOrder, save, pp)
        PlotOrder = ModProbPlot(Plots, PlotOrder, save, pp)
        PlotOrder = IngressCounterPlot(Plots,PlotOrder, save, pp)
        if(actualECDFLength != 0):
            PlotOrder = actualECDFplot(Plots, PlotOrder,save, pp, actualECDFLength)
        if(ecdfLength != 0):
            PlotOrder = ecdfPlot(Plots, PlotOrder,save, pp, ecdfLength)
    if(save):
        pp.close()
    return PlotOrder
    