###############################################################################
 # A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 # Karlstad University 2021.
 # Author: L. Dahlberg
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
import getpass

PvRange = 2**16                                                                                # The PV range of the marker.

# Edit corresponding with the enviornment the code is run in.
if getpass.getuser() == "dv":
    path = "/home/dv/pythonEnv/Notebook_Love/"
    directory = path + "Flows/RecordedRegisterFiles/"
    if path + "Scripts/" not in sys.path:
        sys.path.append(path + "Scripts/")

if getpass.getuser() == "love":
    path = "/home/love/tofino-master/tofino-master-thesis/Current_project/"
    directory = path + "Evaluation/Flows/RecordedRegisterFiles/"
    if path + "Evaluation/Scripts/" not in sys.path:
        sys.path.append(path + "Evaluation/Scripts/")
from Plot import Plot

def sortEntries(values):
    """
    Sorts the order of given debug information, in order to make it easier to read.

    - The "order" list has to be the same as the "PrintOrder" list in the "Debugger" class in "RMpvp_ControlPlane.py".

    - The keys of the "register" dict has to be the same as the keys in "ItemsToPrint" in the "Debugger" class in "RMpvp_ControlPlane.py".

    Args:
        values (dict): sorted information read from np.loadtext(). 

    Returns:
        (dict,list): the post-processed dict and the time values.
    """
    time = values[len(values) -1]
    order = ["delay","CTV", "Marker_pv_1", "Marker_pv_2", "PacketCount_ingress", "interm_prob","prob_before_table","modprob", "Counter Index"]
    registers = {
            "delay"               : [],
            "beta"                : [],
            "alpha"               : [],
            "interm_prob"         : [],
            "prob_before_table"   : [],
            "modprob"             : [],
            "CTV"                 : [],
            "outer_delay"         : [],
            "outer_ctv"           : [],
            "Marker_pv_1"         : [],
            "Marker_pv_2"         : [],
            "PacketCount_ingress" : [],
            "Counter Index"       : []
    }
    for i in range(len(values)-1):
        registers[order[i]] = values[i]
    return registers, time

def readFromFile(name):
    """
    Reads from the files matching the name. 
    The function does not expect to find all types of files, it will only read from those that it can find. 

    Args:
        name (string): Name of the test.

    Returns:
        (dict, list, dict, dict, dict): Dicts (and one list) containing information from the different files. 

    """
    registers = None
    Regtime = None
    ecdfDict = {}
    pv_list = []
    prob_list = []
    ECDFtimer_list = {}
    time = ""
    actualECDF = {}
    if(len(name[len("Ev1_mX"):]) == 0 or len(name[len("Ev1_mXY"):]) == 0 or name == "Ev1" and name[:len("Ev1_m")] == "Ev1_m" or name == "Ev1"):
        try:
            for file in os.listdir(directory):
                if(file[: len("timeEv1_mX")] == "time" + name or file[: len("timeEv1_mXY")] == "time" + name or file[: len("timeEv1_m")] == "timeEv1_m"):
                    combinationName = file[len("time"):-len(".csv")]
                    with open(directory + file, 'r') as f:
                        lines = f.readlines()
                        variants = []
                        for line in lines:
                            if(line.split(':')[0].split('-')[0] == "New"):
                                variants.append(combinationName +"-"+ line.split(':')[0].split('-')[1])
                                line = line.split(':')[1]
                                ECDFtimer_list[variants[-1]] = [[],[]]
                                ECDFtimer_list[variants[-1]][0].append(line.split(",")[0])
                                ECDFtimer_list[variants[-1]][1].append(line.split(",")[1].split("\n")[0])
                            else:
                                ECDFtimer_list[variants[-1]][0].append(float(line.split(",")[0]))
                                ECDFtimer_list[variants[-1]][1].append(float(line.split(",")[1].split("\n")[0])*1000)
        except OSError:
            pass
    else:
        try:
            values = np.loadtxt(directory + "Recording{:s}.csv".format(name), delimiter=",")
            registers, Regtime = sortEntries([[entry[i] for entry in values] for i in range(len(values[0]))])
        except OSError:
            pass
        try:
            with open(directory + "time{:s}.csv".format(name), 'r') as f:
                lines = f.readlines()
                variants = []
                for line in lines:
                    if(line.split(':')[0].split('-')[0] == "New"):
                        variants.append(line.split(':')[0].split('-')[1])
                        line = line.split(':')[1]
                        ECDFtimer_list[variants[-1]] = [[],[]]
                        ECDFtimer_list[variants[-1]][0].append(line.split(",")[0])
                        ECDFtimer_list[variants[-1]][1].append(line.split(",")[1].split("\n")[0])
                    else: 
                        ECDFtimer_list[variants[-1]][0].append(float(line.split(",")[0]))
                        ECDFtimer_list[variants[-1]][1].append(float(line.split(",")[1].split("\n")[0])*1000)
        except OSError:
            pass
        try:
            counter = 0
            with open(directory + "ecdf{:s}.txt".format(name), 'r') as f:
                lines = f.readlines()
                if(lines != []):
                    ecdfDict[0] = [counter,[],[]]
                for line in lines:
                    ecdfDict[0] = [counter,[],[]]
                    if(line[0:len("time")] == "time"):
                        time = line.split(":")[1].split("\n")[0]
                        counter += 1
                        ecdfDict[counter] = [time[1:], [],[]]
                    else:
                        ecdfDict[counter][1].append(float(line.split(",")[0])/1000000)
                        ecdfDict[counter][2].append(float(line.split(",")[1]))
        except OSError:
            pass
        try:
            counter = 0
            binCount = 0
            first = True
            with open(directory + "actualECDF{:s}.txt".format(name), 'r') as f:
                lines = f.readlines()
                if(lines != []):
                    actualECDF[0] = [counter,[],[]]
                for line in lines:
                    actualECDF[0] = [counter,[],[]]
                    if(first and line[0:len("time")] != "time"):
                        actualECDF["n_pvs"] = int(line.split('\n')[0])
                        first = False
                        continue
                    if(line[0:len("time")] == "time"):
                        if "n_pvs" not in actualECDF:
                            actualECDF["n_pvs"] = 2048
                        time = line.split(":")[1].split("\n")[0]
                        counter += 1
                        binCount = 0
                        actualECDF[counter] = [time[1:], [],[]]
                    else:
                        actualECDF[counter][1].append(binCount*PvRange/actualECDF["n_pvs"])
                        binCount += 1
                        actualECDF[counter][2].append(float(line))
        except OSError:
            pass
        if(registers == None and Regtime == None and  ecdfDict == {} and  pv_list == [] and  prob_list == [] and  ECDFtimer_list == [] and  actualECDF == {}):
            print("No files found under name {:s}".format(name))
            return None, None, None, None, None
    return registers, Regtime, ecdfDict, actualECDF, ECDFtimer_list

def GeneratePlots(name):
    """
    Generates instances of the Plot class from the given test name. 
    This is done by reading from the test files found under said name.  

    Args:
        name (string): Name of the test.

    Returns:
        (dict, int, int): dict containing all Plots, and the lengths of two tests.  
    """
    registers, time, ecdfDict, actualECDF, ECDFtimer = readFromFile(name)
    ecdfLength = 0
    actualECDFLength = 0
    Plots = {}
    if(registers == None and time == None and ecdfDict == None and actualECDF == None and ECDFtimer == None):
        return "error", "", ""
    if(registers != None and time != None):
        for register in registers.keys():
            Plots[register] = Plot(time, registers[register])
    if(ECDFtimer != {}):
        for key in ECDFtimer.keys():
            Plots["ECDFtimer " + key] = Plot(ECDFtimer[key][0][1:], ECDFtimer[key][1][1:], option = [ECDFtimer[key][0][0], ECDFtimer[key][1][0]])
    if(actualECDF != {}):
        actualECDFLength = int(actualECDF[0][0]) + 1   
        avgPlot = np.zeros((2,actualECDF["n_pvs"]))
        for i in range(1,actualECDFLength):
            Plots["actualECDF" + str(i)] = Plot(actualECDF[i][1], actualECDF[i][2], actualECDF[i][0])
        for i in range(actualECDF["n_pvs"]):
            avg = 0
            for j in range(1,actualECDFLength):
                avg += actualECDF[j][2][i]
            avg /= actualECDFLength
            avgPlot[0][i] = avg
            avgPlot[1][i] = actualECDF[j][1][i]
        Plots["AverageECDF"] = Plot(avgPlot[1], avgPlot[0])                                     # Average of each point in the ECDF.
    if(ecdfDict != {}):
        ecdfLength = int(ecdfDict[0][0]) + 1
        for i in range(1, ecdfLength):
            Plots["ecdf" + str(i)] = Plot(ecdfDict[i][1], ecdfDict[i][2], ecdfDict[i][0])
    return Plots, ecdfLength, actualECDFLength