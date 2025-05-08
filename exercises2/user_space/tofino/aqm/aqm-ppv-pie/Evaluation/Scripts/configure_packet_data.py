###############################################################################
 # A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 # Karlstad University 2021.
 # Author: L. Dahlberg
###############################################################################

import numpy as np
from scapy.all import *
import os
import time
import multiprocessing as mp
import getpass

# Edit corresponding with the enviornment the code is run in.
if getpass.getuser() == "dv":
    path = "/home/dv/pythonEnv/Notebook_Love/"
    directory = path + "/Flows/SplitPcap/newSplit/"
    if path + "Scripts/" not in sys.path:
        sys.path.append(path + "Scripts/")

if getpass.getuser() == "love":
    path = "/home/love/tofino-master/tofino-master-thesis/Current_project/"
    directory = path + "Evaluation/Flows/SplitPcap/newSplit/"
    if path + "Evaluation/Scripts/" not in sys.path:
        sys.path.append(path + "Evaluation/Scripts/")

from Plot import Plot


# Expand to add additional tests that are run in parallel. Key is name of test and value[1] is path to pcaps. 
Clients = {
    "10.0.0.1_E_Session"    : [ False, "10.0.0.1/EricssonMarker_Connection/"],
    "10.0.0.1_Session"      : [ False, "10.0.0.1/Connection/"],
    "10.0.0.2_Session"      : [ False, "10.0.0.2/Connection/"],
    "10.0.0.1_Marking"      : [ False, "10.0.0.1/Marking/"   ],
    "10.0.0.2_Marking"      : [ False, "10.0.0.2/Marking/"   ]
}


def ResetClients():
    """
    Used to reset the Client dict.

    Returns:
        dict: Original dict structure.
    """
    return {
        "10.0.0.1_E_Session"    : [ False, "10.0.0.1/EricssonMarker_Connection/"],
        "10.0.0.1_Session"      : [ False, "10.0.0.1/Connection/"],
        "10.0.0.2_Session"      : [ False, "10.0.0.2/Connection/"],
        "10.0.0.1_Marking"      : [ False, "10.0.0.1/Marking/"   ],
        "10.0.0.2_Marking"      : [ False, "10.0.0.2/Marking/"   ]
    }


class PcapClient:
    """
    Class used for parsing pcap files and then storing the parsed information in a Control File.

    Args:
        name (string): name of the option to parse.
        pcapFiles (dict): paths to the pcap files.
    """
    def __init__(self, name, pcapFiles):
        self.name = name
        self.pcapFiles = pcapFiles
        self.totalThroughput =  mp.Manager().dict()
        self.timeReference =  mp.Manager().dict()
        self.type =  mp.Manager().dict()
        self.time =  {}
        self.delay =  {}
        self.throughput = {}
        for key in pcapFiles.keys():
            self.time[key] = mp.Manager().list()
            self.delay[key] = mp.Manager().list()
            self.throughput[key] = mp.Manager().list()
            self.totalThroughput[key] = 0
            self.timeReference[key] = 0
    
    def writeToControlFile(self, TestName):
        ControlFile = directory + "ControlFiles/{:s}ControlFile{:s}".format(TestName,self.name)
        start_time = time.time()
        print("\nIn {:s}, writing all variants to controlFile".format(self.name))
        TimeReference = 0
        first = True
        for key in self.pcapFiles.keys():
            if(first):
                TimeReference = self.timeReference[key]
                first = False
            elif(self.timeReference[key] < TimeReference):
                TimeReference = self.timeReference[key]
        with open(ControlFile, 'w+') as f:
            for key in self.pcapFiles.keys():
                if(key in self.totalThroughput and key in self.type and key in self.timeReference and key in self.time and key in self.delay and key in self.throughput):
                    f.write(str(key) + "\nTimeReference: " + str(TimeReference) + "\nTotal throughput: " + str(self.totalThroughput[key]) + "\nType:" + self.type[key] + "\ntime, delay, throughput") # "\nThroughput: " + str(self.throughput[key])
                    for i in range(len(self.delay[key])):
                        f.write("\n" + str(self.time[key][i]) + "," + str(self.delay[key][i]) + "," + str(self.throughput[key][i]))
                    f.write("\n")
        print("\nIn {:s}, done writing to controlFile after {:s} s".format(self.name,str(time.time() - start_time)))
        return ControlFile
    
    def readPacket(self, name):
        print("\nIn {:s}, reading variant {:s}".format(self.name, name.split("-")[1]))
        start_time = time.time()
        packets = rdpcap(self.pcapFiles[name])
        first_packet = True
        Old_packet_sum = 0
        Packet_sum = 0
        Throughput_interval = 0.1 # 100ms
        Convertion = 8*10*10**-6
        Time_start = 0
        for packet in packets:
            if packet.haslayer(IP):
                if(first_packet): 
                    self.timeReference[name] = packet.time
                    Time_start = packet.time
                    first_packet = False
                    if(packet.tos == 8):
                        self.type[name] = "Gold"
                    else:
                        self.type[name] = "Silver"
                if(packet.time - Time_start >= Throughput_interval):
                    Old_packet_sum = Packet_sum * Convertion
                    Time_start = packet.time
                    Packet_sum = len(packet)
                else:
                    Packet_sum += len(packet)
                self.throughput[name].append(Old_packet_sum)
                self.delay[name].append((packet[IP].id << 9)/1000000)
                self.time[name].append(packet.time)
                self.totalThroughput[name] += len(packet)
        print("\nIn {:s}, done reading variant {:s} after {:s} s".format(self.name, name.split("-")[1], str(time.time() - start_time)))
                
    def readAllPackets(self):
        """
        Reads all pcaps using multiple processing.
        """
        Processes = []
        numberOfProcesses = 0
        for key in self.pcapFiles.keys():
            try:
                Processes.append(mp.Process(target = self.readPacket, args = (key, )))
                numberOfProcesses += 1
                Processes[-1].start()
                if(numberOfProcesses >= 8):
                    for process in Processes:
                        process.join()
                    Processes = []
                    numberOfProcesses = 0
            except:
                print("Process died: ")
        for process in Processes:
            process.join()



class DataPointClient:
    """
    Class used to read and store information from a Control file.

    
    Constructor args:
        name (string): name of the option to read.
        ControlFile (string): path to Control File.

    """
    def __init__(self, name, ControlFile):
        self.name = name
        self.ControlFile = ControlFile
        self.time =  mp.Manager().dict()
        self.delay =  mp.Manager().dict()
        self.throughput =  mp.Manager().dict()
        self.totalThroughput = mp.Manager().dict()
        self.timeReference =  mp.Manager().dict()
        self.type = mp.Manager().dict()
    
    def readFromControlFile(self):
        number = 0
        _time = self.time
        _delay = self.delay
        _throughput = self.throughput
        tmp_time = {}
        tmp_delay = {}
        tmp_throughput = {}
        with open(self.ControlFile, 'r') as f: 
            lines = f.readlines()
            for line in lines:
                if(line == ""):
                    continue
                elif(line.split("-")[0] == self.name):
                    number = line.split("-")[1].split("\n")[0]
                    tmp_time[number] = []
                    tmp_delay[number] = []
                    tmp_throughput[number] = []
                    continue
                elif(line.split(":")[0] == "Total throughput"):
                    self.totalThroughput[number] = float(line.split(":")[1])
                    continue
                elif(line.split(":")[0] == "TimeReference"):
                    self.timeReference[number] = float(line.split(":")[1])
                    continue
                elif(line.split(":")[0] == "Type"):
                    self.type[number] = line.split(":")[1].split("\n")[0]
                    continue
                elif(line.split(",")[0] == "time"):
                    continue
                else:
                    tmp_time[number].append(float(line.split(",")[0]) - self.timeReference[number])
                    tmp_delay[number].append(float(line.split(",")[1]))
                    tmp_throughput[number].append(float(line.split(",")[2].split("\n")[0]))
            for key in tmp_time.keys():
                _time[key] = tmp_time[key]
            self.time = _time
            for key in tmp_delay.keys():
                _delay[key] = tmp_delay[key]
            self.delay = _delay
            for key in tmp_throughput.keys():
                _throughput[key] = tmp_throughput[key]
            self.throughput = _throughput

    def getVariants(self):
        return list(self.delay.keys())
    
    def getInformationFromVariant(self, key):
        return self.time[key], self.delay[key], self.throughput[key], self.totalThroughput[key]
    
    def getAllInformation(self):
        return self.time, self.delay, self.throughput, self.totalThroughput
            
def SaveToClient(clientName, Data, testname = ""):
    """
    Parse the Control file and save its content in to the "Data" dict.

    Args:
        clientName (str): name of the option to parse.
        Data (dict): structure to save the parsed information in. 
        testname (str, optional): Specifies test name, "Overview_r200Mus" and "Overview_r2ms" are processed differently. Defaults to "".
    """
    Processes = []
    if(testname == "Overview_r200Mus" or testname ==  "Overview_r2ms" or type(testname) is list):
        for i in range(Clients[clientName][1] if Clients[clientName][1] != 1 else 2):
            client = DataPointClient(clientName, Clients[clientName][i + 2])
            Processes.append(mp.Process(target = client.readFromControlFile))
            Processes[-1].start()
            Data[Clients[clientName][i+2].split("/")[-1]] = client
    else:
        client = DataPointClient(clientName, Clients[clientName][1])
        Processes.append(mp.Process(target = client.readFromControlFile))
        Processes[-1].start()
        Data[clientName] = client
    for process in Processes:
        process.join()


def CheckControlFiles(name):
    """
    Checks if there exist a Control file for the given name. 

    Args:
        name (str): name of test
    """
    if(name == "Overview_r200Mus" or name ==  "Overview_r2ms" or type(name) is list):
        count = 0
        if(not type(name) is list):
            name = [name]
        for item in name:
            count = 0
            for file in os.listdir(directory + "ControlFiles"):
                if(file[:len(item)] == item):
                    for key in Clients.keys():
                        if(key in file):
                            count += 1
                            Clients[key][0] = True
                            Clients[key][1] = count
                            Clients[key].append(directory + "ControlFiles/" + file)
    else:
        for file in os.listdir(directory + "ControlFiles"):
            if(file[:len(name + "ControlFile")] == name + "ControlFile"):
                for key in Clients.keys():
                    if(file[len(name + "ControlFile"):] == key):
                        Clients[key][0] = True
                        Clients[key][1] = directory + "ControlFiles/" + file


def ReadPcaps(clientName, FileName, Data):
    """
    Parse pcap file, save to Control file and then store the information the "Data" dict. 

    Args:
        clientName (str): name of the option to parse.
        FileName (str): name of the test.
        Data (dict): structure to save the parsed information in. 
    """
    Files = []
    if(type(FileName) is list):
        return
    for file in os.listdir(directory + Clients[clientName][1]):
        if(file[:len(FileName.split("-")[0])] == FileName):
            Files.append(directory + Clients[clientName][1] + file)
    if(Files != []):
        client = PcapClient(clientName, {clientName + "-" + str(i+1): Files[i] for i in range(len(Files))})
        client.readAllPackets()
        Clients[clientName][1] = client.writeToControlFile(FileName)
        Clients[clientName][0] = True 
        SaveToClient(clientName, Data, testname = FileName )
        

def GetData(name):
    """
    Get the Data from given test name from pcap or Control file.

    Args:
        name (str): name of the test.

    Returns:
        dict: the content of the Control file.
    """
    Data = {}
    CheckControlFiles(name)       
    for key in Clients.keys():
        if(Clients[key][0] == True):
            SaveToClient(key, Data, name)
        elif(Clients[key][0] == False):
            ReadPcaps(key, name, Data)
    if(Data == {}):
        print("No file named", name)
    return Data
    

def GeneratePlots(name):
    """
    Generates instances of the Plot class from the given test name. 
    This is done either directly from pcap files or from a text file called a "Control File", which stores previously read pcap information.

    Args:
        name (str): name of the test

    Returns:
        (dict, list): dict of all plots and list for configuration.
    """
    global Clients
    Plots = {}
    Data = GetData(name)
    for Datakeys in Data.keys():
        if(Datakeys.split('_')[-1] == "Session"):
            for variant in Data[Datakeys].getVariants():
                Plots[Datakeys + "_" + variant + "-" + "Delay"] = Plot(Data[Datakeys].time[variant], Data[Datakeys].delay[variant], Data[Datakeys].totalThroughput[variant], Data[Datakeys].type[variant] )
                Plots[Datakeys + "_" + variant + "-" + "Throughput"] = Plot(Data[Datakeys].time[variant], Data[Datakeys].throughput[variant], Data[Datakeys].totalThroughput[variant], Data[Datakeys].type[variant])
        if(Datakeys.split('_')[1] == "Marking"):
            for variant in Data[Datakeys].getVariants():
                Plots[Datakeys + "_" + variant + "-" + "Packet value"] = Plot(Data[Datakeys].time[variant], Data[Datakeys].delay[variant], Data[Datakeys].totalThroughput[variant], Data[Datakeys].type[variant])
                Plots[Datakeys + "_" + variant + "-" + "Throughput"] = Plot(Data[Datakeys].time[variant], Data[Datakeys].throughput[variant], Data[Datakeys].totalThroughput[variant], Data[Datakeys].type[variant])   
    Data.clear()
    Clients = ResetClients()
    return Plots, [len(Plots),list(Plots.keys())] 