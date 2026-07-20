##################################################################################
 # A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 # Karlstad University 2021.
 # Author: L. Dahlberg
##################################################################################

from ipaddress import ip_address
import sys
import os
import numpy as np
import bisect
import time as time
from threading import Thread

DirectoryPath = "/home/tofino/projects/Love_project/Master/Final"                               # Path to "Table_generator" file.
sys.path.append(DirectoryPath)
from Table_generator import Generate_alpha_beta, Generate_Div2_Square, GetEcdfStartValues, Generate_counter_table

# Start Variables for evaluation

n_pvs = 2**10                                                                                   # "c" in the evaluation.
squareTableAccuracy = 8                                                                         # "m" in the evaluation.
PvRange = 2**16                                                                                 # The PV range of the marker.
alpha = 0.3125                                                                                  
beta = 3.125

# End Variables for evaluation

p4 = bfrt.RMpvp_main
Ports = [44, 45, 46]                                                                            # Active Ports
Pv_ports = [46]                                                                                 # Ports that are filtered.
Ip_addr = [ip_address("10.0.0.1"), ip_address("10.0.0.2"), ip_address("10.0.0.3")]              # Destination IPs for forwarding.
Queues = [0]                                                                                    # Number of queues per port.                                
counter_gran = PvRange / n_pvs                                                                  # How many PVs per counter entry.


###############
### DEBUG

def Get_ctv_table(): 
    """
    Print the CTV table in the Egress of the Data Plane.

    Returns:
        list: Dump from GetCTV table. 
    """
    return p4.SwitchEgress.EgressProcessingInstance.GetCTV.dump(from_hw=1,table=1) 

class P4register():
    """
    Class that holds the value to be printed for a register in the Data Plane. 
    The register value itself is not stored in this class, it needs to be provided for each print call.

    Constructor args:
        name (string): Name of register.
        Toprint (string): String to print with the register value. 
        function (lambda): Lambda function describing how the list of the register should be handled.
        parameters (list): List of constants that should be used.
    """
    def __init__(self, name, Toprint, function, parameters):
        self.name = name 
        self.Toprint = Toprint
        self.function = function
        self.parameters = parameters

    def PrintRegisterValue(self, registerValue):
        """
        Prints the stored string with the given register value.

        Args:
            registerValue (list): List containing the object to be printed (returned list from the Data Plane). 
        """
        self.parameters.append(registerValue)
        print(self.Toprint.format(self.function(self.parameters)))
        self.parameters.pop()
    
    def ReturnRegisterValue(self, registerValue):
        """
        Return the stored string with the given register value.

        Args:
            registerValue (list): List containing the object to be printed (returned list from the Data Plane). 

        Returns:
            float: Value from the register after being processed by the stored Lambda function.
        """
        self.parameters.append(registerValue)
        value = self.function(self.parameters)
        self.parameters.pop()
        return float(value)
 
class Debugger():
    """
    Class that contains all debugging information. Contains hard-coded paths and strings to structures in the Data Plane.
    This class contains Methods used to print register values at the point of call.

    """
    global P4register
    def __init__(self):
        """ 
        Initilizes the object with the hard-coded strings in the "ItemsToPrint" dict.
        
        -The keys in the dict "ItemsToPrint" must be the same as the entries in the list "PrintOrder".
        
        -The keys in the dict "ItemsToPrint" must be the same as the keys in "registerValue" in the method "extract".
        
        To add registers, simply expand the "ItemsToPrint" dict and add the name in "PrintOrder". 
        Then in the dict "registerValue" in the method "extract", add the same key and a path to the desired register (result must be a list).
        

        """
        self.PrintOrder = ["delay", "beta", "alpha", "interm_prob", "prob_before_table", "modprob", "CTV", "outer_delay", "outer_ctv", "Marker_pv_1", "Marker_pv_2", "PacketCount_ingress","PacketCount_Egress" ,"Counter Index", "Dropped Packets"]
        ItemsToPrint = {
            "delay"                 : ["Delay in (inner) Egress: {:s} ms", (lambda x: str(x[4][x[0]][x[0]]/x[2]))],
            "beta"                  : ["Result from beta: {:s} ms",(lambda x: str((x[4][x[0]][x[0]] - (x[3] if x[4][x[0]][x[0]] >= int("0X80000000",0) else 0))/x[2]))],
            "alpha"                 : ["Result from alpha: {:s} ms", (lambda x: str((x[4][x[0]][x[0]] - (x[3] if x[4][x[0]][x[0]] >= int("0X80000000",0) else 0))/x[2]))],
            "interm_prob"           : ["Probability after beta and alpha: {:s} ms" ,(lambda x: str((x[4][x[0]][x[0]] - (x[3] if x[4][x[0]][x[0]] >= int("0X80000000",0) else 0))/x[2]))],
            "prob_before_table"     : ["Probability before Div2square table: {:s} ms" ,(lambda x: str((x[4][x[0]][x[0]] - (x[3] if x[4][x[0]][x[0]] >= int("0X80000000",0) else 0))/x[2]))],
            "modprob"               : ["Probability after Div2square {:s} ms",(lambda x: str(x[4][x[0]][x[0]]/x[2]))],
            "CTV"                   : ["CTV in the (outer) Egress: {:s}",(lambda x: str(x[4][x[0]][x[0]]))],
            "outer_delay"           : ["Recoded delay in the (outer) Egress: {:s} ms", (lambda x: str(x[4][x[0]][x[0]]/x[2]))],
            "outer_ctv"             : ["CTV after CTV table: {:s}", (lambda x: str(x[4][x[0]][x[0]]))],
            "Marker_pv_1"           : ["Packet value for Gold from the marker: {:s}",(lambda x: str(x[4][x[0]][x[0]]))],
            "Marker_pv_2"           : ["Packet value for Silver from the marker: {:s}",(lambda x: str(x[4][x[0]][x[0]]))],
            "PacketCount_ingress"   : ["Number of packets that passed the ingress since last counter reset: {:s}", (lambda x: str(x[4][x[0]][x[0]]))],
            "PacketCount_Egress"    : ["Number of packets that passed the Egress since last counter reset: {:s}", (lambda x: str(x[4][x[0]][x[0]]))],
            "Counter Index"         : ["Latest counter index to be incremented: {:s}", (lambda x: str(x[4][x[0]][x[0]]))],
            "Dropped Packets"       : ["Number of dropped packets in the Egress: {:s}",(lambda x: str(x[4][x[0]][x[0]]))]
        } 
        self.Registers = {}
        for key in ItemsToPrint:
            self.Registers[key] = P4register(name = key , Toprint = ItemsToPrint[key][0], function = ItemsToPrint[key][1], parameters = [0,1,1000000, 2**32])    

    def extract(self):
        """
        Gets the value of each register. See Instructions on how to use the "registerValue" dict in the constructor.

        """
        self.registerValue = {
            "delay"                 : list(p4.SwitchEgress.EgressProcessingInstance.debug5.get(from_hw=1,register_index = Pv_ports[0],print_ents = 0).data.values()),
            "beta"                  : list(p4.SwitchEgress.EgressProcessingInstance.debug2.get(from_hw=1,register_index = Pv_ports[0],print_ents = 0).data.values()),
            "alpha"                 : list(p4.SwitchEgress.EgressProcessingInstance.debug3.get(from_hw=1,register_index = Pv_ports[0],print_ents = 0).data.values()),
            "interm_prob"           : list(p4.SwitchEgress.EgressProcessingInstance.debug6.get(from_hw=1,register_index = Pv_ports[0],print_ents = 0).data.values()),
            "prob_before_table"     : list(p4.SwitchEgress.EgressProcessingInstance.debug7.get(from_hw=1,register_index = Pv_ports[0],print_ents = 0).data.values()),
            "modprob"               : list(p4.SwitchEgress.EgressProcessingInstance.debug4.get(from_hw=1,register_index = Pv_ports[0],print_ents = 0).data.values()),
            "CTV"                   : list(p4.SwitchEgress.CTV.get(from_hw=1,register_index = Pv_ports[0],print_ents = 0).data.values()),
            "outer_delay"           : list(p4.SwitchEgress.Delay.get(from_hw=1,register_index = Pv_ports[0], print_ents = 0).data.values()),
            "outer_ctv"             : list(p4.SwitchEgress.EgressProcessingInstance.debug1.get(from_hw=1,register_index = Pv_ports[0],print_ents = 0).data.values()),
            "Marker_pv_1"           : list(p4.SwitchIngress.MarkerCTV.get(from_hw=1,register_index = 1,print_ents =0).data.values()),
            "Marker_pv_2"           : list(p4.SwitchIngress.MarkerCTV.get(from_hw=1,register_index = 2,print_ents =0).data.values()),
            "PacketCount_ingress"   : list(p4.SwitchIngress.test3.get(from_hw=1,register_index = 0,print_ents =0).data.values()),
            "PacketCount_Egress"    : list(p4.SwitchEgress.debug4.get(from_hw=1,register_index = 0,print_ents = 0).data.values()),
            "Counter Index"         : list(p4.SwitchIngress.test4.get(from_hw=1,register_index = 0,print_ents =0).data.values()),
            "Dropped Packets"       : list(p4.SwitchEgress.debug2.get(from_hw=1,register_index = 0,print_ents = 0).data.values())
        }

    def PrintAll(self):
        """
        Gets all register values and prints all their values.
        """
        self.extract()
        for name in self.PrintOrder:
            self.Registers[name].PrintRegisterValue(self.registerValue[name])

    def PrintOneOnce(self, name):
        """
        Gets all register values and prints the one specified.

        Args:
            name (string): Name of register value to print.
        """
        self.extract()
        if(name in self.Registers):
            self.Registers[name].PrintRegisterValue(self.registerValue[name])

    def PrintOneX(self, name, numberOfTimes):
        """
        Get all register values and prints the one specified. This is done a specified amount of times.

        Args:
            name (string): Name of register value to print.
            numberOfTimes (int): Number of times to print the value
        """
        for _ in range(numberOfTimes):
            self.extract()
            if(name in self.Registers):
                self.Registers[name].PrintRegisterValue(self.registerValue[name])
    
    def Record(self, names):
        """
        Get all register values and return it after lambda processing. Do this for all names specified. 

        Args:
            names (list): List of keys to process and return.

        Returns:
            list: List of floats containing register values post lambda processing. 
        """
        value = []
        self.extract()
        for name in names:
            if(name in self.Registers):
                value.append(self.Registers[name].ReturnRegisterValue(self.registerValue[name]))
        return value

# Debug object to be used in bfrt_python.
debug = Debugger()
stop = False

# Function to be used in bfrt_python.
def status():
    global debug
    debug.PrintAll()

# Function to be used in bfrt_python.
def spam(name):
    global debug
    debug.PrintOneX(name, 100)

# File path and variables used for evaluation and debugging.
timeCSVFileName = "" 
ToggleTimeRecorder = False
path = '/home/tofino/projects/Love_project/transfer/files/'

def Reset(test = False):
    """
    Creates new recording files depending on the global path and name.

    Args:
        test (bool, optional): Write n_pvs into one file, for debugging. Defaults to False.
    """
    global timeCSVFileName
    global path
    global n_pvs
    p4.SwitchIngress.test3.mod(register_index=0,f1=0)
    with open(path + "time{:s}.csv".format(timeCSVFileName), 'w+') as f:
        pass
    with open(path + "ecdf{:s}.txt".format(timeCSVFileName), 'w+') as f:
        pass
    with open(path + "actualECDF{:s}.txt".format(timeCSVFileName), 'w+') as f:     
        if(test):
            f.write(str(n_pvs) + "\n")

def RegistersRecord(timeToRecord = 30, FileName = ""):
    """
    Records the status of the Registers listed in the "Debugger" class. 

    - The "names" list represents the registers that should be recorded. Each name represents one column in the resulting file.
    - The "names" list has to be identical to the list "order" in the function "sortEntries" in the file "configure_register_data.py".

    Args:
        timeToRecord (int, optional): Specifies to the time of the recording. Defaults to 30s.
        FileName (str, optional): Specifies the name of the file.
    """

    global time
    global Reset
    global debug, ToggleTimeRecorder, timeCSVFileName, path

    names = ["delay","CTV", "Marker_pv_1", "Marker_pv_2", "PacketCount_ingress", "interm_prob", "prob_before_table", "modprob", "Counter Index"]
    timeCSVFileName = FileName
    ToggleTimeRecorder = True
    Reset(test =  True)
    Storage = np.delete(np.zeros(shape = (1,len(names) + 1)), 0, axis = 0)
    time_start = time.time()
    while(timeToRecord > time.time() - time_start):
        row = debug.Record(names)
        row.append(time.time() - time_start)
        Storage = np.append(Storage, [row], axis = 0)
    ToggleTimeRecorder = False
    np.savetxt(path + "Recording{:s}.csv".format(timeCSVFileName), Storage, delimiter=",")


# END DEBUG
################

TableFilled = False
def FillTables(Use_pv_filter = True):
    """
    The function performs all initialization that the Data Plane needs, namely:

    - It fills the forwarding table, specifies what ports that should use PV filtering and empties registers for debugging.

    - It fills the counter offset, alpha, beta and div_square and GetCTV tables.

    Args:
        Use_pv_filter (bool, optional): Set False to deactive PV filtering for all ports. Defaults to True.

    Returns:
        (list, list): The predetermined probability values and a snapshot of the empty counters. These are used during run-time.
    """
    global ip_address
    global Generate_alpha_beta, Generate_Div2_Square, GetEcdfStartValues, Generate_counter_table
    global p4, Ports, Pv_ports, Ip_addr, Queues, n_pvs, squareTableAccuracy,TableFilled, PvRange, alpha, beta

    # Fill forwarding table
    for i in range(len(Ports)):
        p4.SwitchIngress.forward.add_with_hit(Ip_addr[i], Ports[i])

    #Fill table with packet value active ports 
    if(Use_pv_filter):
        for port in Pv_ports:
            p4.SwitchIngress.Pv_port.add_with_accept(port)

    # Initialize registers 
        for port in Pv_ports: 
            p4.SwitchIngress.IngressProcessingInstance.stateMachine.mod(
                register_index=port, state=0, active=0)
            p4.SwitchEgress.CTV.mod(register_index = port, f1 = 0)

    # Fill counter offset table
    CounterSize = n_pvs 
    Keys, Bit_mask = Generate_counter_table(CounterSize)
    for j in range(len(Pv_ports)):
        for i in range(len(Keys)): 
            p4.SwitchIngress.Update_pvHistogram.add_with_update_hist(ucast_egress_port = Pv_ports[j],
            identification = Keys[i],
            identification_mask = Bit_mask,
            offset = i + j * CounterSize)

    # Fill the multiplication tables for alpha and beta
    Keys, Bit_mask, Value_Variants = Generate_alpha_beta(N = 32, m = 8, alpha = alpha, beta = beta)
    for variant in Value_Variants:
        for i in range(len(Keys)):
            if(variant.name == "alpha"):
                p4.SwitchEgress.EgressProcessingInstance.AlphaMult.add_with_AlphaHit(
                    interm_alpha = int(Keys[i],2), interm_alpha_mask = int(Bit_mask[i],2), match_priority = 0, result = variant.Entries[i])
            elif (variant.name == "beta"):
                p4.SwitchEgress.EgressProcessingInstance.BetaMult.add_with_BetaHit(
                    interm_beta = int(Keys[i],2), interm_beta_mask = int(Bit_mask[i],2), match_priority = 0, result = variant.Entries[i])

    # Fill square_div table
    Keys, Bit_mask, Value = Generate_Div2_Square(N = 32, m = squareTableAccuracy)
    for i in range(len(Keys)):
        p4.SwitchEgress.EgressProcessingInstance.ProbModification.add_with_Div2Square(
            prob_before_table = int(Keys[i],2), prob_before_table_mask = int(Bit_mask[i],2), match_priority = 0, result = Value.Entries[i])

    # Prep discrete probabilities to return
    prob_list = list(set(Value.Entries)) #Remove duplicates
    prob_list.sort()

    # Initialize GetCTV table
    pv_list = GetEcdfStartValues(prob_list = prob_list, N = 32, m = 9, maxRange = PvRange)
    for i in range(len(Pv_ports)):
        for j in range(len(Queues)):
            for k in range(len(prob_list)):
                p4.SwitchEgress.EgressProcessingInstance.GetCTV.add_with_CTV(prob = prob_list[k], result = pv_list[k])
    emptyCounters = p4.SwitchIngress.pv_histograms.dump(json=1)
    
    TableFilled = True
    return prob_list, emptyCounters

def cb(a,b,c,d):
    """
    Empty callback function.
    """
    pass

def updateDistBatch(emptyCounters):
    """
    Reads from the counter array in the Data Plane and then reset it in a batch.

    Args:
        emptyCounters (list): A json snapshot of the empty counters.

    Returns:
        list : List containing the counter values.
    """

    global np, time
    global Pv_ports, Queues, n_pvs, p4

    newDist = np.zeros((len(Pv_ports), len(Queues), n_pvs))
    p4.SwitchIngress.pv_histograms.operation_counter_sync(callback=cb)

    for i in range(len(Pv_ports)):
        for j in range(len(Queues)):
            newDist[i][j] = [x.data[b'$COUNTER_SPEC_BYTES'] for x in p4.SwitchIngress.pv_histograms.get(regex=True, print_ents=False)[(i*len(Queues)+j)*n_pvs:(i*len(Queues)+j+1)*n_pvs]]
    p4.SwitchIngress.pv_histograms.add_from_json(emptyCounters)
    return newDist

def ecdf(dist):
    """
    
    Creates an ECDF from the given list. The function can be simplfied and is not very efficient.

    Args:
        dist (list): List to be turned in to an ECDF.

    Returns:
        list: The ECDF.
    """

    global np
    global Pv_ports, Queues, n_pvs
    
    ecdfArray = np.zeros((len(Pv_ports), len(Queues), n_pvs))
    part_sum = 0
    for index, x in np.ndenumerate(dist):
        if(index[2] == 0):
            part_sum = 0
            total_sum = np.sum(dist[index[0]][index[1]])
        if(total_sum != 0):
            part_sum = part_sum + x
            percent = part_sum/total_sum
            ecdfArray[index] = percent
    return ecdfArray

def matchProbToPv(prob_list, ecdfArray):
    """
    Creates and returns the PV list to be used with the probability list. It is created from the given ECDF and probability list.

    Args:
        prob_list (list): list of probability values.
        ecdfArray (list): the ECDF.

    Returns:
        list: The PV list.
    """
    global bisect, np
    global Pv_ports, Queues, counter_gran

    pv_list = np.zeros((len(Pv_ports), len(Queues), len(prob_list[0][0])))
    for index, x in np.ndenumerate(prob_list):
        if(np.sum(ecdfArray[index[0]][index[1]]) == 0):
            pv_list[index] = 0
            return []
        else:
            pv_list[index] = bisect.bisect_left(ecdfArray[index[0]][index[1]], x) * counter_gran
    return pv_list

def writeIECDFTable(prob_list, pv_list, variant, eval_dummy_value): 
    """
    The function writes the given probability and PV list to the table in the Data Plane.
    Variant = 2 is used for evaluation purposes. 

    Args:
        prob_list (list): the list containing the probability.
        pv_list (list):  the list containing the PVs.
        variant (int): Variable used for evaluation if set to 2.
        eval_dummy_value (list): Dummy values to be used of variant is set to 2.
    """
    if(pv_list != []):
        for i in range(len(Pv_ports)):
            for j in range(len(Queues)):
                for k in range(len(prob_list[i][j])):
                    p4.SwitchEgress.EgressProcessingInstance.GetCTV.mod_with_CTV(prob = prob_list[i][j][k], result = pv_list[i][j][k])
    elif(variant == 2):
        for i in range(len(Pv_ports)):
            for j in range(len(Queues)):
                for k in range(len(prob_list[i][j])):
                    p4.SwitchEgress.EgressProcessingInstance.GetCTV.mod_with_CTV(prob = prob_list[i][j][k], result = eval_dummy_value[i])


def WriteToFile(prob_list, pv_list,ecdf, time):
    """
    Write evaluation values to file, namely the probability and PV list as well as the ECDF.

    Args:
        prob_list (list): the list containing the probability.
        pv_list (list):  the list containing the PVs.
        ecdf (list): The ECDF.
        time (int): Current time-stamp.
    """
    global os, bisect, np
    global Queues, Pv_ports, n_pvs, timeCSVFileName, path
    if(time <= timeToRecord):
        with open(path + "ecdf{:s}.txt".format(timeCSVFileName), 'a') as f:
            f.write("time: " + str(time) + "\n")
            for i in range(len(Pv_ports)):
                for j in range(len(Queues)):
                    for k in range(len(prob_list[i][j])):
                        if(pv_list != []):
                            f.write(str(prob_list[i][j][k]) + "," + str(pv_list[i][j][k]) + "\n")
                        else:
                            f.write(str(prob_list[i][j][k]) + "," + "0"+ "\n")
        with open(path + "actualECDF{:s}.txt".format(timeCSVFileName), 'a') as f:
            f.write("time: " + str(time) + "\n")
            for i in range(len(Pv_ports)):
                for j in range(len(Queues)):
                    for k in range(len(ecdf[i][j])):
                        f.write(str(ecdf[i][j][k]) + "\n")
        
def iecdf(prob_list, emptyCounters, loopTime, variant, eval_dummy_value):
    """
    Main function for the Control Plane control loop. Calls other functions in order to:

    - Create the ECDF from the counter array in the Data Plane.

    - Invert the ECDF to a PV list.

    - Writes the values to the Data Plane.

    Args:
        prob_list (list): Predetermined list of possible probabilities.
        emptyCounters (list): Json snapshot of the empty counters.
        loopTime (int): Time for evaluation purposes.
        variant (int): For evaluation purposes. If set to 2, the control loop does not create nor invert the ECDF and only writes dummy values to the Data Plane.  
        eval_dummy_value (list): For evaluation purposes. If variant is set to 2, write this to the Data Plane.
    """
    global np, time
    global ecdf, matchProbToPv, writeIECDFTable, WriteToFile, updateDistBatch
    global p4, Ports,Pv_ports, Queues, n_pvs, timeCSVFileName, ToggleTimeRecorder, TestMode
    
    pv_list = []
    ecdfArray = []
    newDist = updateDistBatch(emptyCounters)
    if(variant == 1):
        ecdfArray = ecdf(newDist)
        pv_list = matchProbToPv(np.divide(prob_list, 1000000), ecdfArray)
    writeIECDFTable(prob_list, pv_list, variant, eval_dummy_value)
    if(TestMode):
        WriteToFile(prob_list, pv_list, ecdfArray, time.time() - loopTime)

def Stop():
    """
    Manually stop the control loop. The control loop cannot be started without restarting the Control Plane entierly.
    """
    global loopVar
    loopVar = False

counter = 2
def record(startTime, recordTime, variant):
    """
    Records control loop cycle times for evaluation purposes.

    Args:
        startTime (int): The time the evaluation started.
        recordTime (int): Current time.
        variant (int): Used to seperate two different types of control loop cycles (depreciated).
    """
    global cb
    global p4, Pv_ports, counter, timeCSVFileName, path, n_pvs
    if(time.time() - startTime <= 10):
        with open(path + "time{:s}.csv".format(timeCSVFileName), 'a') as f:
            if(counter == 1 and variant == 1):
                f.write("New-Variant 1:" + str(squareTableAccuracy) + "," + str(n_pvs) + "\n")
                counter = 2
            elif(counter == 2 and variant == 2):
                f.write("New-Variant 2:" + str(squareTableAccuracy) + "," + str(n_pvs) + "\n")
                counter = 3
            else:
                f.write(str(time.time() - startTime) +  "," + str(recordTime) + "\n")

def loop(Use_pv_filter = True):
    """
    The control loop (runs until "Stop" function is called). Start this function to initilize the Data Plane and give current PV statistics to the Data Plane on a cycle basis. 

    - First all data structures of the Data Plane are initilized.

    - If evaluation or debug mode is active, set approprite values.

    - Run the ECDF loop.

    Args:
       Use_pv_filter (bool, optional): Set False to deactive PV filtering for all ports. Defaults to True.
    """
    
    global time
    global FillTables,iecdf, record
    global ToggleTimeRecorder,loopVar, debug, TestMode, EvalMode, timeToRecord

    prob_list_pre, emptyCounters = FillTables(Use_pv_filter)
    prob_list = np.zeros((len(Pv_ports),len(Queues),len(prob_list_pre)))
    for i in range(len(Pv_ports)):
        for j in range(len(Queues)):
            for k in range(len(prob_list_pre)):
                prob_list[i][j][k] = prob_list_pre[k]
    variant = 1
    if(EvalMode):
        eval_dummy_value = [ i for i in range(len(prob_list[0][0]))]
        Reset()
        variant = 2
    else:
        eval_dummy_value = []
    if(EvalMode or TestMode):
        print("Tables filled")
    prev_time = time.time()
    startTime = time.time()
    loopVar = True
    while(loopVar):
        prev_time = time.time()
        iecdf(prob_list, emptyCounters, startTime, variant, eval_dummy_value)
        if(EvalMode):
            record(startTime, time.time() - prev_time, variant)

loopVar = True
TestMode = True
EvalMode = True
timeToRecord = 10
def startCP(Testmode = False, FileName = "", Evalmode = False,  Use_pv_filter = True, TimeToRecord = 10):
    """
    Starts the control loop. Call this function manually from the run-time enviornment.

    Args:
        Testmode (bool, optional): Set True if debug information should be written to file. This will make the Control Plane work slower since inefficient threads are used. Defaults to False.
        FileName (str, optional): Name of the debug and/or evaluation files. Defaults to "".
        Evalmode (bool, optional): Set True if evaluation information should be written to file. This will make the Control Plane work slower since inefficient threads are used. Defaults to False.
        Use_pv_filter (bool, optional): Set to False to deactive PV filtering on all ports (deactivate AQM essentially). Defaults to True.
        TimeToRecord (int, optional): Time to record the debug and/or evaluation information. Defaults to 10.
    """
    global Thread, time
    global RegistersRecord, loop, Reset
    global loopVar, timeCSVFileName, TestMode, EvalMode, timeToRecord

    timeToRecord = TimeToRecord
    EvalMode = Evalmode
    TestMode = Testmode
    loopVar = False
    timeCSVFileName = FileName
    Thread(target=loop, args = (Use_pv_filter, )).start()
    if(TestMode):
        Reset()
        time.sleep(8)                                                                           # This is a cheap way of timing with the time it takes for filling the Data Plane tables. Can be removed or lowered for low "m" and "c"
        print("start recording")
        if(FileName != ""):
            RegistersRecord(FileName = FileName, timeToRecord = TimeToRecord)
