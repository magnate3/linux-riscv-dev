##################################################################################
 # A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 # Karlstad University 2021.
 # Author: L. Dahlberg
##################################################################################

import math
import os
import numpy as np
import bisect
import json

def Generate_keys(N = 32, m = 6, debug = False, signed = True):
    """ 
    This function generates all permutations of signed or unsigned N bits with m accuracy. It also creates a bit mask for respective bit pattern. 
    If N = m, all permuations will be generated. Set debug == True for debug info.\n

    - returns the list of permutations and the bit_masks \n

    This solution is based on a string generation pattern given by the authors of the paper 
    "Evaluating the Power of Flexible Packet Processing for Network Resource Allocation", section 3.2.
    This function modifies their solution to work with the negative and positive side of signed integers.   

    BUG:
        Does not work for smaller values such as N = 3, m = 2.
     
    
    """
    
    def Generate_string(N, m, n, value, bit_mask = False):
        """
        This function generates a bit string of value given N, m, and n. The value and resulting bit string is always positive. 
        If bit_mask = True: instead calculate the given bit mask required for given N, m and n.

        - returns the generated string 

        @Autor L. Dahlberg

        """
        Container = ""
        if(not bit_mask):
            Container += "0"*(n+1)                                                                          # The (n+1) is used as a correction, so that positive and negative numbers do not mix.
            Container += "1"
            Container += "{0:b}".format(value).zfill(min(m-2, N-n-2))                                       # The source of the generated bit pattern. "Add the bit pattern of i filled with 0s".
            Container += "0"*max(0, N-n-m)                                                                  # Fill with 0s behind the value if needed.
        else:                                                                                               # bit mask, put 1 in the positions that should match and 0s as don't care
            Container += "1"*(n+2)                                                                                                                                                         
            Container += "1"*min(m-2, N-n-2)
            Container += "0"*max(0, N-n-m)

        return Container
    
    if(signed):
        variants = 2
    else:
        variants = 1                                                             
    bit_mask = (1 << N) - 1                                                                                 # Used with bit wise "and" to convert positive bits to negative.
    modify_length = [N + 2, N + 3]                                                                          # Used to represent the different length of negative and positive bits(extra "-" makes negative number larger).
    discard = 0                                                                                             # Is set to 1 when a number is to be discarded. This is necessary at the moment because the patterns generate invalid numbers. 
    bit_mask_list = []                                                                                      # The list that holds the final bit mask list.
    final_list = []                                                                                         # The list that holds the final result.
    if(debug):
        Proportion_counter = 0                                                                              # Used for debug: counts proportins of positive and negative numbers.
    for j in range(variants):                                                                               # One iteration for positive and negative numbers( j == 0 for positive and j == 1 for negative)
        for n in range(N):                                                                                  # One iteration for each bit position.
            if(j==0):   
                intermediate_pattern = ["0b" for i in range(2**min(m-1, N-n-1))]                            # Create a list for the positive numbers with given size.
            else:
                intermediate_pattern = ["-0b" for i in range(2**min(m-1, N-n-1))]                           # Create a list for the negative numbers with given size.
            intermediate_bit_mask_list = ["0b" for i in range(2**min(m-1, N-n-1))]                          # Create an intermediate bit mask list.
            for i in range(2**min(m-1, N-n-1)):                                                             # 2**min(m-1, N-n-1) is how many number will be generated for each n. 
                intermediate_pattern[i] += Generate_string(N, m, n, value = i)                              # Calls function that generates the given bit pattern
                intermediate_bit_mask_list[i] += Generate_string(N, m, n, value = i, bit_mask = True)       # Calls the function with bit_mask = True, to get the bit mask bit pattern
                if(len(intermediate_pattern[i]) > modify_length[j]):                                        # Captures numbers that are out of bounds (This part can be skipped if above algorithm is improved).
                    discard = 1
                    if(n == N - 1):
                        intermediate_pattern[i] = intermediate_pattern[i][0:modify_length[j]-1] + "1"       # Corrects the last number of the sequence.
                        intermediate_bit_mask_list[i] = intermediate_bit_mask_list[i][0:modify_length[0]]   # Corrects the last number of the bit mask sequence.
                        discard = 0
                if(j == 1):
                    intermediate_pattern[i] = str(bin((int(intermediate_pattern[i],2) & bit_mask)))         # Transform the generated bit pattern to its negative equivalent.
                if(discard == 0):
                    final_list.append(intermediate_pattern[i])                                              # Adds undiscarded strings to the final list.
                    bit_mask_list.append(intermediate_bit_mask_list[i])                                     # Adds undiscarded bit mask strings.
                    if(intermediate_pattern[i][2] == "0" and debug):                                        
                        Proportion_counter += 1                                                             # For counting number of 0s.
                discard = 0
        if(j==0):
            final_list.append("0b" + Generate_string(N, m, n = -1, value = 0))                              # Adds the largest negative number at the end of the positive numbers.
            bit_mask_list.append("0b" + Generate_string(N, m, n = 0, value = 0, bit_mask = True))
    if(debug):
        for i in range(len(final_list)):
            print(final_list[i], "\t\t", bit_mask_list[i])
        print("Total list size", len(final_list))
        print("Total bit_mask_list size", len(bit_mask_list))
        print("Proportion of 1s", len(final_list)- Proportion_counter)
        print("Proportion of 0s", Proportion_counter)
        print("Size of numbers", len(final_list[0]))
    return final_list, bit_mask_list


class Values:
    """
    This class is used to represent the values for respective keys. \n
    It contains: \n 
    - A list of entries containing the values, the size of the given key list length.
    - A name (ex alpha or beta).
    - A constant to be used for beta or alpha (ex 5/16).

    @Autor L. Dahlberg

    """
    Entries = []
    def __init__(self, name, Constant, Keys_len):
        self.name = name
        self.Constant = Constant
        self.Entries = [0 for i in range(Keys_len)]

def Generate_values_alpha_beta(Keys, N = 32, alpha = 5/16, beta = 50/16, debug = False):
    """
    This function generates the values for given keys as the result of alpha and beta multiplication.
    Set debug == True for debug info.\n 

    - returns a list containing two instances of the class Values, which contains the list of alpha or beta values respectively.

    The multiplication is done after doing the correct bit convertions (for positive and negative numbers respectively).
    Afterwards, the value is rounded to closest integer. The integer is put to 2**N - 1 if it is too large.

    @Autor L. Dahlberg  

    """

    Alpha_Values = Values("alpha", alpha, len(Keys))                                                        # The instantiation of the class Values for alpha.
    Beta_Values = Values("beta", beta, len(Keys))                                                           # The instantiation of the class Values for beta.
    Value_Variants = [Alpha_Values, Beta_Values]                                                            # A list containing both classes, for looping purposes.
    constant = 1000000

    for Value in Value_Variants:
        for i in range(len(Keys)):
            if(Keys[i][2] == "1"):                                                                          # If the number is negative
                Value.Entries[i] = min(2**N - 1,round((((int(Keys[i],2) - 2**N) / constant) * Value.Constant) * constant)) # Transfer to integer (it is a now big and positive) and correct it with - 2^N (making it negative) then apply appropriate constants.
                if(debug):
                    print(Keys[i], "\t", Value.name, "\t Conversion: ", int(Keys[i],2) - 2**N,
                     "\t After mult: ", (int(Keys[i],2) - 2**N)* Value.Constant,
                      "\t After mult and round: ", Value.Entries[i])
            else:                                                                                           # else if number is positive
                Value.Entries[i] = min(2**N - 1 ,round((((int(Keys[i],2)) / constant) * Value.Constant) * constant))   # Same as previous entry, but without 2^N correction. 
                if(debug):
                    print(Keys[i], "\t", Value.name, "\t Conversion: ", int(Keys[i],2), 
                    "\t After mult: ", int(Keys[i],2)* Value.Constant,
                     "\t After mult and round: ", Value.Entries[i])
    return Value_Variants
    

def Generate_alpha_beta(N = 32, m = 6, alpha = 5/16, beta = 50/16, debug = False):
    """
    This function generates the key-bitmask-value lists for alpha and beta. 
    They are supposed be used to approximate multiplication on the Tofino switch using ternary table matches.
    Set debug = True for debug info. \n  

    - returns the Key list, the bit mask list and a list containing two instances of the class Values, 
    which contains the value list of alpha or beta values respectively.

    @Autor L. Dahlberg

    """
    
    Keys, Bit_mask = Generate_keys(N,m)
    Value_Variants = Generate_values_alpha_beta(Keys, N, alpha, beta)
    
    if(debug):
        big_values = 0
        for variant in Value_Variants:
            print("\n", variant.name)
            for i in range(len(Keys)):
                print(Keys[i],"\t", Bit_mask[i],  "\t", variant.Entries[i])
                if(variant.Entries[i] >= 2**N):
                    big_values += 1
        for variant in Value_Variants:
            print("Total list size", variant.name, len(variant.Entries))
        print(big_values)
    return Keys, Bit_mask, Value_Variants

def DEBUG_writeToFile_Generate_alpha_beta(N = 32, m = 8, file_name = "test_Generate_alpha_beta.txt"):
    """
       Debug function used for testing Generate_alpha_beta. 

        
    """
    Keys, Bit_mask ,Value_Variants = Generate_alpha_beta(N, m)
    with open(file_name, "w") as filehandle:
        for i in range(len(Keys)):
            filehandle.write("%s\t\t" % Keys[i])
            filehandle.write("%s" % Bit_mask[i])
            filehandle.write("\t\t%s\n" % Value_Variants[1].Entries[i])                                     # Uses beta values.
    print(file_name, "written to", os.getcwd())


def Generate_keys_Div2_Square(N = 32, m = 9, debug = False):
    """
    This function generates all permutations of signed N bits with m accuracy used by the division by two squared operation. 
    
    The function calls Generate_keys(signed = False) and modifies its output as is needed for the operation.

    - returns the list of permutations and the bit_masks \n

    TODO:
        The function currently returns a Key list that has multiple entries that all results in max_value.
        Optimally, it should catch all values that are "too large" under one entry. 

     
    """
    
    
    Keys, Bit_mask = Generate_keys(N,m, signed = False)                                                      
    Bit_mask[-1] = "0b10000000000000000000000000000000"                                                     # Modify the last bit to catch all negative numbers
    Keys.append("0b00000000000000000000000000000000")                                                       # Adds 0 to the key list, to remove misses in p4 debugging
    Bit_mask.append("0b11111111111111111111111111111111")                                                   
    return Keys, Bit_mask

def Generate_values_Div2_Square(Keys, N = 32, operation = "(x/2)^2", max_value = 1000000, debug = False):
    """
    This function generates the values for given keys as the result division by two squared.
    Set debug == True for debug info.\n 

    - returns an instance of the class Values, which contains the list of values.

    The operation is done after doing converting the keys from nanoseconds to miliseconds.
    After the operation, the value is converted to nanoseconds and then it is rounded to closest integer. 
    The integer is put to max_value if it is too large.

    @Autor L. Dahlberg  
    """
    
    Square_Values  = Values(operation, 0, len(Keys))
    constant = 1000000
    for i in range(len(Keys)):
        if(Keys[i][2] == "1"):                                                                              # If the number is negative
            Square_Values.Entries[i] = 0
        else:                                                                     
            Square_Values.Entries[i] = min(max_value,round(                                                 # round the generated integer and set it to max_value if too large.
                                                            (
                                                                (
                                                                    (
                                                                        int(Keys[i],2) / constant           # Apply appropriate constant.
                                                                    )/2         
                                                                )**2
                                                            )*constant))
        if(debug):
            print(Keys[i], "\t", Square_Values.name, "\t Conversion: ", int(Keys[i],2),
                "\t After div2 and square: ", (int(Keys[i],2)/2)**2 ,
                "\t After mult and round: ", Square_Values.Entries[i])
    return Square_Values

def Generate_Div2_Square(N = 32, m = 9, debug = False):
    """
    This function generates the key-bitmask-value lists for division by two squared. 
    They are supposed be used to approximate division by two squared operation on the Tofino switch using ternary table matches.
    Set debug = True for debug info. \n  

    - returns the Key list, the bit mask list one instances of the class Values, which contains the value list called Entries.

    @Autor L. Dahlberg

    """
    
    
    Keys, Bit_mask = Generate_keys_Div2_Square(N,m)
    Value = Generate_values_Div2_Square(Keys, N)
    if(debug):
        print("\n", Value.name)
        for i in range(len(Keys)):
            print(Keys[i],"\t", Bit_mask, "\t", Value.Entries[i])
        print("Total list size of operation ", Value.name," is: ", len(Value.Entries))
    return Keys, Bit_mask, Value 

def DEBUG_writeToFile_Generate_Div2_Square(N = 32, m = 9, file_name = "test_Div2_Square.txt"):
    """
       Debug function used for testing Generate_alpha_beta. 

        
    """
    Keys, Bit_mask ,Value = Generate_Div2_Square(N, m)
    with open(file_name, "w+") as filehandle:
        for i in range(len(Keys)):
            filehandle.write("%s\t\t" % Keys[i])
            filehandle.write("%s" % Bit_mask[i])
            filehandle.write("\t\t%s\n" % Value.Entries[i])
    print(file_name, "written to", os.getcwd())

def GetEcdfStartValues(prob_list, N = 32, m = 9, maxRange = 2**16,  length = 2**11):
    """
       This function generates a hard coded and generic ECDF. 
       The ECDF has the following structure:
            1%  of data is within 0-35%   of length
            13% of data is within 35-38%  of length    
            22% of data is within 38-41%  of length
            28% of data is within 41-44%  of length
            22% of data is within 44-47%  of length
            13% of data is within 47-50%  of length
            1%  of data is within 50-100% of length
        
        - returns a list of the invereted ECDF.

        
    """
    def fill(start,end,percent, total_sum, ecdf_list):
        for i in range(start,end):
            total_sum += (maxRange * percent) / (end - start)
            ecdf_list[i] = int(total_sum)
        return total_sum
    prob_list.sort()
    prob_list = np.true_divide(prob_list, 1000000)
    Probability = [0 for _ in range(length)]
    ecdf_list = [0 for _ in range(length)]
    pv_list = [0 for _ in prob_list]
    total_sum = 0
    total_sum = fill(0,int(0.35*length),0.01, total_sum, ecdf_list)
    total_sum = fill(int(0.35*length),int(0.38*length),0.13, total_sum, ecdf_list)
    total_sum = fill(int(0.38*length),int(0.41*length),0.22, total_sum, ecdf_list)
    total_sum = fill(int(0.41*length),int(0.44*length),0.28, total_sum, ecdf_list)
    total_sum = fill(int(0.44*length),int(0.47*length),0.22, total_sum, ecdf_list)
    total_sum = fill(int(0.47*length),int(0.50*length),0.13, total_sum, ecdf_list)
    total_sum = fill(int(0.50*length),length,0.01, total_sum, ecdf_list)
    for i in range(length):
        Probability[i] = ecdf_list[i] / maxRange
    for i in range(len(prob_list)):
        pv_list[i] = bisect.bisect_left(Probability, prob_list[i]) * N
        if(pv_list[i] >= maxRange):
            pv_list[i] = maxRange - 1
    return pv_list
    

def DEBUG_GetEcdfStartValues(N = 32, m = 9, maxRange = 2**16, file_name = "test_GetEcdfStartValues.txt"):
    """
       Debug function used for testing GetEcdfStartValues. 

    """
    Keys, Bit_mask, Value = Generate_Div2_Square(N, m)
    prob_list = list(set(Value.Entries))
    pv_list = GetEcdfStartValues(prob_list = prob_list, N = N, m = m, maxRange = maxRange)
    with open(file_name, "w+") as filehandle:
        for i in range(len(prob_list)):
            filehandle.write(str(prob_list[i]/1000000) + ",\t"+ str(pv_list[i]) + "\n")
    print(file_name, "written to", os.getcwd())

def getJsonOfCounter(size = 2**11):
    Data = []
    for i in range(size + 1):
        Entry = {
            "table_name" : "SwitchIngress.pv_histograms", 
            "data"       : {"$COUNTER_SPEC_BYTES": 0},
            "key"        : {"$COUNTER_INDEX": i}, 
            "action"     : "null"
        }
        Data.append(Entry)
    return json.loads(json.dumps(Data))

def Generate_counter_table(size = 2**11, maxRange = 2**16):
    """ 
    This function generates the counter index table. 

    - returns the list of counter indexes for one port and a bit mask.  

    """
    granularity = int(math.log(maxRange,2) - math.log(size,2))
    Bit_mask = int("0b" + "1"*int(math.log(size,2)) + "0"*granularity , 2)
    Keys = [int(maxRange / size) * i for i in range(size)]
    return Keys, Bit_mask

def DEBUG_Generate_counter_table(size = 2**11, maxRange = 2**16, file_name = "test_counterTable.txt"):
    """ 
    Debug function used for testing Generate_counter_table
    
     
    """
    Keys, Bit_mask = Generate_counter_table(size, maxRange)
    with open(file_name, "w+") as filehandle:
        for i in range(len(Keys)):
            filehandle.write("0b"+"{0:b}".format(int(bin(Keys[i]),2)).zfill(16) + "\t" + str(bin(Bit_mask)) + "\t" + str(i) + "\n")
    print(file_name, "written to", os.getcwd())

def GetTableSizesFor(N = 32, m = None, CounterSize = None, Pv_ports = [46], maxRange = 2**16, alpha = 0.125, beta = 1.25, CTVupdate = 5000000, refdelay = 2000000):
    """ 
    Help function used for determining Data and Control plane variable sizes. 
    
     
    """
    ControlPlaneVariables = []
    DataPlaneVariables = []

    if(CounterSize != None):
        DataPlaneVariables.append("The 'COUNTER_SIZE' variable in the Data Plane is {:s}".format(str(len(Pv_ports) * CounterSize + 1 )))
        ControlPlaneVariables.append("The 'n_pvs' variable in the Control Plane is {:s}".format("2^" + str(int(math.log(CounterSize,2)))))
    if(m != None):
        Keys, Bit_mask, Value = Generate_Div2_Square(N, m)
        prob_list = list(set(Value.Entries))
        prob_list.sort()
        DataPlaneVariables.append("The 'CTV_SIZE' variable in the Data Plane is {:s}".format(str(len(prob_list) + 1)))
        DataPlaneVariables.append("The 'PROB_SIZE' variable in the Data Plane is {:s}".format(str(len(Keys) + 1)))
        ControlPlaneVariables.append("The 'squareTableAccuracy' variable in the Control Plane is {:s}".format(str(m)))
    ControlPlaneVariables.append("The PvRange variable in the Control Plane is {:s}".format("2^" + str(int(math.log(maxRange,2)))))
    ControlPlaneVariables.append("The alpha variable in the Control Plane is {:s}".format(str(alpha)))
    ControlPlaneVariables.append("The beta variable in the Control Plane is {:s}".format(str(beta)))
    DataPlaneVariables.append("The 'CONTROL_INTERVAL' (CTV update interval) variable in the Data Plane is {:s} msec = {:s} µs = {:s} nsec".format(str(round(CTVupdate*10**-6,10)), str(round(CTVupdate*10**-3,10)), str(CTVupdate)))
    DataPlaneVariables.append("The 'REFERENCE_DELAY' variable in the Data Plane is {:s} msec  = {:s} µs = {:s} nsec".format(str(round(refdelay*10**-6,10)), str(round(refdelay*10**-3,10)), str(refdelay)))
    for variable in ControlPlaneVariables:
        print(variable)
    print("\n")
    for variable in DataPlaneVariables:
        print(variable)
    print("\n")
        

GetTableSizesFor(maxRange = 2**16 , alpha = 5/16, beta = 50/16, CTVupdate = 5000000, refdelay = 2000000 ,  m = 8, CounterSize = 2**10)

