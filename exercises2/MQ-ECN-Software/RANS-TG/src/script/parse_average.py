import os.path
import sys
import re
import numpy as np


#helper
#ref:nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

#ref:http://stackoverflow.com/questions/10919664/averaging-list-of-lists-python
def mean(a):
    return sum(a) / len(a)    
##############################################################################

if __name__ == '__main__':
	if len(sys.argv) is not 4:
		print "usage: "+ sys.argv[0] + " <logs dir> <prefix> <postfix - without extension>"
		sys.exit()

	result_script="result.py"
	log_dir=sys.argv[1]
	if sys.argv[3]=="reqs":
		write_directory=log_dir+"arct/"
	elif sys.argv[3]=="flows":
		write_directory=log_dir+"afct/"
	if not os.path.exists(write_directory):
		os.mkdir( write_directory );
	else:
		os.system("rm -rf "+write_directory+sys.argv[2]+"*.csv")

	

	read_dir=log_dir
	prev_load = -1
	run_number = -1
	load_number = -1
	load_avgs=[[0.0]*20,[0.0]*20,[0.0]*20]
	loads=[None]*20
	load=-1
	
	for file in sorted(os.listdir(read_dir),key=alphanum_key):
		if file.startswith(sys.argv[2]) and file.endswith(sys.argv[3]+".txt"):
			out = os.popen("python "+result_script+" "+log_dir+file).read().split("\n")
			# print out
			avg = out[1]
			std_dev = out[2]
			tail = out[3]
	
			load = file.split("_")[1]
			if int(load) != prev_load:
				prev_load=int(load)
				if load_number is not -1:
					load_avgs[0][load_number] = float(load_avgs[0][load_number])/run_number
					load_avgs[1][load_number] = float(load_avgs[1][load_number])/run_number
					load_avgs[2][load_number] = float(load_avgs[2][load_number])/run_number
				run_number = 1
				load_number += 1
				loads[load_number]=int(load)/10
			else:
				run_number += 1

			load_avgs[0][load_number] += int(avg)
			load_avgs[1][load_number] += int(std_dev)
			load_avgs[2][load_number] += int(tail)
	load_avgs[0][load_number] = float(load_avgs[0][load_number])/run_number
	load_avgs[1][load_number] = float(load_avgs[1][load_number])/run_number
	load_avgs[2][load_number] = float(load_avgs[2][load_number])/run_number
	loads[load_number]=int(load)/10
	
	with open(write_directory+sys.argv[2]+".csv", 'a') as csvfile:
		for load_index in range(load_number+1): 
		   csvfile.write(str(loads[load_index]))
		   csvfile.write(",")
		   csvfile.write(str(float(load_avgs[0][load_index])/1000)) #in ms
		   csvfile.write("\n")
 	with open(write_directory+sys.argv[2]+"_stdev.csv", 'a') as csvfile:
		for load_index in range(load_number+1): 
		   csvfile.write(str(loads[load_index]))
		   csvfile.write(",")
		   csvfile.write(str(float(load_avgs[1][load_index])/1000)) #in ms
		   csvfile.write("\n")

	with open(write_directory+sys.argv[2]+"_tail.csv", 'a') as csvfile:
		for load_index in range(load_number+1): 
		   csvfile.write(str(loads[load_index]))
		   csvfile.write(",")
		   csvfile.write(str(float(load_avgs[2][load_index])/1000)) #in ms
		   csvfile.write("\n")
