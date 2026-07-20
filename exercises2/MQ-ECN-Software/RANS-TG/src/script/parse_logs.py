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


def analyse_log(logfile, analysis_script, out_dir, seed, load):
	out = os.popen("python "+analysis_script+" "+logfile).read().split("\n")
	with open(out_dir+"/afct"+seed+".csv", 'a') as csvfile:
		csvfile.write(str(load))
		csvfile.write(",")
		csvfile.write(str(float(out[1])/1000.0)) #in ms
		csvfile.write("\n")
	with open(out_dir+"/std_dev"+seed+".csv", 'a') as csvfile:
		csvfile.write(str(load))
		csvfile.write(",")
		csvfile.write(str(float(out[2])/1000.0))
		csvfile.write("\n")
	with open(out_dir+"/tail"+seed+".csv", 'a') as csvfile:
		csvfile.write(str(load))
		csvfile.write(",")
		csvfile.write(str(float(out[3])/1000.0)) #in ms
		csvfile.write("\n")
	with open(out_dir+"/75percentile"+seed+".csv", 'a') as csvfile:
		csvfile.write(str(load))
		csvfile.write(",")
		csvfile.write(str(float(out[4])/1000.0)) #in ms
		csvfile.write("\n")
	with open(out_dir+"/median"+seed+".csv", 'a') as csvfile:
		csvfile.write(str(load))
		csvfile.write(",")
		csvfile.write(str(float(out[5])/1000.0)) #in ms
		csvfile.write("\n")
	with open(out_dir+"/goodput"+seed+".csv", 'a') as csvfile:
		csvfile.write(str(load))
		csvfile.write(",")
		csvfile.write(str(float(out[6]))) #in Mbps
		csvfile.write("\n")

def run_analysis(filehandle, log_dir, write_directory,folder):
	global analysis_script, bandwidth
	traffic = filehandle.split("_")[1]
	seed = filehandle.split("_")[2]
	if not os.path.exists(write_directory+"/"+folder):
		os.mkdir( write_directory+"/"+folder);
	analyse_log(log_dir+"/"+filehandle, analysis_script, write_directory+"/"+folder, seed, float(traffic)*100/bandwidth)


def average_over_seeds(startswith, analysis_dir):
	#afct
	loads_list=[]
	y_list=[]
	for file in sorted(os.listdir(analysis_dir),key=alphanum_key):
		loads=[]
		y=[]
		# print "%s" % str(file)
		if file.startswith(startswith):
			with open(analysis_dir+file, "r") as f1:
				for line in f1:
					loads.append(float(line.split(",")[0]))
					y.append(float(line.split("\n")[0].split(",")[1]))
					
			loads_list.append(loads)
			if len(loads_list)>0:
				assert(loads==loads_list[0]) #check loads across files are same
			y_list.append(y)


	y_arr=np.array(y_list)
	y_mean = np.mean(y_arr,axis=0)
	y_error = np.std(y_arr,axis=0)
	# avg_y= map(mean, zip(*y_list))

	if not os.path.exists(analysis_dir+"averages"):
		os.mkdir( analysis_dir+"averages" )

	for index in range(len(loads_list[0])):
		with open(analysis_dir+"averages/" +startswith+".csv", 'a') as csvfile:
			csvfile.write(str(loads_list[0][index]))
			csvfile.write(",")
			csvfile.write(str(y_mean[index]))
			csvfile.write("\n")
		with open(analysis_dir+"averages/" +startswith+"_error.csv", 'a') as csvfile:
			csvfile.write(str(loads_list[0][index]))
			csvfile.write(",")
			csvfile.write(str(y_error[index]))
			csvfile.write("\n")

if __name__ == '__main__':
	if len(sys.argv) is not 2:
		print "usage: "+ sys.argv[0] + " <logs dir>"
		sys.exit()
	else:
		print "parsing..."


	bandwidth=1000.0
	analysis_script="result.py"
	
	log_dir=sys.argv[1]
	write_directory=log_dir+"analysis/"
	if not os.path.exists(write_directory):
		os.mkdir( write_directory );
	else:
		os.system("rm -rf "+write_directory+"*")

	for file in sorted(os.listdir(log_dir),key=alphanum_key):
		if file.endswith("flows.txt"):
			run_analysis(file, log_dir, write_directory, "flows" )
			
		elif file.endswith("reqs.txt"):
			run_analysis(file, log_dir, write_directory, "reqs" )
			
	if os.path.exists(write_directory+"flows/"):
		average_over_seeds("afct",write_directory+"flows/")
		average_over_seeds("std_dev",write_directory+"flows/")
		average_over_seeds("tail",write_directory+"flows/")
		average_over_seeds("75percentile",write_directory+"flows/")
		average_over_seeds("median",write_directory+"flows/")
		average_over_seeds("goodput",write_directory+"flows/")
	if os.path.exists(write_directory+"reqs/"):
		average_over_seeds("afct",write_directory+"reqs/")
		average_over_seeds("std_dev",write_directory+"reqs/")
		average_over_seeds("tail",write_directory+"reqs/")
		average_over_seeds("75percentile",write_directory+"reqs/")
		average_over_seeds("median",write_directory+"reqs/")
		average_over_seeds("goodput",write_directory+"reqs/")

	print "parsing complete"
