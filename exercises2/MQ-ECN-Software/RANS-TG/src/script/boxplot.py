import sys
import os
import numpy
import matplotlib as mpl 
import re

## agg backend is used to create plot as a .png file
mpl.use('agg')

import matplotlib.pyplot as plt 

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


''' Parse a file to get FCT and goodput results '''
def parse_file(file_name):
    results = []
    f = open(file_name)
    while True:
        line = f.readline().rstrip()
        if not line:
            break
        arr = line.split()
        '''size, fct, dscp, sending rate, goodput'''
        if len(arr) >= 5:
            '''[size, fct, goodput]'''
            results.append([int(arr[0]), int(arr[1]), int(arr[4])])
    f.close()
    return results

''' Get average result '''
def average_result(input_tuple_list, index):
    input_list = [x[index] for x in input_tuple_list]
    print "hello: " + str(input_list)
    if len(input_list) > 0:
        return sum(input_list) / len(input_list)
    else:
        return 0

def box_plot(input_dirs, outputfile):

	
	# Create a figure instance
	fig = plt.figure(1)#, figsize=(9, 6))
	# Create an axes instance
	ax = fig.add_subplot(111)
	colors = ['r','g','b']
	linestyles=['-','--',':']
	xticks=[]
	# plt.ylim(0,1000000000)


	for log_dir in input_dirs:
		data_to_plot=[]
		final_results = []
		color=colors.pop(0)
		linestyle=linestyles.pop(0)
		for file in sorted(os.listdir(log_dir),key=alphanum_key):
			if file.endswith("flows.txt"):
				final_results=(parse_file(log_dir+file))
				input_list = [x[1] for x in final_results]
				if len(input_list) > 0:
					seed = int(file.split("_")[2])
					if seed==1:
						# if int(file.split("_")[1])>=400:
						data_to_plot.append(input_list)
						xticks.append(file.split("_")[1])
						# 	print input_list
						# 	exit()
		# Create the boxplot
		# bp = ax.boxplot(data_to_plot)
		# bp = ax.boxplot(data_to_plot, sym="", showmeans=True, meanline=True)
		bp = ax.boxplot(data_to_plot, whis=3.0)#, sym="")#, whis=100)#, patch_artist=True)
		ax.set_xticklabels(xticks)
		ax.set_ylabel("flow completion times (us)")
		ax.set_xlabel("load (Mbps)")

		# for a in ax.flatten():
		# ax.set_yscale('log')
		# ax.set_yticklabels([])

		## change outline color, fill color and linewidth of the boxes
		for box in bp['boxes']:
			# change outline color
			# box.set( color='#7570b3', linewidth=2)
			box.set( color=color, linewidth=2,linestyle=linestyle)#, alpha=0.5)
			# change fill color
			# box.set( facecolor = '#1b9e77' )





		## change color and linewidth of the whiskers
		for whisker in bp['whiskers']:
			# whisker.set(color='#7570b3', linewidth=2)
			whisker.set(color=color, linewidth=1,linestyle=linestyle)#,alpha=0.5)

		## change color and linewidth of the caps
		for cap in bp['caps']:
			# cap.set(color='#7570b3', linewidth=2)
			cap.set(color=color, linewidth=1,linestyle=linestyle)#,alpha=0.5)

		## change color and linewidth of the medians
		for median in bp['medians']:
			# median.set(color='#b2df8a', linewidth=2)
			median.set(color='b', linewidth=2)
			# median.set(color=color, linewidth=2,linestyle="-")#,alpha=0.5)

		## change the style of fliers and their fill
		for flier in bp['fliers']:
			# flier.set(marker='o', color='#e7298a', alpha=0.5)
			flier.set(marker='o', color=color, alpha=0.5)

	# Save the figure
	fig.savefig(outputfile, bbox_inches='tight',dpi=600)

def print_usage(prog_name):
	print "usage: "+ prog_name + " -d <dir1> -l <label1> -d <dir2> -l <label2> ... \n-o <outputfile> -x <xlabel> -y <ylabel>  -t <title>"

if __name__ == '__main__':
	input_dirs=[]
	input_labels=[]
	outputfile="plot.png"
	xlabel=""
	ylabel=""
	title=""
	if len(sys.argv) > 2 and "-d" in sys.argv:
		iterator = xrange(1,len(sys.argv)).__iter__()
		for arg_index in iterator:
			if sys.argv[arg_index] == "-d":
				iterator.next()	
				arg_index+=1
				input_dirs.append(sys.argv[arg_index])
			elif sys.argv[arg_index] == "-l":
				iterator.next()	
				arg_index+=1
				input_labels.append(sys.argv[arg_index])
			elif sys.argv[arg_index] == "-o":
				iterator.next()	
				arg_index+=1
				outputfile=sys.argv[arg_index]
			elif sys.argv[arg_index] == "-x":
				iterator.next()	
				arg_index+=1
				xlabel=sys.argv[arg_index]
			elif sys.argv[arg_index] == "-y":
				iterator.next()	
				arg_index+=1
				ylabel=sys.argv[arg_index]
			elif sys.argv[arg_index] == "-t":
				iterator.next()	
				arg_index+=1
				title=sys.argv[arg_index]
			else:
				print_usage(sys.argv[0])
				sys.exit()
	else:
		print_usage(sys.argv[0])
		sys.exit()

	if not os.path.exists(os.path.dirname(outputfile)):
		os.makedirs( os.path.dirname(outputfile) );

	box_plot(input_dirs, outputfile)
	